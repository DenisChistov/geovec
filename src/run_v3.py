"""Stack geographically clustered forecasts with disjoint inner/outer folds."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import KFold

from experiment import metrics, save_json, spherical_blend
from run_v2 import checked_prediction, pca_features

DATA = Path(os.environ.get("GEO_CV_ROOT", "/content/train250k-cv"))
OUT = DATA / "v3"
V2 = DATA / "v2"
SEED = 623748
SPLIT_SEED = 523748
REFERENCE = {"mae_km": 319.0814680967899, "rmse_km": 1065.55135816962}


def fingerprint():
    digest = hashlib.sha256()
    for name in ("dev_rows.npy", "test_rows.npy", "folds.npy"):
        digest.update((DATA / name).read_bytes())
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    target = path.with_suffix(".tmp.npz")
    np.savez(target, **arrays)
    target.replace(path)


def unit(v):
    return checked_prediction(np.asarray(v, dtype=np.float32))


def blend(a, b, weight):
    return unit(spherical_blend(a, b, weight))


def deep(key, X, Y, train, query, seed, save_path=None):
    # The inner folds intentionally reuse V2's deterministic split, so those
    # predictions already exclude precisely these query rows.
    previous = V2 / f"deep-{key}-residual-angle.npz"
    marker = V2 / "split.sha256"
    reuse = marker.exists() and marker.read_text().strip() == fingerprint()
    if save_path is None and reuse and previous.exists():
        with np.load(previous) as source:
            answer = source["xyz"]
        if len(answer) != len(query):
            raise ValueError("Cached DNN predictions have the wrong row count")
        return unit(answer)
    from deep_v2 import CONFIGS, fit_predict
    match = re.fullmatch(r"o(\d)(?:-i(\d))?", key)
    if match:
        fold = int(match[1])
        seed = SPLIT_SEED+fold if match[2] is None else SPLIT_SEED+100+fold*3+int(match[2])
    elif key == "final":
        seed = SPLIT_SEED+5
    return unit(fit_predict(X, Y, train, query, CONFIGS["residual-angle"],
                            seed, save_path=save_path))


def make_bundle(key, X, Y, train, query, seed, save=False):
    from posterior_v3 import fit_posterior
    from refine_v3 import refine

    path = OUT / f"bundle-{key}.npz"
    if path.exists() and not save:
        with np.load(path) as saved:
            return dict(saved)
    t0 = perf_counter()
    save_to = lambda name: OUT / name if save else None
    a = fit_posterior(X, Y, train, query, 512, seed, save_path=save_to("cells-512-a.pt"))
    b = fit_posterior(X, Y, train, query, 1024, seed+173, save_path=save_to("cells-1024.pt"))
    c = fit_posterior(X, Y, train, query, 512, seed+389, save_path=save_to("cells-512-b.pt"))
    d = deep(key, X, Y, train, query, seed+1, save_path=save_to("direct.pt"))
    ref = refine(X, Y, train, query, a, seed=seed)
    out = combine_predictions(a, b, c, d, ref)
    atomic_npz(path, **out)
    print(f"Bundle {key}: {perf_counter()-t0:.1f}s", flush=True)
    return out


def combine_predictions(a, b, c, d, ref):
    n = len(d)
    multiscale = blend(blend(a["pred"], c["pred"], .5), b["pred"], 1/3)
    sharper = blend(blend(a["t08"], c["t08"], .5), b["t08"], 1/3)
    softer = blend(blend(a["t12"], c["t12"], .5), b["t12"], 1/3)
    local = blend(multiscale, ref["refined_mean"], .33)
    hybrid = blend(ref["joint"], d, .5)
    experts = np.stack([a["pred"], multiscale, sharper, softer,
                        ref["refined_mean"], ref["refined_mode"],
                        hybrid], axis=1).astype(np.float32)
    # Every cell feature is rank-aligned, not indexed by arbitrary cluster ID.
    spread = [(experts[:, i] * experts[:, j]).sum(axis=1)
              for i in range(experts.shape[1]) for j in range(i+1, experts.shape[1])]
    features = np.column_stack([
        experts.reshape(n, -1), a["features"], b["features"],
        c["features"], ref["features"], ref["statistics"], *spread,
    ]).astype(np.float32)
    out = {
        "features": features, "experts": experts,
        "reference_512": a["pred"], "temp_06": a["t06"], "temp_08": a["t08"],
        "temp_12": a["t12"], "temp_16": a["t16"],
        "ensemble_512": blend(a["pred"], c["pred"], .5),
        "ensemble_multiscale": multiscale, "ensemble_sharper": sharper,
        "ensemble_softer": softer, "local_refinement": ref["refined_mean"],
        "local_mode": ref["refined_mode"], "local_blend": ref["refined_blend"],
        "ensemble_local": local, "ensemble_dnn_retrieval": blend(multiscale, hybrid, .25),
    }
    for name, value in out.items():
        assert np.isfinite(value).all(), f"Nonfinite data in {name}"
    return out


def folds_of(folds):
    for fold in range(5):
        train, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner = list(KFold(3, shuffle=True, random_state=SPLIT_SEED+fold).split(train))
        yield fold, train, val, inner


def take_features(bundle, projection):
    return np.column_stack([bundle["features"], projection]).astype(np.float32)


def collect(fold, X, Y, latlon, train, val, inner):
    from fusion_v3 import CONFIGS, fit_predict
    path = OUT / f"fold-{fold}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = perf_counter()
    outer = make_bundle(f"o{fold}", X, Y, train, val, SPLIT_SEED+200+fold)
    p_train, p_val, *_ = pca_features(X, train, val, SEED+fold)
    f_val = take_features(outer, p_val)
    f_train = np.empty((len(train), f_val.shape[1]), dtype=np.float32)
    e_train = np.empty((len(train), *outer["experts"].shape[1:]), dtype=np.float32)
    seen = np.zeros(len(train), dtype=bool)
    for i, (t, q) in enumerate(inner):
        assert not seen[q].any()
        pair = make_bundle(f"o{fold}-i{i}", X, Y, train[t], train[q], SEED+fold*3+i)
        f_train[q] = take_features(pair, p_train[q])
        e_train[q] = pair["experts"]
        seen[q] = True
    assert seen.all()
    del pair
    torch.cuda.empty_cache()
    predictions = {k: v for k, v in outer.items() if k not in ("features", "experts")}
    for name, config in CONFIGS.items():
        print(f"Fusion {name}, fold {fold+1}", flush=True)
        predictions[f"fusion_{name}"] = fit_predict(
            f_train, Y[train], f_val, e_train, outer["experts"], config, SEED+fold)
    scores = {name: metrics(unit(value), latlon[val]) for name, value in predictions.items()}
    save_json(path, scores)
    print(f"Completed fold {fold+1}/5 in {perf_counter()-t0:.1f}s", flush=True)
    print({name: round(record["rmse_km"], 2) for name, record in scores.items()}, flush=True)
    return scores


def cv(X, Y, latlon, folds):
    summaries = [collect(fold, X, Y, latlon, train, val, inner)
                 for fold, train, val, inner in folds_of(folds)]
    leaderboard = []
    for name in summaries[0]:
        parts = [fold[name] for fold in summaries]
        n = sum(p["n"] for p in parts)
        mae = sum(p["mae_km"] * p["n"] for p in parts) / n
        mse = sum(p["mse_km2"] * p["n"] for p in parts) / n
        leaderboard.append({"name": name, "cv_mae_km": mae, "cv_rmse_km": mse**.5,
                            "fold_rmse_km": [p["rmse_km"] for p in parts]})
    leaderboard.sort(key=lambda row: row["cv_rmse_km"])
    locked = {"winner": leaderboard[0]["name"], "candidates": leaderboard,
              "split_sha256": fingerprint()}
    save_json(OUT / "cv-summary.json", locked)
    print(json.dumps(locked, indent=2), flush=True)


def final(X, Y, folds):
    locked = json.loads((OUT / "cv-summary.json").read_text())
    assert locked["split_sha256"] == fingerprint()
    if (OUT / "results.json").exists():
        old = json.loads((OUT / "results.json").read_text())
        assert old["winner"] == locked["winner"]
        print(json.dumps(old, indent=2), flush=True)
        return
    winner = locked["winner"]
    print("Locked by CV:", winner, flush=True)
    test_x = np.load(DATA / "test_x.npy", mmap_mode="r")
    x = np.concatenate([X, test_x])
    y = np.concatenate([Y, np.full((len(test_x), 3), np.nan, dtype=np.float32)])
    train, query = np.arange(len(X)), np.arange(len(X), len(x))
    bundle = make_bundle("final", x, y, train, query, SPLIT_SEED+205, save=True)
    if winner.startswith("fusion_"):
        from fusion_v3 import CONFIGS, fit_predict
        p_train, p_test, center, projection = pca_features(x, train, query, SEED+5)
        atomic_npz(OUT / "best-pca.npz", center=center, projection=projection)
        f_test = take_features(bundle, p_test)
        f_train = np.empty((len(train), f_test.shape[1]), np.float32)
        e_train = np.empty((len(train), *bundle["experts"].shape[1:]), np.float32)
        for i, _, v, _ in folds_of(folds):
            with np.load(OUT / f"bundle-o{i}.npz") as cache:
                f_train[v] = take_features(cache, p_train[v])
                e_train[v] = cache["experts"]
        torch.cuda.empty_cache()
        xyz = fit_predict(f_train, Y, f_test, e_train, bundle["experts"],
                          CONFIGS[winner.removeprefix("fusion_")], SEED+5,
                          save_path=OUT / "best-fusion.cbm")
    else:
        xyz = bundle[winner]
    xyz = unit(xyz)
    atomic_npz(OUT / "test-predictions.npz", xyz=xyz)
    truth = np.load(DATA / "test_latlon.npy", mmap_mode="r")
    result = {"winner": winner, "test": metrics(xyz, truth), "cv": locked["candidates"][0],
              "historical_reference": REFERENCE, "split_sha256": fingerprint()}
    save_json(OUT / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cv", "final"))
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU required."
    OUT.mkdir(parents=True, exist_ok=True)
    digest = fingerprint()
    marker = OUT / "split.sha256"
    if marker.exists() and marker.read_text().strip() != digest:
        raise RuntimeError("Cached artifacts use a different split")
    marker.write_text(digest + "\n")
    X = np.load(DATA / "dev_x.npy", mmap_mode="r")
    Y = np.load(DATA / "dev_y.npy", mmap_mode="r")
    folds = np.load(DATA / "folds.npy")
    assert X.shape == (212500,1280) and np.bincount(folds).tolist() == [42500]*5
    if args.phase == "cv":
        cv(X, Y, np.load(DATA / "dev_latlon.npy", mmap_mode="r"), folds)
    else:
        final(X, Y, folds)


if __name__ == "__main__":
    main()
