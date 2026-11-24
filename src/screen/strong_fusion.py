"""Evaluate and refit stronger combinations of the V4, meta, paper and retrieval experts.

Run after full_search.py and before final_full.py. The CV entry point reads only
development labels. Final prediction accepts query predictions without labels.
"""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from full_search import DATA, ROOT, PRIOR, PAPER
from calibration_ideas import predict as calibrate
from experiment import metrics, save_json
from meta_models import fit_predict as fit_meta, views
from run_v3 import folds_of, fingerprint, pca_features, take_features
from strong_rules import RULES, fit_rule, apply_rule


_SEED = 1_034_429
_FORMAT = 1
_PARTS = 4
_PCA_VIEWS = {"all", "compact", "visual_pca"}


def member_ids(selection):
    """The retrieval champion ranked first among the screen's three raw families."""
    return [PRIOR, selection["champions"]["meta"], PAPER, selection["champions"]["retrieval"]]


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as output:
        temporary = Path(output.name)
        try:
            np.savez(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _xyz(a, n):
    a = np.asarray(a, dtype=np.float32)
    if a.shape != (n, 3) or not np.isfinite(a).all():
        raise ValueError(f"Expected {n} finite XYZ directions, got {a.shape}")
    size = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(size < .5) or np.any(abs(size - 1) > .02):
        raise ValueError("Expected unit ECEF directions")
    return np.ascontiguousarray(a / size)


def _read_members(path, names, rows):
    with np.load(path, allow_pickle=False) as data:
        missing = set(names).difference(data.files)
        if missing:
            raise ValueError(f"Missing base predictions in {path}: {sorted(missing)}")
        return {name: _xyz(data[name], len(rows)) for name in names}


def _checkpoint_read(path, signature, rows, names):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if not {"signature", "rows", *names}.issubset(data.files):
            return None
        if str(data["signature"]) != signature or not np.array_equal(data["rows"], rows):
            return None
        result = {}
        for name in names:
            stored = np.ascontiguousarray(data[name], dtype=np.float32)
            _xyz(stored, len(rows))
            # Keep the exact stored bits. Renormalizing here alone would make a
            # resumed CV score differ slightly from the first pass.
            result[name] = stored
        return result


def _checkpoint_write(path, signature, rows, predictions):
    _atomic_npz(path, {"signature": np.array(signature), "rows": rows, **predictions})


def _signature(split, members, paths):
    digest = hashlib.sha256(json.dumps([_FORMAT, _SEED, _PARTS, split, members],
                                      separators=(",", ":")).encode())
    for p in paths:
        stat = p.stat()
        digest.update(f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode())
    # Any change to the fitting or combination code invalidates cached predictions.
    import strong_rules
    for p in (Path(__file__), Path(strong_rules.__file__)):
        digest.update(p.read_bytes())
    return digest.hexdigest()


def _fit_inner_level2(source, X, Y, train_base, held_block, meta_key, seed):
    """V3 base predictors saw train_base, never any label from held_block.

    Split held_block again and fit the second-level predictors on its complement.
    Thus neither the V3 features nor second-level weights for a query used that
    query's label. The base feature generator is shared safely across complements.
    """
    if np.intersect1d(train_base, held_block).size or len(np.unique(held_block)) != len(held_block):
        raise ValueError("Invalid nested V3 base/query split")
    _, view, method = meta_key.split("/")
    with np.load(source, allow_pickle=False) as bundle:
        f, e = bundle["features"], bundle["experts"]
    if e.shape != (len(held_block), 7, 3) or len(f) != len(e):
        raise ValueError(f"Unexpected V3 nested bundle shape in {source}")
    if view in _PCA_VIEWS:
        # Fit the PCA on the original V3 base training rows, outside all stage-2 folds.
        _, projected, _, _ = pca_features(X, train_base, held_block, seed)
        f = take_features({"features": f}, projected)
    meta_x = next(matrix for candidate, matrix in views(f, e) if candidate == view)
    y = np.asarray(Y[held_block])
    out = {key: np.full((len(y), 3), np.nan, dtype="f") for key in (PRIOR, meta_key)}
    assigned = np.zeros(len(y), dtype=bool)
    parts = np.array_split(np.random.default_rng(seed).permutation(len(y)), _PARTS)
    for j, held in enumerate(parts):
        keep = np.ones(len(y), dtype=bool)
        keep[held] = False
        if len(held) == 0 or not keep.any() or assigned[held].any():
            raise ValueError("Invalid fold for second-level calibration")
        stage_seed = seed + j + 1
        out[PRIOR][held] = calibrate(f[keep], y[keep], f[held], e[keep], e[held], stage_seed,
                                     only="cal_interaction")["cal_interaction"]
        out[meta_key][held] = fit_meta(method, meta_x[keep], y[keep], meta_x[held],
                                        e[keep], e[held], stage_seed)
        assigned[held] = True
    if not assigned.all():
        raise ValueError("Incomplete nested meta/V4 predictions")
    return {key: _xyz(value, len(y)) for key, value in out.items()}


def _as_experts(mapping, members, n):
    return np.stack([_xyz(mapping[name], n) for name in members], axis=1)


def _fold_scores(fold, X, Y, latlon, train, valid, inner, members, split):
    work = ROOT / "strong"
    outer = ROOT / f"base-outer-{fold}.npz"
    inputs = [(ROOT / f"base-inner-{fold}-{j}.npz", DATA / "v3" / f"bundle-o{fold}-i{j}.npz")
              for j in range(len(inner))]
    signature = _signature(split, members, [outer, *[p for pair in inputs for p in pair]])
    pred_path = work / f"oof-{fold}.npz"
    predictions = _checkpoint_read(pred_path, signature, valid, RULES)
    if predictions is None:
        query = _as_experts(_read_members(outer, members, valid), members, len(valid))
        expert_train = np.full((len(train), len(members), 3), np.nan, dtype=np.float32)
        assigned = np.zeros(len(train), dtype=bool)
        for j, (a, b) in enumerate(inner):
            base_cache, v3_bundle = inputs[j]
            rows = train[b]
            if assigned[b].any() or np.intersect1d(train[a], rows).size:
                raise ValueError("Overlapping inner predictions")
            pair = members[:2]
            inner_path = work / f"level2-{fold}-{j}.npz"
            nested_sig = _signature(split, pair, [v3_bundle])
            meta = _checkpoint_read(inner_path, nested_sig, rows, pair)
            if meta is None:
                print(f"Strong fusion, outer {fold+1}/5, inner {j+1}/{len(inner)}: cross-fitting V4/meta", flush=True)
                meta = _fit_inner_level2(v3_bundle, X, Y, train[a], rows, pair[1],
                                         _SEED + 19 * fold + 4 * j)
                _checkpoint_write(inner_path, nested_sig, rows, meta)
            raw = _read_members(base_cache, members[2:], rows)
            expert_train[b] = _as_experts({**meta, **raw}, members, len(b))
            assigned[b] = True
        if not assigned.all() or not np.isfinite(expert_train).all():
            raise ValueError("Incomplete inner specialist predictions")
        predictions = {}
        for name in RULES:
            state = fit_rule(name, expert_train, Y[train], _SEED + fold)
            predictions[name] = _xyz(apply_rule(name, state, query), len(valid))
        _checkpoint_write(pred_path, signature, valid, predictions)
    return {name: metrics(pred, latlon[valid]) for name, pred in predictions.items()}


def merge_ranking(summary, parts):
    """Insert pooled five-fold strong scores into a full-search CV summary."""
    if len(parts) != 5 or any(not set(RULES).issubset(scores) for scores in parts):
        raise ValueError("Incomplete five-fold strong candidates")
    candidates = [row for row in summary["candidates"] if not row["name"].startswith("strong:")]
    for name in RULES:
        m = [s[name] for s in parts]
        if any(s["n"] <= 0 or not np.isfinite([s["mae_km"], s["mse_km2"], s["rmse_km"]]).all() for s in m):
            raise ValueError(f"Invalid metrics for {name}")
        n = sum(s["n"] for s in m)
        candidates.append({"name": name,
                           "cv_mae_km": sum(s["mae_km"] * s["n"] for s in m) / n,
                           "cv_rmse_km": (sum(s["mse_km2"] * s["n"] for s in m) / n)**.5,
                           "fold_rmse_km": [s["rmse_km"] for s in m]})
    candidates.sort(key=lambda row: (row["cv_rmse_km"], row["name"]))
    return {**summary, "candidates": candidates, "winner": candidates[0]["name"]}


def _write_rule(path, name, members, state):
    fields = {f"param_{k}": np.asarray(v) for k, v in state.items()}
    if any(value.dtype.hasobject for value in fields.values()):
        raise ValueError("Fusion checkpoint parameters must be NumPy-serializable without pickle")
    _atomic_npz(path, {**fields, "format": _FORMAT, "name": np.array(name),
                       "members": np.asarray(members)})


def predict_saved(path, query_predictions):
    """Replay a fitted strong rule on query predictions indexed by specialist key."""
    with np.load(path, allow_pickle=False) as archive:
        name = str(archive["name"])
        members = archive["members"].tolist()
        if int(archive["format"]) != _FORMAT or name not in RULES or len(members) != 4:
            raise ValueError("Unknown fusion checkpoint")
        state = {k[6:]: archive[k] for k in archive.files if k.startswith("param_")}
    n = len(query_predictions[members[0]])
    q = _as_experts(query_predictions, members, n)
    return _xyz(apply_rule(name, state, q), n)


def final_prediction(name, selection, query_predictions, Y, folds, save_path=None):
    """Refit the fusion on five cross-fitted dev predictions and apply to new queries."""
    members = member_ids(selection)
    folds = np.asarray(folds)
    if name not in RULES or folds.ndim != 1 or len(folds) != len(Y) or set(np.unique(folds)) != set(range(5)):
        raise ValueError("Expected a valid strong rule and all five development folds")
    dev = np.full((len(folds), len(members), 3), np.nan, dtype=np.float32)
    for f in range(5):
        rows = np.flatnonzero(folds == f)
        base = _read_members(ROOT / f"base-outer-{f}.npz", members, rows)
        dev[rows] = _as_experts(base, members, len(rows))
    state = fit_rule(name, dev, _xyz(Y, len(Y)), _SEED + 5)
    n = len(query_predictions[members[0]])
    pred = _xyz(apply_rule(name, state, _as_experts(query_predictions, members, n)), n)
    if save_path is not None:
        _write_rule(save_path, name, members, state)
        if not np.allclose(pred, predict_saved(save_path, query_predictions), rtol=0, atol=2e-5):
            raise ValueError("Strong fusion checkpoint replay differs from the fitted rule")
    return pred


def main():
    lock_path = ROOT / "cv-summary.json"
    original = json.loads(lock_path.read_text())
    selection = json.loads((ROOT / "selection.json").read_text())
    split = fingerprint()
    if selection != original["selection"] or original["split_sha256"] != split:
        raise ValueError("Full-search selection or split changed")
    marker = DATA / "v3" / "split.sha256"
    if marker.read_text().strip() != split:
        raise ValueError("V3 bundles use a different development split")
    members = member_ids(selection)
    if not set(members).issubset(selection["required"]):
        raise ValueError("A required fusion specialist was not retained in full search")
    X, Y, latlon, folds = [np.load(DATA / f"{n}.npy", mmap_mode="r", allow_pickle=False)
                           for n in ("dev_x", "dev_y", "dev_latlon", "folds")]
    if set(np.unique(folds)) != set(range(5)) or not len(X) == len(Y) == len(latlon) == len(folds):
        raise ValueError("Invalid development split")
    parts = []
    for f, tr, va, inner in folds_of(folds):
        scores = _fold_scores(f, X, Y, latlon, tr, va, inner, members, split)
        save_json(ROOT / "strong" / f"fold-{f}.json", scores)
        parts.append(scores)
        print(f"Strong fusion fold {f+1}/5: " + ", ".join(f"{k} {v['rmse_km']:.2f}"
                                                       for k, v in scores.items()), flush=True)
    merged = merge_ranking(original, parts)
    merged["strong_fusion"] = {"format": _FORMAT, "members": members, "rules": dict(RULES),
                               "meta_calibration_parts_per_inner_block": _PARTS}
    # Re-running a completed CV merge remains safe even after final scoring.
    if merged != original:
        if any((ROOT / s).exists() for s in ("final-provenance.json", "test-predictions.npz", "results.json")):
            raise RuntimeError("The existing CV lock already has final artifacts")
        if json.loads(lock_path.read_text()) != original:
            raise RuntimeError("Full-search CV summary changed while fitting strong fusion")
        save_json(lock_path, merged)
    print(f"Strong fusion merged into five-fold CV. Winner {merged['winner']}", flush=True)


if __name__ == "__main__":
    main()
