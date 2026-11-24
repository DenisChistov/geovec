"""Measure incremental value of calibration-selected, less-correlated raw experts on development folds."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from pyproj import Geod

import full_search as search
import strong_fusion as strong
from diversity_rules import RULES, fit_rule, apply_rule
from experiment import decode, save_json
from run_v3 import fingerprint, folds_of


DATA, ROOT = search.DATA, search.ROOT
_FORMAT = 1
_SEED = 1049251
_GEOD = Geod(ellps="WGS84")


def _json(p):
    return p.read_text(encoding="utf-8")


def _selection(members):
    members = list(members)
    if not (5 <= len(members) <= 7 and len(members) == len(set(members))):
        raise ValueError("Expected four baseline experts and one to three distinct additions")
    if any(k.split("/")[0] not in {"neural", "geo", "retrieval"} for k in members[4:]):
        raise ValueError("Only raw specialists qualify as diversity additions")
    return members


def member_ids(selection):
    if isinstance(selection, (tuple, list)):
        return _selection(selection)
    if "diversity_fusion" in selection:
        return _selection(selection["diversity_fusion"]["members"])
    base = selection.get("selection", selection)
    picked = json.loads(_json(DATA / "diversity500" / "selection.json"))["full_cv_additions"]
    return _selection([*strong.member_ids(base), *picked])


@lru_cache(maxsize=1)
def _sources():
    import diversity_rules
    import features, geo_models, geo_geometry, neural_models, retrieval_models
    h = hashlib.sha256()
    for module in (diversity_rules, features, geo_models, geo_geometry, neural_models,
                   retrieval_models, search, strong):
        h.update(Path(module.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def _signature(members, split, seed, paths):
    source = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
    value = json.dumps([_FORMAT, _sources(), list(members), split, seed, source],
                       separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _extra_predictions(X, Y, fit_rows, query_rows, names, original, dest, split, seed):
    signature = _signature(names, split, seed, [original])
    saved = strong._checkpoint_read(dest, signature, query_rows, names)
    if saved is not None:
        return saved
    with np.load(original, allow_pickle=False) as base:
        result = {k: strong._xyz(base[k], len(query_rows)) for k in names if k in base.files}
    missing = [k for k in names if k not in result]
    if missing:
        print(f"Diversity: fitting {', '.join(missing)} for {dest.name}", flush=True)
        new = search.predict_split(missing, X, Y, fit_rows, query_rows, None, None, None, None, seed)
        result.update({k: strong._xyz(new[k], len(query_rows)) for k in missing})
    strong._checkpoint_write(dest, signature, query_rows, result)
    return result


def _nested_pair(path, names, rows):
    with np.load(path, allow_pickle=False) as ar:
        if not np.array_equal(ar["rows"], rows) or not {"signature", *names}.issubset(ar.files):
            raise ValueError(f"Strong-fusion nested cache has incompatible rows or members: {path}")
        return {k: strong._xyz(ar[k], len(rows)) for k in names}


def collect_fold(fold, X, Y, train, val, inner, selection):
    """Return member names, honest training OOF XYZ, and outer validation XYZ in index order."""
    members = member_ids(selection)
    core, extra = members[:4], members[4:]
    if np.intersect1d(train, val).size:
        raise ValueError("Outer training and validation overlap")
    split = fingerprint()
    work = ROOT / "diversity"
    outer = ROOT / f"base-outer-{fold}.npz"
    query = strong._read_members(outer, core, val)
    query.update(_extra_predictions(X, Y, train, val, extra, outer,
                                    work / outer.name, split, search.SEED + fold))
    table = np.full((len(train), len(members), 3), np.nan, dtype=np.float32)
    seen = np.zeros(len(train), dtype=bool)
    for j, (a, b) in enumerate(inner):
        if seen[b].any() or np.intersect1d(a, b).size or len(a) + len(b) != len(train):
            raise ValueError("Invalid inner assignments")
        rows = train[b]
        path = ROOT / f"base-inner-{fold}-{j}.npz"
        part = _nested_pair(ROOT / "strong" / f"level2-{fold}-{j}.npz", core[:2], rows)
        part.update(strong._read_members(path, core[2:], rows))
        part.update(_extra_predictions(X, Y, train[a], rows, extra, path, work / path.name,
                                       split, search.SEED + 100 + fold * 3 + j))
        table[b] = strong._as_experts(part, members, len(rows))
        seen[b] = True
    if not seen.all() or not np.isfinite(table).all():
        raise ValueError("Incomplete inner predictions for diversity fusion")
    return members, table, strong._as_experts(query, members, len(val))


def final_oof(members, folds):
    """Load all outer specialist forecasts in original development row order."""
    members, folds = _selection(members), np.asarray(folds)
    if folds.ndim != 1 or set(np.unique(folds)) != set(range(5)):
        raise ValueError("Expected five nonempty outer folds")
    table = np.full((len(folds), len(members), 3), np.nan, dtype=np.float32)
    for fold in range(5):
        rows = np.flatnonzero(folds == fold)
        p = ROOT / f"base-outer-{fold}.npz"
        base = strong._read_members(p, members[:4], rows)
        base.update(_nested_pair(ROOT / "diversity" / p.name, members[4:], rows))
        table[rows] = strong._as_experts(base, members, len(rows))
    return table


def final_prediction(name, members, query_predictions, Y, folds, save_path=None):
    members = _selection(members)
    if name not in RULES or len(folds) != len(Y):
        raise ValueError("Unknown diversity rule or mismatched development labels")
    train = final_oof(members, folds)
    state = fit_rule(name, train, strong._xyz(Y, len(Y)), _SEED + 5)
    n = len(query_predictions[members[0]])
    answer = strong._xyz(apply_rule(name, state, strong._as_experts(query_predictions, members, n)), n)
    if save_path is not None:
        strong._write_rule(save_path, name, members, state)
        if not np.allclose(predict_saved(save_path, query_predictions), answer, rtol=0, atol=2e-5):
            raise ValueError("Diversity checkpoint replay differs from fit")
    return answer


def predict_saved(path, query_predictions):
    with np.load(path, allow_pickle=False) as ar:
        name, members = str(ar["name"]), _selection(ar["members"].tolist())
        if name not in RULES or int(ar["format"]) != _FORMAT:
            raise ValueError("Unsupported diversity checkpoint")
        state = {k[6:]: ar[k] for k in ar.files if k.startswith("param_")}
    n = len(query_predictions[members[0]])
    return strong._xyz(apply_rule(name, state, strong._as_experts(query_predictions, members, n)), n)


def _controls(name):
    return "four_only/" + name


def _err(latlon, pred):
    lat, lon = decode(pred)
    az, _, meters = _GEOD.inv(latlon[:, 1], latlon[:, 0], lon, lat)
    d = np.asarray(meters, dtype=np.float64) * .001
    a = np.radians(az)
    en = d[:, None] * np.column_stack([np.sin(a), np.cos(a)])
    return d, en


def _score(d):
    mse = float(np.mean(d*d))
    return {"n": len(d), "mae_km": float(np.mean(d)), "mse_km2": mse, "rmse_km": mse**.5}


def _diagnostics(members, query, predictions, latlon):
    # Held-out truth enters this function only after all full and ablated forecasts exist.
    distances, vectors = zip(*[_err(latlon, query[:, j]) for j in range(len(members))])
    d = np.stack(distances, axis=1)
    en = np.stack(vectors, axis=1)
    stats = {"members": list(members), "n": len(d), "sum_d": d.sum(0).tolist(),
             "dot_d": (d.T @ d).tolist(), "sum_en": en.sum(0).tolist(),
             "dot_en": np.einsum("nid,njd->ij", en, en, optimize=True).tolist(),
             "ablation": {}}
    scores = {}
    for k in RULES:
        full = _err(latlon, predictions[k])[0]
        base = _err(latlon, predictions[_controls(k)])[0]
        scores[k], scores[_controls(k)] = _score(full), _score(base)
        tail = base >= 1500
        stats["ablation"][k] = {
            "wins": int(np.sum(full < base)), "ties": int(np.sum(full == base)),
            "tail_n": int(tail.sum()), "tail_mse_diff_sum": float(np.sum((full[tail]-base[tail])*(full[tail]+base[tail])))}
    return scores, stats


def _pooled(parts, name):
    s = [x[name] for x in parts]
    n = sum(v["n"] for v in s)
    return {"name": name, "cv_mae_km": sum(v["mae_km"] * v["n"] for v in s)/n,
            "cv_rmse_km": (sum(v["mse_km2"] * v["n"] for v in s)/n)**.5,
            "fold_rmse_km": [v["rmse_km"] for v in s]}


def complementarity(parts, diagnostics, members):
    count = sum(p["n"] for p in diagnostics)
    def summed(k):
        return sum(np.asarray(p[k], dtype=np.float64) for p in diagnostics)
    sum_d, dot_d, sum_en, dot_en = [summed(k) for k in ("sum_d", "dot_d", "sum_en", "dot_en")]
    mag_cov = dot_d - np.outer(sum_d, sum_d)/count
    vector_cov = dot_en - (sum_en @ sum_en.T)/count
    def corr(cov, i, j):
        denom = float(max(cov[i, i], 0) * max(cov[j, j], 0))**.5
        return float(np.clip(cov[i, j] / denom, -1, 1)) if denom > 0 else None
    pairs = [{"a": a, "b": members[j], "distance_error_pearson": corr(mag_cov, i, j),
              "east_north_error_correlation": corr(vector_cov, i, j)}
             for i, a in enumerate(members) for j in range(i+1, len(members))]
    ablations = {}
    for name in RULES:
        full, base = _pooled(parts, name), _pooled(parts, _controls(name))
        local = [p["ablation"][name] for p in diagnostics]
        tail_n = sum(p["tail_n"] for p in local)
        ablations[name] = {"full": full, "four_experts_refit": base,
                           "delta_rmse_km": full["cv_rmse_km"] - base["cv_rmse_km"],
                           "delta_mae_km": full["cv_mae_km"] - base["cv_mae_km"],
                           "fraction_rows_improved": sum(p["wins"] for p in local)/count,
                           "fraction_ties": sum(p["ties"] for p in local)/count,
                           "four_expert_error_at_least_1500_km_n": tail_n,
                           "delta_mse_on_that_tail_km2": sum(p["tail_mse_diff_sum"] for p in local)/tail_n if tail_n else None}
    return {"n": count, "members": list(members), "pairwise_errors": pairs, "paired_ablation": ablations,
            "standalone": {k: {"cv_mae_km": float(sum_d[j]/count),
                               "cv_rmse_km": float(np.sqrt(dot_d[j,j]/count))} for j,k in enumerate(members)}}


def _evaluate(fold, X, Y, ll, train, val, inner, members, split):
    members, t, q = collect_fold(fold, X, Y, train, val, inner, members)
    work = ROOT / "diversity"
    paths = [ROOT / f"base-outer-{fold}.npz", work / f"base-outer-{fold}.npz"]
    for j in range(len(inner)):
        paths.extend([ROOT / "strong" / f"level2-{fold}-{j}.npz",
                      ROOT / f"base-inner-{fold}-{j}.npz", work / f"base-inner-{fold}-{j}.npz"])
    sig = _signature(members, split, _SEED + fold, paths)
    cache = work / f"oof-{fold}.npz"
    names = [*RULES, *[_controls(k) for k in RULES]]
    pred = strong._checkpoint_read(cache, sig, val, names)
    if pred is None:
        pred = {}
        for name in RULES:
            for key, train_x, query_x in ((name,t,q), (_controls(name),t[:, :4],q[:, :4])):
                state = fit_rule(name, train_x, Y[train], _SEED + fold)
                pred[key] = strong._xyz(apply_rule(name, state, query_x), len(val))
        strong._checkpoint_write(cache, sig, val, pred)
    return _diagnostics(members, q, pred, ll[val])


def main():
    lock_path = ROOT / "cv-summary.json"
    original = json.loads(_json(lock_path))
    if "strong_fusion" not in original or not set(original["strong_fusion"]["rules"]).issubset(
            {row["name"] for row in original["candidates"]}):
        raise ValueError("Complete strong_fusion.py before diversity CV")
    split = fingerprint()
    if original["split_sha256"] != split:
        raise ValueError("Frozen split differs from the full-search summary")
    source = DATA / "diversity500" / "selection.json"
    members = member_ids(original["selection"])
    if members[:4] != original["strong_fusion"]["members"]:
        raise ValueError("Strong-fusion baseline specialists changed")
    X,Y,ll,folds = [np.load(DATA / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                    for name in ("dev_x","dev_y","dev_latlon","folds")]
    if len(X) != len(folds) or len(Y) != len(folds) or len(ll) != len(folds) or set(np.unique(folds)) != set(range(5)):
        raise ValueError("Invalid development arrays")
    parts, diagnostics = [], []
    for f, t, v, inner in folds_of(folds):
        scores, diag = _evaluate(f, X, Y, ll, t, v, inner, members, split)
        parts.append(scores)
        diagnostics.append(diag)
        save_json(ROOT/"diversity"/f"fold-{f}.json", scores)
        print(f"Diversity fold {f+1}/5: " + ", ".join(f"{k} {scores[k]['rmse_km']:.2f}" for k in RULES), flush=True)
    detail = complementarity(parts, diagnostics, members)
    candidates = [r for r in original["candidates"] if not r["name"].startswith("diverse:")]
    candidates.extend(_pooled(parts,k) for k in RULES)
    candidates.sort(key=lambda x:(x["cv_rmse_km"],x["name"]))
    merged = {**original,"candidates":candidates,"winner":candidates[0]["name"],
              "diversity_fusion":{"format":_FORMAT,"members":members,"rules":dict(RULES),
                                  "selection_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                                  "diagnostics":"diversity/complementarity.json"}}
    if merged != original and (any((ROOT/p).exists() for p in ("final-provenance.json","test-predictions.npz","results.json"))
                               or json.loads(_json(lock_path)) != original):
        raise RuntimeError("CV lock changed or final artifacts already exist")
    save_json(ROOT/"diversity"/"complementarity.json",detail)
    if merged != original:
        save_json(lock_path,merged)
    print("Diversity merged. Development winner",merged["winner"],flush=True)


if __name__ == "__main__":
    main()
