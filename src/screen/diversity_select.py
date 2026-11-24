"""Calibrate complementary geographic specialists using saved screening predictions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from pyproj import Geod
from threadpoolctl import threadpool_limits


GEOD = Geod(ellps="WGS84")
E2 = 1 - (1 - 1 / 298.257223563) ** 2
SEED = 904317
RIDGE = .05
FOLDS = 4
MIN_GAIN_KM = .05
SHORTLIST = 12
SOURCES = {"retrieval", "geo", "neural"}


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def lonlat(xyz):
    x, y, z = np.asarray(xyz, dtype=np.float64).T
    if not np.isfinite(xyz).all() or np.any(x*x + y*y + z*z < 1e-10):
        raise ValueError("Nonfinite or zero geographic forecast")
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, (1 - E2) * np.hypot(x, y)))


def offset(origin, dest):
    az, _, dist = GEOD.inv(*origin, *dest)
    az = np.radians(az)
    return np.column_stack((np.sin(az), np.cos(az))) * (dist * .001)[:, None]


def direct(origin, vector):
    az = np.degrees(np.arctan2(vector[:, 0], vector[:, 1]))
    lon, lat, _ = GEOD.fwd(*origin, az, 1000 * np.linalg.norm(vector, axis=1))
    return lon, lat


def losses(pred, truth):
    return np.asarray(GEOD.inv(*pred, *truth)[2]) * .001


def metrics(d):
    d = np.asarray(d, dtype=np.float64)
    mse = float(d @ d / len(d))
    return dict(n=len(d), mae_km=float(d.mean()), mse_km2=mse, rmse_km=float(np.sqrt(mse)))


def correlation(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    scale = np.linalg.norm(x.ravel()) * np.linalg.norm(y.ravel())
    return None if scale < 1e-12 else float(np.clip(np.sum(x*y) / scale, -1, 1))


def diagnostics(baseline_error, candidate_error, distance, cand_distance):
    # Both signed errors are WGS84 truth -> prediction in the SAME local frame.
    large = distance >= np.quantile(distance, .90)
    return dict(error_corr_east=correlation(baseline_error[:, 0], candidate_error[:, 0]),
                error_corr_north=correlation(baseline_error[:, 1], candidate_error[:, 1]),
                error_corr_vector=correlation(baseline_error, candidate_error),
                error_corr_distance=correlation(distance, cand_distance),
                candidate_closer_fraction=float(np.mean(cand_distance < distance)),
                closer_in_baseline_worst_decile=float(np.mean(cand_distance[large] < distance[large])),
                baseline_worst_decile_rmse_km=metrics(distance[large])["rmse_km"],
                candidate_on_baseline_worst_decile_rmse_km=metrics(cand_distance[large])["rmse_km"])


def raw_file(folder, name):
    return folder / (name.replace("/", "__") + ".npy")


def fit_subset(grams, moments, ids):
    g = grams[np.ix_(ids, ids)].copy()
    g.flat[::len(ids)+1] += RIDGE * np.maximum(1e-6, np.diag(g))
    try:
        w = np.linalg.solve(g, moments[ids])
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(g, moments[ids], rcond=1e-10)[0]
    return np.clip(w, -.40, .65)


def stats_by_fold(design, target, fold):
    stats = []
    for i in range(FOLDS):
        idx = fold == i
        x = design[idx].reshape(-1, design.shape[-1]).astype(np.float64)
        y = target[idx].ravel()
        stats.append((x.T @ x, x.T @ y, float(y @ y)))
    return tuple(np.stack([row[i] for row in stats]) for i in range(3))


def fit_cv(stats, ids):
    g, m, yy = stats
    total_g, total_m = g.sum(0), m.sum(0)
    weights, sse = [], 0.
    for i in range(FOLDS):
        w = fit_subset(total_g - g[i], total_m - m[i], ids)
        weights.append(w)
        sse += yy[i] - 2 * w @ m[i, ids] + w @ g[i][np.ix_(ids, ids)] @ w
    return np.stack(weights), fit_subset(total_g, total_m, ids), max(0., float(sse))


def apply_weights(design, weights, fold=None):
    if fold is None:
        return np.einsum("nkj,j->nk", design, weights)
    return np.einsum("nkj,nj->nk", design, weights[fold])


def exact(origin, truth, design, weights, fold=None):
    return metrics(losses(direct(origin, apply_weights(design, weights, fold)), truth))


def greedy(design, target, origin, truth, base_rmse, names, rows, fold):
    stats = stats_by_fold(design, target, fold)
    current_rmse = base_rmse
    selected, steps = [], []
    for round_id in range(4):
        pool = []
        for idx in range(len(names)):
            if idx in selected:
                continue
            ids = [*selected, idx]
            weights, all_w, sse = fit_cv(stats, ids)
            pool.append((sse, idx, weights, all_w))
        pool.sort(key=lambda r: (r[0], names[r[1]]))
        if round_id == 0:
            # Every singleton has already been scored using exact WGS84.
            pool.sort(key=lambda r: (rows[r[1]]["cal_signed"]["rmse_km"], names[r[1]]))
        tested = []
        for sse, idx, weights, all_w in pool[:SHORTLIST]:
            ids = [*selected, idx]
            score = rows[idx]["cal_signed"] if round_id == 0 else exact(origin, truth, design[:, :, ids], weights, fold)
            tested.append((score["rmse_km"], idx, all_w, score, sse))
        tested.sort(key=lambda x: (x[0], names[x[1]]))
        rmse, idx, all_w, score, _ = tested[0]
        accepted = current_rmse - rmse >= MIN_GAIN_KM
        steps.append(dict(round=round_id+1, candidate=names[idx], selected=accepted,
                          marginal_cal_gain_km=current_rmse-rmse, cal_joint_oof=score,
                          full_cal_weights={names[i]: float(w) for i,w in zip([*selected,idx], all_w)},
                          exact_shortlist=[dict(name=names[i], rmse_km=v, tangent_rmse_km=(s/len(fold))**.5)
                                           for v,i,_,_,s in tested]))
        if not accepted:
            break
        selected.append(idx)
        current_rmse = rmse
    return selected, steps


def calculate(prior_xyz, y_cal, names, loader, expected_n, seed=SEED):
    origin, truth = lonlat(prior_xyz), lonlat(y_cal)
    target, base_error = offset(origin, truth), offset(truth, origin)
    d0 = np.linalg.norm(base_error, axis=1)
    baseline = metrics(d0)
    n = len(y_cal)
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(seed + 771).permutation(n)] = np.arange(n) % FOLDS
    design = np.empty((n, 2, len(names)), np.float32)
    error_matrix = np.empty_like(design)
    rows = []
    for index, name in enumerate(names):
        data = loader(name)
        if data.shape != (expected_n, 3):
            raise ValueError(f"Invalid raw prediction {name}: {data.shape}")
        location = lonlat(data[:n])
        cand_err = offset(truth, location)
        distance = np.linalg.norm(cand_err, axis=1)
        vector = offset(origin, location).astype(np.float32)
        design[:, :, index] = vector
        error_matrix[:, :, index] = cand_err
        vv = np.bincount(fold, weights=np.square(vector.astype(np.float64)).sum(1), minlength=FOLDS)
        vy = np.bincount(fold, weights=(vector * target).sum(1), minlength=FOLDS)
        scale = np.maximum((1 + RIDGE) * (vv.sum() - vv), 1e-10)
        w = ((vy.sum() - vy) / scale).clip(-.40, .65)
        all_w = float(np.clip(vy.sum() / max(1e-10, (1+RIDGE)*vv.sum()), -.40, .65))
        signed = exact(origin, truth, vector[:, :, None], w[:, None], fold)
        positive = (signed if (w >= 0).all() else baseline if (w <= 0).all()
                    else exact(origin, truth, vector[:, :, None], w.clip(0)[:, None], fold))
        standalone = metrics(distance)
        rows.append(dict(name=name, cal_standalone=standalone, cal_signed=signed, cal_positive=positive,
                    cal_gain_signed_km=baseline["rmse_km"]-signed["rmse_km"],
                    cal_gain_positive_km=baseline["rmse_km"]-positive["rmse_km"],
                    cal_standalone_gap_to_v4_km=standalone["rmse_km"]-baseline["rmse_km"],
                    signed_weight=all_w, positive_weight=max(0., all_w), signed_oof_weights=w.tolist(),
                    **diagnostics(base_error, cand_err, d0, distance)))
        if (index + 1) % 50 == 0:
            print(f"Calibration: {index+1}/{len(names)} candidates", flush=True)
    chosen, steps = greedy(design, target, origin, truth, baseline["rmse_km"], names, rows, fold)
    for j in chosen:
        for k, r in enumerate(rows):
            r.setdefault("error_corr_vector_to_greedy_selected", {})[names[j]] = correlation(
                error_matrix[:, :, j], error_matrix[:, :, k])
    by_rank = sorted(rows, key=lambda r:(-r["cal_gain_signed_km"], r["name"]))
    solo_order = sorted(rows, key=lambda r:(r["cal_standalone"]["rmse_km"], r["name"]))
    for ranking, label in ((by_rank, "signed_gain_rank"), (solo_order, "standalone_rank")):
        for i, row in enumerate(ranking, 1):
            row[label] = i
    return by_rank, [names[i] for i in chosen], steps, baseline


def route(rows, selected, old):
    retained = set(old["required"])
    champions = set(old["champions"].values())
    ranked = [r["name"] for r in rows]
    gain = {r["name"]: r["cal_gain_signed_km"] for r in rows}
    answer = []
    for n in selected:
        if n not in champions and (n in retained or sum(x not in retained for x in answer) < 2):
            answer.append(n)
        if len(answer) == 3:
            break
    # Shortlist novelty for the costly full refits only after the independent
    # four-step greedy pool and all singleton rankings have been frozen.
    novel = next((n for n in ranked if n not in retained and n not in champions and gain[n] >= MIN_GAIN_KM), None)
    if novel and all(n in retained for n in answer):
        if len(answer) == 3:
            answer.pop()
        answer.append(novel)
    if not answer:
        answer.append(next(n for n in ranked if n not in champions))
    return answer, [n for n in answer if n not in retained]


def screen_prior(data, train, val, folds, fit, cal, evaluate, y_fit):
    from sklearn.model_selection import KFold
    sys.path.insert(0, str(data))
    from calibration_ideas import predict
    # cal_interaction reads columns 21..197 and the seven expert directions.
    # PCA is omitted on both sides: its 32 appended columns are unused by this
    # exact model. This avoids sharing GPU memory with the ongoing CV run.
    pos_fit, pos_cal = np.searchsorted(train, fit), np.searchsorted(train, cal)
    pos_eval = np.searchsorted(val, evaluate)
    needed = np.sort(np.concatenate((pos_fit, pos_cal)))
    put = np.full(len(train), -1, dtype=int)
    put[needed] = np.arange(len(needed))
    train_f = train_e = None
    inner = KFold(3, shuffle=True, random_state=523748).split(train)
    for i, (_, query) in enumerate(inner):
        with np.load(data/"v3"/f"bundle-o0-i{i}.npz", allow_pickle=False) as obj:
            keep = put[query] >= 0
            feat, exp = obj["features"], obj["experts"]
            if train_f is None:
                train_f = np.empty((len(needed), feat.shape[1]), np.float32)
                train_e = np.empty((len(needed), *exp.shape[1:]), np.float32)
            train_f[put[query[keep]]] = feat[keep]
            train_e[put[query[keep]]] = exp[keep]
    with np.load(data/"v3"/"bundle-o0.npz", allow_pickle=False) as obj:
        eval_f, eval_e = obj["features"][pos_eval], obj["experts"][pos_eval]
    query_f = np.concatenate([train_f[put[pos_cal]], eval_f])
    query_e = np.concatenate([train_e[put[pos_cal]], eval_e])
    p = predict(train_f[put[pos_fit]], y_fit, query_f, train_e[put[pos_fit]], query_e,
                SEED, only="cal_interaction")["cal_interaction"]
    # screen.py calls run_v3.unit on this array before appending to the library.
    p = np.asarray(p, np.float32)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def paired_interval(baseline_d, model_d, seed=SEED+13):
    """Percentile paired bootstrap of fixed predictions, resampling individual rows."""
    baseline_d, model_d = (np.asarray(v, dtype=np.float64) for v in (baseline_d, model_d))
    bsq, msq = baseline_d**2, model_d**2
    rng = np.random.default_rng(seed)
    mae, rmse = [], []
    for start in range(0, 1500, 25):
        rows = rng.integers(0, len(model_d), size=(min(25,1500-start), len(model_d)))
        mae.extend((model_d[rows]-baseline_d[rows]).mean(axis=1))
        rmse.extend(np.sqrt(msq[rows].mean(axis=1))-np.sqrt(bsq[rows].mean(axis=1)))
    return dict(seed=seed, resamples=1500, unit="development evaluation row", contrast="model minus V4",
                delta_mae_km_ci95=np.quantile(mae,[.025,.975]).tolist(),
                delta_rmse_km_ci95=np.quantile(rmse,[.025,.975]).tolist())


def eval_frozen(locked, prior_xyz, eval_ll, loader, n_cal):
    truth = (eval_ll[:, 1], eval_ll[:, 0])
    origin = lonlat(prior_xyz)
    base_error = offset(truth, origin)
    base_d = np.linalg.norm(base_error, axis=1)
    rows = {r["name"]:r for r in locked["candidate_calibration"]}
    union = list(dict.fromkeys([*locked["selected"], *locked["full_cv_additions"]]))
    out = {}
    design = {}
    base = metrics(base_d)
    for name in union:
        location = lonlat(loader(name)[n_cal:])
        vector = offset(origin, location)
        design[name] = vector
        cand_err = offset(truth, location)
        distance = np.linalg.norm(cand_err, axis=1)
        item = rows[name]
        positive = metrics(losses(direct(origin, vector * item["positive_weight"]), truth))
        signed = metrics(losses(direct(origin, vector * item["signed_weight"]), truth))
        out[name] = dict(standalone=metrics(distance), positive=positive, signed=signed,
                         gain_signed_km=base["rmse_km"]-signed["rmse_km"],
                         gain_positive_km=base["rmse_km"]-positive["rmse_km"],
                         **diagnostics(base_error, cand_err, base_d, distance))
    steps = []
    final_d = base_d
    for step in locked["greedy_steps"]:
        if not step["selected"]:
            continue
        weights = step["full_cal_weights"]
        vector = sum(weights[k]*design[k] for k in weights)
        final_d = losses(direct(origin, vector), truth)
        score = metrics(final_d)
        steps.append(dict(weights=weights, metrics=score, gain_over_v4_km=base["rmse_km"]-score["rmse_km"]))
    return dict(v4=base, selected_or_routed=out, greedy_steps=steps, greedy_joint=(steps[-1]["metrics"] if steps else base),
                joint_paired_uncertainty=paired_interval(base_d, final_d))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_lock(data, screen, names, prior, y, cal, loader):
    with threadpool_limits(limits=3, user_api="blas"):
        rows, selected, steps, baseline = calculate(prior[:len(cal)], y[cal], names, loader, len(prior))
    old = read(data/"full500"/"selection.json")
    additions, new = route(rows, selected, old)
    if len(additions)>3 or len(new)>2 or set(additions).intersection(old["champions"].values()):
        raise AssertionError("Invalid full-CV routing")
    locked = dict(format_version=1, raw_models=300, calibration_n=len(cal), evaluation_n=len(prior)-len(cal),
        split_sha256=digest(screen/"split.npz"), seed=SEED, v4_calibration=baseline,
        selected=selected, full_cv_additions=additions, new_for_full_cv=new,
        greedy_steps=steps, candidate_calibration=rows,
        procedure=dict(anchor="V4 cal_interaction trained on screen fit (60000), V3 inner OOF features for fit/cal; "
                              "same model as prior/prior/v4 component, not the full-fit report control",
            geometry="WGS84 Geod.inv/fwd in km; predictor-to-truth tangent offsets around V4 fit weights. "
                     "Signed east/north error correlations use truth-to-prediction offsets for both models in a shared frame.",
            calibration="20000 calibration rows partitioned into 4 deterministic shuffled blocks. Each calibration "
                        "blend is scored out-of-fold with weights fitted on the other 3. Selection across OOF scores "
                        "still tunes on this calibration sample.",
            weights="ridge 0.05 times component Gram diagonal; signed coefficients clipped to [-0.40, 0.65], "
                    "positive singleton coefficients clipped to [0,0.65]; all weights refit on full calibration before evaluation",
            selection="All 300 ranked by measured exact WGS84 singleton gain over V4, separate positive and signed gains. "
                      "Forward search, max four, accepts marginal gains >=0.05 km. At later steps all remaining candidates "
                      "ranked by OOF tangent MSE; best 12 rescored by exact WGS84. Ties broken by name.",
            full_cv_routing="Post-lock resource routing: exclude previously retained family champions, take up to three "
                            "additions, at most two absent from previous required cache. Include best positive-gain unseen "
                            "candidate if greedy has none. If necessary route best eligible singleton explicitly as exploratory.",
            evaluation="Only after writing selection, evaluate selected and resource-routed specialists plus greedy "
                       "blend on 12500 pre-existing screening validation rows. This partition was historically used "
                       "for previous screen selection, and is not an untouched final test."))
    return locked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("GEO_CV_ROOT", "/content/train250k-cv")))
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    data = args.data
    screen, dest = data/"screen500", data/"diversity500"
    dest.mkdir(parents=True, exist_ok=True)
    registry = read(screen/"registry.json")
    names = sorted(k for k,v in registry.items() if v["family"] in SOURCES)
    if len(names) != 300 or any(not raw_file(screen,k).is_file() for k in names):
        raise ValueError("Expected all 300 measured raw configuration forecasts")
    with np.load(screen/"split.npz", allow_pickle=False) as split:
        fit, cal, evaluate = (split[k] for k in ("fit", "cal", "evaluate"))
    folds = np.load(data/"folds.npy")
    train, val = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    if ((len(fit), len(cal), len(evaluate)) != (60000,20000,12500)
        or any(np.any(np.diff(idx) <= 0) for idx in (fit, cal, evaluate))
        or not (np.isin(fit, train).all() and np.isin(cal, train).all() and np.isin(evaluate, val).all())
        or np.intersect1d(fit,cal).size):
        raise ValueError("Unexpected or overlapping screening partitions")
    y = np.load(data/"dev_y.npy", mmap_mode="r")
    prior = screen_prior(data,train,val,folds,fit,cal,evaluate,y[fit])
    loader = lambda name: np.load(raw_file(screen,name), mmap_mode="r", allow_pickle=False)
    # An existing selection is immutable, including when repeating just the
    # diagnostics. Standalone/OOF rankings are never rerun using observed eval.
    lock_path = dest/"selection.json"
    if lock_path.exists():
        locked = read(lock_path)
        if (locked.get("split_sha256") != digest(screen/"split.npz")
            or locked.get("seed") != SEED
            or {row["name"] for row in locked["candidate_calibration"]} != set(names)):
            raise ValueError("Existing diversity selection has incompatible split or candidates")
    else:
        locked = select_lock(data, screen, names, prior, y, cal, loader)
        write(lock_path, locked)
    print(f"Locked {locked['selected']}, full CV additions {locked['full_cv_additions']}", flush=True)
    eval_ll = np.load(data/"dev_latlon.npy", mmap_mode="r")[evaluate]
    evaluated = eval_frozen(locked, prior[len(cal):], eval_ll, loader, len(cal))
    report = dict(selection_sha256=digest(lock_path), **evaluated)
    write(dest/"results.json", report)
    print("Frozen evaluation:", json.dumps(report["greedy_joint"]), flush=True)
    if args.supplement:
        from diversity_context import supplement
        extra = supplement(data, screen, names, cal, evaluate, val, prior[:len(cal)], prior[len(cal):],
                           y[cal], y[evaluate])
        write(dest/"supplement.json", extra)


if __name__ == "__main__":
    main()
