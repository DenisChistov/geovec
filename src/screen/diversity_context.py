"""Exploratory, post-lock comparisons on the development screening evaluation."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from diversity_select import diagnostics, direct, lonlat, losses, metrics, offset


class _Reference:
    def __init__(self, truth, prior, name):
        self.truth = lonlat(truth)
        self.point = lonlat(prior)
        self.error = offset(self.truth, self.point)
        self.distance = np.linalg.norm(self.error, axis=1)
        self.metrics = metrics(self.distance)
        self.name = name

    def row(self, name, prediction):
        point = lonlat(prediction)
        error = offset(self.truth, point)
        dist = np.linalg.norm(error, axis=1)
        score = metrics(dist)
        # A uniform evaluation-only diagnostic. No weight is selected by
        # looking at this set and its output is never a selection candidate.
        blend = direct(self.point, .25 * offset(self.point, point))
        mixed = metrics(losses(blend, self.truth))
        return dict(name=name, reference=self.name, standalone=score,
                    standalone_gain_vs_reference_km=self.metrics["rmse_km"]-score["rmse_km"],
                    fixed_quarter_blend=mixed,
                    quarter_blend_gain_vs_reference_km=self.metrics["rmse_km"]-mixed["rmse_km"],
                    **diagnostics(self.error, error, self.distance, dist))


def _full_cache(path, ref_key, targets, eval_ids, val_ids, default_prior):
    if not path.is_file():
        return {"status": "cache_missing", "filename": path.name, "rows": []}
    with np.load(path, allow_pickle=False) as saved:
        ref_name = ref_key if ref_key in saved.files else "screen_v4_calibrator_fit_60000"
        lookup = np.argsort(val_ids)
        pos = np.searchsorted(val_ids[lookup], eval_ids)
        if np.any(pos >= len(lookup)) or not np.array_equal(val_ids[lookup[pos]], eval_ids):
            raise ValueError("Evaluation IDs are not contained in outer fold zero")
        indices = lookup[pos]
        def read(name):
            a = saved[name]
            if a.shape != (len(val_ids), 3):
                raise ValueError(f"Misaligned outer prediction in {path.name}: {name}")
            return a[indices]
        baseline = read(ref_key) if ref_key in saved.files else default_prior
        base = _Reference(targets, baseline, ref_name)
        rows = [base.row(name, read(name)) for name in sorted(saved.files) if name != ref_key]
    return {"status": "ready", "filename": path.name, "reference": base.name,
            "reference_metrics": base.metrics, "rows": rows}


def _replay(screen_dir, raw_names, ncal, neval, prior_cal, prior_eval, cal_y, baseline, summary):
    group_file = screen_dir / "ensemble-components.json"
    if not group_file.is_file():
        return {"status": "component_list_missing", "rows": []}
    groups = json.loads(group_file.read_text())
    known = set(raw_names)
    prior_key = "prior/prior/v4"
    paths = {name: screen_dir / (name.replace("/", "__") + ".npy")
             for names in groups.values() for name in names if name != prior_key}
    missing = sorted(name for name, path in paths.items() if name not in known or not path.is_file())
    if missing:
        return {"status": "component_predictions_missing", "missing": missing, "rows": []}
    from ensemble_models import METHODS, one
    # Read only members of stored groups. There is no need to load or rank
    # all 300 raw configurations, retrain their models, or refit a V3 expert.
    arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
    if any(a.shape != (ncal + neval, 3) for a in arrays.values()):
        raise ValueError("Misaligned screening component predictions")
    rows = []
    with threadpool_limits(limits=2, user_api="blas"):
        for group, names in groups.items():
            c = np.stack([prior_cal if name == prior_key else arrays[name][:ncal] for name in names], axis=1)
            e = np.stack([prior_eval if name == prior_key else arrays[name][ncal:] for name in names], axis=1)
            for rule in METHODS:
                key = f"ensemble/{group}/{rule}"
                pred = one(rule, c, cal_y, e)
                row = baseline.row(key, pred)
                if key in summary:
                    row["screen_saved_rmse_km"] = summary[key]["rmse_km"]
                    row["replay_minus_saved_rmse_km"] = row["standalone"]["rmse_km"]-summary[key]["rmse_km"]
                rows.append(row)
    return {"status": "ready", "n_groups": len(groups), "n_replayed": len(rows),
            "reference": baseline.name, "reference_metrics": baseline.metrics, "rows": rows}


def supplement(data_dir, screen_dir, raw_names, cal_ids, eval_ids, val_ids,
               prior_cal, prior_eval, cal_y, eval_y):
    """Call only after selection lock and release of screening evaluation labels.

    Arrays are aligned with their respective ID sequences; y and priors are
    geocentric ECEF. The function reads caches, returns JSON-ready diagnostics,
    and never changes any ranking, selection, or file.
    """
    data_dir, screen_dir = Path(data_dir), Path(screen_dir)
    cal_ids, eval_ids, val_ids = map(np.asarray, (cal_ids, eval_ids, val_ids))
    prior_cal, prior_eval = np.asarray(prior_cal), np.asarray(prior_eval)
    cal_y, eval_y = np.asarray(cal_y), np.asarray(eval_y)
    if (prior_cal.shape != cal_y.shape or cal_y.shape != (len(cal_ids), 3)
            or prior_eval.shape != eval_y.shape or eval_y.shape != (len(eval_ids), 3)
            or np.intersect1d(cal_ids, eval_ids).size
            or any(len(np.unique(ids)) != len(ids) for ids in (cal_ids, eval_ids, val_ids))):
        raise ValueError("Misaligned or overlapping calibration and evaluation arrays")
    split_file = screen_dir / "split.npz"
    if split_file.exists():
        with np.load(split_file, allow_pickle=False) as split:
            if not np.array_equal(cal_ids, split["cal"]) or not np.array_equal(eval_ids, split["evaluate"]):
                raise ValueError("Screening cache row order differs from supplied row IDs")
    base = _Reference(eval_y, prior_eval, "screen_v4_calibrator_fit_60000")
    path = screen_dir / "summary.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    summary = {r["name"]: r for r in saved.get("candidates", [])}
    reported = [summary[k] for k in sorted(summary) if k.startswith("ensemble/decorrelated/")]
    return {
        "usage": "Только исследовательская диагностика после фиксации выбора. Эти результаты исключены из отбора.",
        "evaluation": "Подвыборка исходного скрининга из нулевого внешнего фолда development.",
        "n_evaluate": len(eval_ids), "n_calibration": len(cal_ids),
        "gain_definition": "Положительный выигрыш означает снижение RMSE WGS84 относительно указанного V4.",
        "correlation_definition": "Векторы ошибки направлены от истины к прогнозу в общей локальной системе восток/север WGS84.",
        "saved_decorrelated": {"rows": reported, "reference_v4_full_train": saved.get("reference_v4"),
                               "group_selection": "Группа исходного скрининга выбрана по корреляции длин ошибок на калибровке."},
        "full_outer_cache": _full_cache(data_dir/"full500"/"base-outer-0.npz", "prior/prior/v4",
                                        eval_y, eval_ids, val_ids, prior_eval),
        "literature_outer_cache": _full_cache(data_dir/"v5"/"literature-predictions-0.npz", "reference_v4",
                                              eval_y, eval_ids, val_ids, prior_eval),
        "screen_ensemble_replay": _replay(screen_dir, raw_names, len(cal_ids), len(eval_ids),
                                         prior_cal, prior_eval, cal_y, base, summary),
    }
