"""Post-lock signed WGS84 error diagnostics for raw, meta, or V4 candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from diversity_context import _Reference
from diversity_select import SEED as SCREEN_SEED, SOURCES, raw_file, screen_prior


def _json(path):
    return json.loads(path.read_text())


def _write(path, obj):
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2))
    temp.replace(path)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path):
    return np.load(path, allow_pickle=False, mmap_mode="r")


class _Report:
    def __init__(self, data, phase, base, metadata):
        self.root = data/"diversity500"/"extended"
        self.root.mkdir(parents=True, exist_ok=True)
        self.phase, self.base, self.meta = phase, base, metadata
        self.rows = {}

    def group(self, group, names, predict):
        path = self.root/f"{self.phase}-{group}.json"
        saved = _json(path) if path.exists() else {}
        rows = saved.get("rows", {}) if saved.get("metadata") == self.meta else {}
        want = [k for k in names if k not in rows]
        if want:
            start = perf_counter()
            for name, pred in predict(want):
                if name not in want or name in rows:
                    raise ValueError(f"Unexpected repeated candidate {name}")
                rows[name] = self.base.row(name, pred)
                _write(path, {"metadata": self.meta, "rows": rows})
                print(f"{self.phase}/{group}: {len(rows)}/{len(names)} ({perf_counter()-start:.1f} s)", flush=True)
        if set(rows) != set(names):
            raise ValueError(f"Incomplete group {group}")
        self.rows.update(rows)

    def finish(self, expected):
        if len(self.rows) != expected:
            raise ValueError(f"Expected {expected} rows, got {len(self.rows)}")
        path = self.root/f"{self.phase}.json"
        _write(path, {"metadata": self.meta, "reference_metrics": self.base.metrics,
                      "n_candidates": len(self.rows), "rows": [self.rows[k] for k in sorted(self.rows)]})
        print(f"Completed {len(self.rows)} {self.phase} candidates: {path}", flush=True)


def _raw(report, screen, ncal, neval):
    registry = _json(screen/"registry.json")
    for family in sorted(SOURCES):
        names = sorted(k for k, spec in registry.items() if spec["family"] == family)
        def predict(want):
            for key in want:
                a = _load(raw_file(screen, key))
                if a.shape != (ncal+neval, 3):
                    raise ValueError(f"Unexpected cached shape for {key}")
                yield key, a[ncal:]
        report.group(family, names, predict)
    report.finish(300)


def _inputs(data, y, folds, cal, evaluate, seed):
    import torch
    from run_v3 import folds_of, pca_features, take_features
    from run_v4 import load_bundle, inner_features
    fold, train, val, inner = next(iter(folds_of(folds)))
    if fold != 0:
        raise ValueError("Expected the original outer fold zero")
    x = _load(data/"dev_x.npy")
    outer = load_bundle("o0")
    pc_train, pc_val, *_ = pca_features(x, train, val, seed)
    ft, et = inner_features(outer, pc_train, fold, inner)
    fv = take_features(outer, pc_val)
    torch.cuda.empty_cache()
    return train, val, outer, ft, et, fv, np.searchsorted(train, cal), np.searchsorted(val, evaluate)


def _meta(report, prepared, y, cal):
    import meta_models
    from run_v3 import unit
    _, _, outer, ft, et, fv, ic, ie = prepared
    ec, eq = et[ic], outer["experts"][ie]
    for (view, x), (qview, q) in zip(meta_models.views(ft[ic], ec), meta_models.views(fv[ie], eq)):
        if view != qview:
            raise ValueError("Unaligned meta features")
        names = [f"meta/{view}/{m}" for m in meta_models.METHODS]
        def predict(want):
            for key in want:
                m = key.rsplit("/", 1)[1]
                pred = meta_models.fit_predict(m, x, y[cal], q, ec, eq, SCREEN_SEED)
                yield key, unit(pred)
        report.group(view, names, predict)
    report.finish(100)


def _v4(report, prepared, y, seed):
    from run_v3 import unit
    from run_v4 import modules
    train, _, outer, ft, et, fv, _, ie = prepared
    direct, learned = modules()
    for module in (*direct, *learned):
        keys = list(module.DESCRIPTIONS)
        def predict(want):
            if module in direct:
                preds = module.predict(outer)
            else:
                # Preserve original full validation batch and backend. Some
                # modules choose their numerical backend by query row count.
                preds = module.predict(ft, y[train], fv, et, outer["experts"], seed, only=want)
            for name in want:
                yield name, unit(preds[name])[ie]
        report.group(module.__name__, keys, predict)
    report.finish(50)


def run(data, phase):
    data = Path(data).resolve()
    os.environ["GEO_CV_ROOT"] = str(data)
    # Colab may hold the V3/V4 sources beside the cached arrays or in the
    # source parent directory when the repository checkout is used.
    sys.path.extend([str(data), str(Path(__file__).resolve().parent.parent)])
    screen = data/"screen500"
    lock = data/"diversity500"/"selection.json"
    split_hash = _digest(screen/"split.npz")
    if _json(lock)["split_sha256"] != split_hash:
        raise ValueError("Diversity selection does not match the screening split")
    with np.load(screen/"split.npz", allow_pickle=False) as split:
        fit, cal, evaluate = (split[k] for k in ("fit", "cal", "evaluate"))
    folds = _load(data/"folds.npy")
    train, val = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    if (tuple(map(len, (fit, cal, evaluate))) != (60000, 20000, 12500)
            or not np.isin(evaluate, val).all() or not np.isin(np.r_[fit, cal], train).all()
            or np.intersect1d(fit, cal).size):
        raise ValueError("Unexpected screening partitions")
    y = _load(data/"dev_y.npy")
    prepared = None
    if phase == "v4":
        from run_v4 import SEED as V4_SEED
        prepared = _inputs(data, y, folds, cal, evaluate, V4_SEED)
        _, _, outer, ft, et, fv, _, ie = prepared
        import calibration_ideas
        prior = calibration_ideas.predict(ft, y[train], fv, et, outer["experts"], V4_SEED,
                                          only="cal_interaction")["cal_interaction"][ie]
        label = "v4_cal_interaction_full_fold_fit_170000"
    else:
        prior = screen_prior(data, train, val, folds, fit, cal, evaluate, y[fit])[len(cal):]
        label = "screen_v4_calibrator_fit_60000"
    base = _Reference(y[evaluate], prior, label)
    metadata = {"phase": phase, "selection_sha256": _digest(lock), "split_sha256": split_hash,
                "code_sha256": _digest(Path(__file__)), "reference": label, "n": len(evaluate),
                "fixed_candidate_weight": .25, "exploratory_only": True,
                "scope": "Диагностика после фиксации набора; результаты не участвуют в отборе.",
                "gain_definition": "Снижение RMSE WGS84 относительно указанного V4, в километрах."}
    report = _Report(data, phase, base, metadata)
    if phase == "raw":
        _raw(report, screen, len(cal), len(evaluate))
    elif phase == "meta":
        prepared = _inputs(data, y, folds, cal, evaluate, SCREEN_SEED)
        _meta(report, prepared, y, cal)
    else:
        _v4(report, prepared, y, V4_SEED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["raw", "meta", "v4"])
    parser.add_argument("--data", default=os.environ.get("GEO_CV_ROOT", "/content/train250k-cv"))
    args = parser.parse_args()
    run(args.data, args.phase)
