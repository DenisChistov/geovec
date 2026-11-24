"""Refit the locked full-search winner and score its committed holdout predictions."""

import hashlib
import json
from pathlib import Path

import numpy as np

import full_search as search
from experiment import metrics, save_json
from literature import METHODS as PAPER_METHODS, fit_cells, fit_ranker
from run_v3 import blend, fingerprint, pca_features, unit


FINAL_SEED = search.SEED + 5
LITERATURE_SEED = 823748 + 5
V4_SEED = 723748 + 5
N_DEV = 212500
N_TEST = 37500


def _json(path):
    return json.loads(path.read_text())


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _xy(xyz, rows):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.shape != (rows, 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid prediction shape or nonfinite coordinates")
    if not np.allclose(np.linalg.norm(xyz, axis=1), 1, atol=2e-5, rtol=0):
        raise ValueError("Expected unit WGS84 ECEF predictions")
    return xyz


class _FeatureRows:
    """Index two disjoint memory maps without copying the 250,000-row dataset."""

    ndim = 2

    def __init__(self, dev, test):
        self.dev, self.test = dev, test
        self.shape = (len(dev) + len(test), dev.shape[1])

    def __getitem__(self, indices):
        idx = np.asarray(indices, dtype=np.intp)
        if np.all((0 <= idx) & (idx < len(self.dev))):
            source, local = self.dev, idx
        elif np.all((len(self.dev) <= idx) & (idx < self.shape[0])):
            source, local = self.test, idx - len(self.dev)
        else:
            raise IndexError("Training and test feature blocks must be indexed separately")
        if len(local) == len(source) and np.array_equal(local, np.arange(len(source))):
            return source
        return source[local]


def winner_plan(winner, selection, diversity=None, tuned=None):
    plan = {"members": [], "fusion": None, "joint": False, "prior_blend": False, "literature": None}
    if winner == "reference_v4":
        plan["members"] = [search.PRIOR]
    elif winner.startswith("literature:"):
        name = winner.removeprefix("literature:")
        plan["prior_blend"] = name.endswith("_plus_v4")
        name = name.removesuffix("_plus_v4")
        if name not in PAPER_METHODS:
            raise ValueError(f"Unknown literature winner {name}")
        plan["literature"] = name
    elif winner.startswith("tuned:"):
        if not tuned or not tuned.get("members") or winner not in tuned["rules"]:
            raise ValueError(f"Missing merged tuning selection for {winner}")
        plan["members"] = list(tuned["members"])
        plan["tuned"] = winner
    elif winner.startswith("diverse:"):
        if not diversity or winner not in diversity["rules"]:
            raise ValueError(f"Missing merged diversity selection for {winner}")
        plan["members"] = list(diversity["members"])
        plan["diverse"] = winner
    elif winner.startswith("strong:"):
        from strong_fusion import member_ids
        plan["members"] = list(member_ids(selection))
        plan["strong"] = winner
    elif winner.startswith("joint:"):
        rule = winner.removeprefix("joint:")
        if rule not in search.JOINT_RULES:
            raise ValueError(f"Unknown joint rule {rule}")
        plan.update(members=selection["joint_members"], fusion=rule, joint=True)
    else:
        name = winner.removeprefix("blend:")
        if name not in selection["finalists"]:
            raise ValueError(f"Unknown full-search finalist {name}")
        plan["prior_blend"] = winner.startswith("blend:")
        if name in selection["screen_ensembles"]:
            plan.update(members=selection["screen_ensembles"][name], fusion=name.rsplit("/", 1)[1])
        else:
            plan["members"] = [name]
    return plan


def _cached_v4(data, split, count):
    path = data / "v4"
    required = ["v4/cv-summary.json", "v4/test-predictions.npz", "v4/best-pca.npz",
                "v4/cal_interaction.npz", "v3/cells-512-a.pt", "v3/cells-512-b.pt",
                "v3/cells-1024.pt", "v3/direct.pt"]
    if any(not (data / file).is_file() for file in required):
        return None
    prior = _json(path / "cv-summary.json")
    if prior["winner"] != "cal_interaction" or prior["split_sha256"] != split:
        raise ValueError("The expected all-development V4 model differs from the saved model")
    with np.load(path / "test-predictions.npz", allow_pickle=False) as saved:
        return _xy(saved["xyz"], count)


def _ensure_final_v3(data, x, y, train, test):
    from run_v3 import make_bundle, SPLIT_SEED

    source = data / "v3"
    required = ("bundle-final.npz", "cells-512-a.pt", "cells-1024.pt", "cells-512-b.pt", "direct.pt")
    if any(not (source / file).is_file() for file in required):
        # Training routines access y only at the supplied training indices.
        # The label array deliberately has no positions for any test row.
        make_bundle("final", x, y, train, test, SPLIT_SEED + 205, save=True)


def _v3_features(data, folds, train_ids, test_ids, x, model_dir, need_pca, pca_seed=FINAL_SEED):
    """Use the five V3 outer folds for dev and its all-dev fit for test."""
    source = data / "v3"
    with np.load(source / "bundle-final.npz", allow_pickle=False) as bundle:
        test_f = bundle["features"]
        test_e = bundle["experts"]
    width = test_f.shape[1]
    if need_pca:
        pt, pq, center, basis = pca_features(x, train_ids, test_ids, pca_seed)
        _npz(model_dir / "meta-pca.npz", center=center, projection=basis)
        test_f = np.column_stack((test_f, pq))
    dev_f = np.empty((len(folds), test_f.shape[1]), np.float32)
    dev_e = np.empty((len(folds), *test_e.shape[1:]), np.float32)
    if need_pca:
        dev_f[:, width:] = pt
    for i in range(5):
        rows = np.flatnonzero(folds == i)
        with np.load(source / f"bundle-o{i}.npz", allow_pickle=False) as bundle:
            dev_f[rows, :width] = bundle["features"]
            dev_e[rows] = bundle["experts"]
    return dev_f, dev_e, test_f, test_e


def _oof_predictions(root, folds, members):
    oof = {k: np.empty((len(folds), 3), np.float32) for k in members}
    for fold in range(5):
        ids = np.flatnonzero(folds == fold)
        with np.load(root / f"base-outer-{fold}.npz", allow_pickle=False) as archive:
            for k in members:
                oof[k][ids] = _xy(archive[k], len(ids))
    return oof


def _fusion(plan, y, folds, query, root, model_dir):
    from fusion_checkpoint import save_rule, predict_saved

    ids = plan["members"]
    raw = search.trainable_members(ids)
    oof = _oof_predictions(root, folds, raw)
    pred = search.apply_fusion(plan["fusion"], ids, oof, query, y, joint=plan["joint"])
    file = model_dir / "fusion-rule.npz"
    save_rule(file, plan["fusion"], np.stack([oof[k] for k in raw], axis=1), y)
    replay = unit(predict_saved(file, np.stack([query[k] for k in raw], axis=1)))
    fixed = [k for k in ids if k not in raw]
    if plan["joint"]:
        meta = next(k for k in fixed if k != search.PRIOR)
        replay = blend(blend(replay, query[meta], .25), query[search.PRIOR], 1/3)
    else:
        for j, k in enumerate(fixed):
            replay = blend(replay, query[k], 1/(len(raw) + j + 1))
    if not np.allclose(pred, replay, atol=2e-5, rtol=0):
        raise ValueError("Saved fusion model differs from the five-fold fitting rule")
    return pred


def _fit_locked(data, root, model_dir, lock, folds):
    winner, split = lock["winner"], lock["split_sha256"]
    plan = winner_plan(winner, lock["selection"], lock.get("diversity_fusion"), lock.get("tuned_fusion"))
    needed = list(dict.fromkeys(plan["members"] + ([search.PRIOR] if plan["prior_blend"] else [])))
    query = {}
    dependencies = []
    incumbent_source = None
    if search.PRIOR in needed:
        prior = _cached_v4(data, split, N_TEST)
        if prior is not None:
            query[search.PRIOR] = prior
            incumbent_source = "Original all-development V4 cal_interaction predictions"
            dependencies += ["v4/test-predictions.npz", "v4/cv-summary.json", "v4/cal_interaction.npz",
                             "v4/best-pca.npz"]
    refit_prior = search.PRIOR in needed and search.PRIOR not in query
    fitted = [k for k in needed if k != search.PRIOR]
    literature = plan["literature"]
    seed_by_component = {k: FINAL_SEED for k in fitted}
    y = None
    if fitted or literature or refit_prior:
        dev_x = np.load(data / "dev_x.npy", mmap_mode="r", allow_pickle=False)
        test_x = np.load(data / "test_x.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(data / "dev_y.npy", mmap_mode="r", allow_pickle=False)
        if dev_x.shape != (N_DEV, 1280) or test_x.shape != (N_TEST, 1280) or y.shape != (N_DEV, 3):
            raise ValueError("Unexpected final feature or development coordinate dimensions")
        x = _FeatureRows(dev_x, test_x)
        train = np.arange(len(dev_x))
        test = np.arange(len(dev_x), x.shape[0])
        need_meta = any(k.startswith("meta/") for k in fitted)
        need_bundles = need_meta or literature == "candidate_ranker"
        ft, et, fq, eq = (None,) * 4
        if need_bundles or refit_prior:
            _ensure_final_v3(data, x, y, train, test)
            dependencies += ["v3/bundle-final.npz", *[f"v3/bundle-o{i}.npz" for i in range(5)]]
        if refit_prior:
            from calibration_ideas import predict as calibrate
            destination = model_dir / "v4-refit"
            destination.mkdir(exist_ok=True)
            t, e, q, qe = _v3_features(data, folds, train, test, x, destination, True, V4_SEED)
            query[search.PRIOR] = unit(calibrate(t, y, q, e, qe, V4_SEED, only="cal_interaction",
                                                 save_dir=destination)["cal_interaction"])
            del t, e, q, qe
            incumbent_source = "Refit cal_interaction on all development V3 outer-fold features with original V4 seed"
            seed_by_component[search.PRIOR] = V4_SEED
        if need_bundles:
            ft, et, fq, eq = _v3_features(data, folds, train, test, x, model_dir, need_meta)
        if fitted:
            query.update(search.predict_split(fitted, x, y, train, test, ft, et, fq, eq, FINAL_SEED, save_dir=model_dir))
            views = {k.split("/")[1] for k in fitted if k.startswith(("neural/", "geo/", "retrieval/"))}
            if any(not (model_dir / f"feature-{view}.pt").is_file() for view in views):
                raise FileNotFoundError("The selected feature transform was not saved by predict_split")
            if any(k.startswith("retrieval/") for k in fitted):
                dependencies += ["dev_x.npy", "dev_y.npy"]
        if literature:
            offset = list(PAPER_METHODS).index(literature)
            seed = LITERATURE_SEED + offset
            seed_by_component["literature:" + literature] = seed
            save = model_dir / f"literature-{literature}.pt"
            if literature == "candidate_ranker":
                pred = fit_ranker({"features": ft, "experts": et}, y,
                                  {"features": fq, "experts": eq}, seed, save_path=save)
            else:
                pred = fit_cells(literature, x, y, train, test, seed, save_path=save)
            pred = unit(pred)
    if plan.get("tuned"):
        from tune_fusion import final_prediction, predict_saved
        dest = model_dir / "tuned-rule.npz"
        pred = unit(final_prediction(winner, lock, query, y, folds, save_path=dest))
        replay = unit(predict_saved(dest, query))
        if not np.allclose(pred, replay, atol=2e-5, rtol=0):
            raise ValueError("The saved tuned fusion differs from its fitted predictions")
        dependencies += [f"full500/base-outer-{i}.npz" for i in range(5)]
        dependencies += [f"full500/diversity/base-outer-{i}.npz" for i in range(5)]
    elif plan.get("diverse"):
        from diversity_cv import final_prediction, predict_saved
        dest = model_dir / "diversity-rule.npz"
        pred = unit(final_prediction(winner, plan["members"], query, y, folds, save_path=dest))
        replay = unit(predict_saved(dest, query))
        if not np.allclose(pred, replay, atol=2e-5, rtol=0):
            raise ValueError("The saved diversity fusion differs from its fitted predictions")
        dependencies += [f"full500/base-outer-{i}.npz" for i in range(5)]
        dependencies += [f"full500/diversity/base-outer-{i}.npz" for i in range(5)]
    elif plan.get("strong"):
        from strong_fusion import final_prediction, predict_saved
        dest = model_dir / "strong-rule.npz"
        pred = unit(final_prediction(winner, lock["selection"], query, y, folds, save_path=dest))
        replay = unit(predict_saved(dest, query))
        if not np.allclose(pred, replay, atol=2e-5, rtol=0):
            raise ValueError("The saved strong fusion rule does not reproduce its predictions")
        dependencies += [f"full500/base-outer-{i}.npz" for i in range(5)]
    elif plan["fusion"]:
        pred = _fusion(plan, y, folds, query, root, model_dir)
        dependencies += [f"full500/base-outer-{i}.npz" for i in range(5)]
    elif not literature:
        pred = query[plan["members"][0]]
    if plan["prior_blend"]:
        pred = blend(query[search.PRIOR], pred, .33)
    if search.PRIOR in needed or any(k.startswith("meta/") for k in fitted) or literature == "candidate_ranker":
        dependencies += ["v3/cells-512-a.pt", "v3/cells-1024.pt", "v3/cells-512-b.pt",
                         "v3/direct.pt", "dev_x.npy", "dev_y.npy"]
    layer = {"winner": winner, "plan": plan, "component_training_seeds": seed_by_component,
             "new_components_refit_on": N_DEV if fitted or literature or refit_prior else 0,
             "incumbent_source": incumbent_source,
             "refit_data": "Development embeddings and labels; V3 outer OOF features for trained meta heads",
             "fusion_training": "Specialist predictions from five outer folds, reassembled in dev row order"
                                if plan["fusion"] or any(k in plan for k in ("strong","diverse","tuned")) else None,
             "fusion_raw_order": (plan["members"] if any(k in plan for k in ("strong","diverse","tuned")) else
                                  search.trainable_members(plan["members"]) if plan["fusion"] else []),
             "final_external_blend": {"incumbent": .67, "challenger": .33, "interpolation": "spherical"}
                                     if plan["prior_blend"] else None,
             "fit_and_inference_dependencies_relative_to_data": list(dict.fromkeys(dependencies)),
             "retrieval_replay": "Apply the saved feature transform to the original ordered development bank and "
                                 "queries, then use retrieval-{view}.pt to replay the fitted retrieval calibration."
                                 if any(k.startswith("retrieval/") for k in fitted) else None,
             "files": sorted(str(p.relative_to(model_dir)) for p in model_dir.rglob("*") if p.is_file())}
    return _xy(pred, N_TEST), layer


def main():
    root, data = search.ROOT, search.DATA
    lock_path = root / "cv-summary.json"
    lock = _json(lock_path)
    split = fingerprint()
    if lock["split_sha256"] != split or lock["winner"] != lock["candidates"][0]["name"]:
        raise ValueError("The final winner or split does not match the full five-fold CV lock")
    if len(lock["candidates"][0]["fold_rmse_km"]) != 5:
        raise ValueError("Final refit requires all five development folds")
    strong_names = set(lock.get("strong_fusion", {}).get("rules", {}))
    if not strong_names or not strong_names.issubset(row["name"] for row in lock["candidates"]):
        raise ValueError("The final CV lock must include the evaluated strong fusion candidates")
    diverse_names = set(lock.get("diversity_fusion", {}).get("rules", {}))
    if not diverse_names or not diverse_names.issubset(row["name"] for row in lock["candidates"]):
        raise ValueError("The final CV lock must include the evaluated diversity fusion candidates")
    tuned_names = set(lock.get("tuned_fusion", {}).get("rules", {}))
    if not tuned_names or not tuned_names.issubset(row["name"] for row in lock["candidates"]):
        raise ValueError("The final CV lock must include the evaluated ensemble hyperparameter candidates")
    folds = np.load(data / "folds.npy", allow_pickle=False)
    if folds.shape != (N_DEV,) or np.bincount(folds).tolist() != [42500] * 5:
        raise ValueError("Unexpected frozen development split")
    digest = _sha(lock_path)
    winner = lock["winner"]
    prediction_path, provenance_path = root / "test-predictions.npz", root / "final-provenance.json"
    results_path = root / "results.json"
    if prediction_path.exists():
        with np.load(prediction_path, allow_pickle=False) as saved:
            if str(saved["cv_sha256"]) != digest or str(saved["winner"]) != winner or str(saved["split_sha256"]) != split:
                raise ValueError("Existing final predictions belong to a different CV winner or split")
            pred = _xy(saved["xyz"], N_TEST)
        provenance = _json(provenance_path)
        if provenance["cv_sha256"] != digest:
            raise ValueError("Final provenance differs from the CV lock")
    else:
        if results_path.exists():
            raise ValueError("Final results exist without their committed predictions")
        model_dir = root / "final-models" / digest[:16]
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Refitting locked winner {winner} on {N_DEV:,} development rows", flush=True)
        pred, provenance = _fit_locked(data, root, model_dir, lock, folds)
        provenance.update(cv_sha256=digest, split_sha256=split, training_rows=N_DEV, query_rows=N_TEST,
                          models_relative_to_full500=str(model_dir.relative_to(root)),
                          model_file_sha256={name: _sha(model_dir / name) for name in provenance["files"]},
                          test_row_order="Unmodified test_x.npy order, identified by test_rows.npy",
                          selection="Minimum pooled squared WGS84 geodesic distance across five development folds")
        save_json(provenance_path, provenance)
        _npz(prediction_path, xyz=pred, winner=winner, split_sha256=split, cv_sha256=digest)
    pred_sha = _sha(prediction_path)
    if results_path.exists():
        result = _json(results_path)
        if result["cv_sha256"] != digest or result["predictions_sha256"] != pred_sha:
            raise ValueError("The recorded score does not match the saved predictions and CV lock")
        print(json.dumps(result, indent=2), flush=True)
        return
    # The first access to any test labels occurs after the winning predictions
    # and the exact development-only model selection lock are committed to disk.
    with np.load(prediction_path, allow_pickle=False) as saved:
        committed = _xy(saved["xyz"], N_TEST)
    truth = np.load(data / "test_latlon.npy", mmap_mode="r", allow_pickle=False)
    if truth.shape != (N_TEST, 2):
        raise ValueError("Unexpected test label shape")
    score = metrics(committed, truth)
    historical_path = data / "v4" / "results.json"
    historical = (_json(historical_path)["test"] if historical_path.is_file() else
                  {"mae_km": 266.6239811486437, "rmse_km": 975.2377056762869,
                   "mse_km2": 951088.5825727478, "n": 37500})
    result = {"winner": winner, "test": score, "cv": lock["candidates"][0], "split_sha256": split,
              "cv_sha256": digest, "predictions_sha256": pred_sha, "development_rows": N_DEV,
              "test_rows": N_TEST, "component_training_seeds": provenance["component_training_seeds"],
              "provenance": provenance_path.name,
              "historical_reference": historical,
              "historical_reference_source": "v4/results.json" if historical_path.is_file() else "previously published V4 aggregate"}
    save_json(results_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
