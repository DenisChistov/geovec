"""Predict WGS84 locations from the saved full-search winner.

Input columns are 512 GeoCLIP dimensions followed by 768 StreetCLIP dimensions.
The model directory is full500, including its CV summary, provenance and the
selected checkpoints under final-models. V3-based experts also require the
original V3 checkpoints and ordered development reference arrays. Retrieval
uses its frozen transform, calibration and original development bank. For
older runs without persisted retrieval calibration, the original training-only
index statistics are reconstructed using the recorded seed.
"""

import argparse
from functools import cached_property
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

# In the published project, the shared V3/V4 modules sit directly above screen/.
_SRC = Path(__file__).resolve().parent.parent
if (_SRC / "predict_v3.py").is_file() and str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))

from experiment import decode
from predict_v3 import normalize_features, build_bundle, _reference_data, _project_saved
from run_v3 import unit, take_features, blend


PRIOR = "prior/prior/v4"
PAPER = "paper/distance_risk"


def _digest(path):
    with Path(path).open("rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


class _Inference:
    def __init__(self, x, reference, full_dir):
        self.x = normalize_features(x)
        self.reference = Path(reference)
        root = Path(full_dir) if full_dir is not None else self.reference / "full500"
        self.record = json.loads((root / "final-provenance.json").read_text())
        lock_path = root / "cv-summary.json"
        lock = json.loads(lock_path.read_text())
        if (self.record["winner"] != lock["winner"] or self.record["cv_sha256"] != _digest(lock_path)
                or self.record["split_sha256"] != lock["split_sha256"]):
            raise ValueError("The final model and CV selection disagree")
        self.model_dir = root / self.record["models_relative_to_full500"]
        for name, signature in self.record.get("model_file_sha256", {}).items():
            if _digest(self.model_dir / name) != signature:
                raise ValueError(f"Model file does not match final provenance: {name}")
        self.plan = self.record["plan"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def bank(self):
        x, y = _reference_data(self.reference)
        if len(x) != self.record["training_rows"]:
            raise ValueError("Reference bank row count differs from the all-development model")
        return x, y

    @cached_property
    def bundle(self):
        return build_bundle(self.x, self.reference / "v3", self.reference)

    def _v4(self):
        from calibration_ideas import predict_saved

        if PRIOR in self.record["component_training_seeds"]:
            model_dir = self.model_dir / "v4-refit"
            pca = model_dir / "meta-pca.npz"
        else:
            model_dir = self.reference / "v4"
            pca = model_dir / "best-pca.npz"
        projection = _project_saved(pca, self.x, self.device)
        f = take_features(self.bundle, projection)
        return unit(predict_saved(f, self.bundle["experts"], model_dir / "cal_interaction.npz"))

    def _paper(self, method, filename):
        from replay_literature import predict

        return unit(predict(method, self.model_dir / filename, self.x,
                            self.bundle if method == "candidate_ranker" else None))

    def _raw(self, keys, answers):
        import features
        import geo_models
        import replay_heads
        import retrieval_models

        views = list(dict.fromkeys(key.split("/")[1] for key in keys))
        for view in views:
            selected = [k for k in keys if k.split("/")[1] == view]
            z = features.transform_saved(self.x, view, self.model_dir)
            retrieval = [k for k in selected if k.startswith("retrieval/")]
            if retrieval:
                seed = self.record["component_training_seeds"][retrieval[0]]
                if any(self.record["component_training_seeds"][k] != seed for k in retrieval):
                    raise ValueError("Retrieval components sharing the view have different reference seeds")
                x_ref, y_ref = self.bank
                bank = features.transform_saved(x_ref, view, self.model_dir)
                state = self.model_dir / f"retrieval-{view}.pt"
                if not state.is_file():
                    if state.name in self.record.get("files", []):
                        raise FileNotFoundError(state)
                    warnings.warn(f"Rebuilding legacy retrieval calibration for {view} from the original "
                                  f"ordered development bank, seed {seed}.", RuntimeWarning, stacklevel=2)
                pred = retrieval_models.predict(bank, y_ref, z, seed, state_path=state if state.is_file() else None)
                for key in retrieval:
                    answers[key] = unit(pred[key.split("/")[2]])
                del bank, pred
            for key in selected:
                family = key.split("/")[0]
                path = self.model_dir / key.replace("/", "__")
                if family == "geo":
                    answers[key] = unit(geo_models.predict_saved(z, path))
                elif family == "neural":
                    answers[key] = unit(replay_heads.neural(path.with_suffix(".pt"), z))

    def _meta(self, keys, answers):
        import meta_models
        from replay_heads import meta

        projection = _project_saved(self.model_dir / "meta-pca.npz", self.x, self.device)
        e = self.bundle["experts"]
        for view, z in meta_models.views(take_features(self.bundle, projection), e):
            for key in keys:
                if key.split("/")[1] == view:
                    path = self.model_dir / (key.replace("/", "__") + ".pt")
                    answers[key] = unit(meta(path, z, e))

    def _fuse(self, answers):
        plan = self.plan
        if plan.get("tuned"):
            from tune_fusion import predict_saved
            return unit(predict_saved(self.model_dir / "tuned-rule.npz", answers))
        if plan.get("diverse"):
            from diversity_cv import predict_saved
            return unit(predict_saved(self.model_dir / "diversity-rule.npz", answers))
        if plan.get("strong"):
            from strong_fusion import predict_saved
            return unit(predict_saved(self.model_dir / "strong-rule.npz", answers))
        if plan["fusion"]:
            from fusion_checkpoint import predict_saved
            raw = self.record["fusion_raw_order"]
            predictions = np.stack([answers[k] for k in raw], axis=1)
            pred = unit(predict_saved(self.model_dir / "fusion-rule.npz", predictions))
            fixed = [k for k in plan["members"] if k not in raw]
            if plan["joint"]:
                meta = next(k for k in fixed if k != PRIOR)
                return blend(blend(pred, answers[meta], .25), answers[PRIOR], 1/3)
            for j, key in enumerate(fixed):
                pred = blend(pred, answers[key], 1/(len(raw) + j + 1))
            return pred
        if plan["literature"]:
            name = plan["literature"]
            return self._paper(name, f"literature-{name}.pt")
        return answers[plan["members"][0]]

    def run(self):
        if not len(self.x):
            return np.empty((0, 3), np.float32)
        required = list(dict.fromkeys(self.plan["members"] + ([PRIOR] if self.plan["prior_blend"] else [])))
        answers = {}
        raw = [k for k in required if k.startswith(("neural/", "geo/", "retrieval/"))]
        if raw:
            self._raw(raw, answers)
        if PRIOR in required:
            answers[PRIOR] = self._v4()
        if PAPER in required:
            answers[PAPER] = self._paper("distance_risk", "paper-distance-risk.pt")
        meta = [k for k in required if k.startswith("meta/")]
        if meta:
            self._meta(meta, answers)
        if set(answers) != set(required):
            raise ValueError(f"Unsupported saved expert keys: {set(required) - set(answers)}")
        pred = self._fuse(answers)
        return blend(answers[PRIOR], pred, .33) if self.plan["prior_blend"] else pred


def predict_xyz(X, reference, full_dir=None):
    """Return [N,3] unit geocentric ECEF directions for new embedding rows."""
    return _Inference(X, reference, full_dir).run()


def predict(X, reference, full_dir=None):
    """Return WGS84 geodetic latitude and longitude, in that order, in degrees."""
    return np.column_stack(decode(predict_xyz(X, reference, full_dir)))


def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--input", type=Path, required=True, help="Floating-point N x 1280 NumPy array")
    cmd.add_argument("--output", type=Path, required=True, help="Output .csv or .npy, latitude then longitude")
    cmd.add_argument("--reference", type=Path, default=Path(os.environ.get("GEO_CV_ROOT", "/content/train250k-cv")),
                     help="Data root containing development reference arrays and V3/V4 weights when required")
    cmd.add_argument("--full-dir", type=Path, help="Full-search artifact directory; default REFERENCE/full500")
    args = cmd.parse_args()
    ext = args.output.suffix.lower()
    if ext not in {".csv", ".npy"}:
        cmd.error("--output must end in .csv or .npy")
    coords = predict(np.load(args.input, allow_pickle=False, mmap_mode="r"), args.reference, args.full_dir)
    if ext == ".csv":
        np.savetxt(args.output, coords, delimiter=",", header="latitude,longitude", comments="", fmt="%.10f")
    else:
        with args.output.open("wb") as output:
            np.save(output, coords, allow_pickle=False)
    print(f"Saved {len(coords)} WGS84 predictions to {args.output}")


if __name__ == "__main__":
    main()
