"""Inference from saved final V3 checkpoints and the training reference index."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from cell_v2 import CellClassifier
from deep_v2 import GeoRegressor
from experiment import decode
from posterior_v3 import _infer
from refine_v3 import refine
from run_v3 import SPLIT_SEED, combine_predictions, take_features, unit


def normalize_features(X):
    """Return float32 embeddings with independent GeoCLIP/StreetCLIP L2 norms."""
    values = np.array(X, dtype=np.float32, order="C", copy=True)
    if values.ndim != 2 or values.shape[1] != 1280:
        raise ValueError("Expected [rows,1280]: GeoCLIP (512) followed by StreetCLIP (768)")
    for start in range(0, len(values), 4096):
        batch = values[start:start + 4096]
        for block in (batch[:, :512], batch[:, 512:]):
            norms = np.linalg.norm(block.astype(np.float64), axis=1)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Every embedding block must be finite and nonzero")
            # Preserve prepared float32 inputs exactly when already normalized.
            change = np.abs(norms - 1.0) > 2e-6
            if change.any():
                block[change] /= norms[change, None]
    return values


class _ReferenceQueryRows:
    """Serve disjoint retrieval index/query rows without concatenating on CPU."""

    ndim = 2

    def __init__(self, references, queries):
        self.references = references
        self.queries = queries
        self.shape = (len(references) + len(queries), references.shape[1])

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        if np.all(rows < len(self.references)):
            return self.references[rows]
        if np.all(rows >= len(self.references)):
            return self.queries[rows - len(self.references)]
        raise ValueError("Retrieval must access disjoint reference and query batches")


def _gpu_features(X, scale, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return torch.as_tensor(X, dtype=dtype, device=device) * float(scale)


def _posterior(path, X, reference_size, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    k = int(checkpoint["k"])
    model = CellClassifier(X.shape[1], k).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    counts = checkpoint["cluster_counts"]
    if int(counts.sum()) != reference_size:
        raise ValueError(f"Reference row count differs from training for {path.name}")
    return _infer(
        model, _gpu_features(X, checkpoint["input_scale"], device),
        checkpoint["centers"].to(device), counts.to(device), checkpoint["cell_radii_rad"].to(device),
    )


def _direct(path, X, reference_y, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GeoRegressor(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    features = _gpu_features(X, checkpoint["input_scale"], device)
    with torch.inference_mode(), torch.autocast("cuda", enabled=device.type == "cuda"):
        xyz = torch.cat([model(batch)[0] for batch in features.split(checkpoint["config"]["batch_size"])]).cpu().numpy()
    missing = np.linalg.norm(xyz, axis=1) <= 1e-8
    if missing.any():
        fallback = np.asarray(reference_y.mean(axis=0), dtype=np.float32)
        if np.linalg.norm(fallback) <= 1e-8:
            fallback = np.array([1, 0, 0], dtype=np.float32)
        xyz[missing] = fallback / np.linalg.norm(fallback)
    return unit(xyz)


def _reference_data(reference_dir):
    root = Path(reference_dir)
    X = np.load(root / "dev_x.npy", mmap_mode="r", allow_pickle=False)
    Y = np.load(root / "dev_y.npy", mmap_mode="r", allow_pickle=False)
    if X.ndim != 2 or X.shape[1] != 1280 or not len(X) or Y.shape != (len(X), 3):
        raise ValueError("The reference data must be the original development arrays dev_x.npy and dev_y.npy")
    if X.dtype != np.float32 or Y.dtype != np.float32:
        raise ValueError("Reference arrays must use the float32 prepared training format")
    if not np.isfinite(Y).all() or not np.allclose(np.linalg.norm(Y, axis=1), 1, atol=2e-5):
        raise ValueError("Reference coordinates must be finite unit WGS84 ECEF directions")
    return X, Y


def _project_saved(path, X, device):
    with np.load(path, allow_pickle=False) as saved:
        center = saved["center"]
        projection = saved["projection"]
    if center.shape != (X.shape[1],) or projection.ndim != 2 or projection.shape[0] != X.shape[1]:
        raise ValueError("Malformed PCA checkpoint")
    mean = torch.as_tensor(center, device=device, dtype=torch.float32)
    basis = torch.as_tensor(projection, device=device, dtype=torch.float32)
    return np.concatenate([
        ((torch.as_tensor(batch, device=device) - mean) @ basis).cpu().numpy()
        for batch in np.array_split(X, max(1, (len(X) + 8191) // 8192))
    ])


def build_bundle(X, model_dir, reference_dir):
    """Build all pretrained V3 forecasts for normalized query embeddings."""
    model_dir = Path(model_dir)
    reference_x, reference_y = _reference_data(reference_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = _posterior(model_dir / "cells-512-a.pt", X, len(reference_x), device)
    b = _posterior(model_dir / "cells-1024.pt", X, len(reference_x), device)
    c = _posterior(model_dir / "cells-512-b.pt", X, len(reference_x), device)
    d = _direct(model_dir / "direct.pt", X, reference_y, device)
    train = np.arange(len(reference_x), dtype=np.intp)
    query = np.arange(len(reference_x), len(reference_x) + len(X), dtype=np.intp)
    # Refinement reads Y only at train indices. No query or test labels exist.
    reference_query = _ReferenceQueryRows(reference_x, X)
    ref = refine(reference_query, reference_y, train, query, a, seed=SPLIT_SEED + 205)
    return combine_predictions(a, b, c, d, ref)


def predict_xyz(X, model_dir, reference_dir):
    """Predict unit geocentric directions with the cross-validation-selected V3 model."""
    X = normalize_features(X)
    if not len(X):
        return np.empty((0, 3), dtype=np.float32)
    model_dir = Path(model_dir)
    locked = json.loads((model_dir / "cv-summary.json").read_text())
    winner = locked["winner"]
    bundle = build_bundle(X, model_dir, reference_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if winner.startswith("fusion_"):
        from fusion_v3 import predict_saved

        projection = _project_saved(model_dir / "best-pca.npz", X, device)
        features = take_features(bundle, projection)
        xyz = predict_saved(features, bundle["experts"], model_dir / "best-fusion.cbm")
    else:
        if winner in {"features", "experts"} or winner not in bundle:
            raise ValueError(f"Unknown V3 model selection {winner!r}")
        xyz = bundle[winner]
    return unit(xyz)


def predict(X, model_dir, reference_dir):
    """Return [rows,2] float64 geodetic latitude/longitude in degrees (WGS84)."""
    latitude, longitude = decode(predict_xyz(X, model_dir, reference_dir))
    return np.column_stack((latitude, longitude))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input .npy, [rows,1280] embeddings")
    parser.add_argument("--output", required=True, type=Path, help="Output .npy or .csv, latitude then longitude")
    parser.add_argument("--reference", type=Path, default=Path(os.environ.get("GEO_CV_ROOT", "/content/train250k-cv")), help="Original development directory with dev_x.npy and dev_y.npy")
    parser.add_argument("--models", type=Path, help="Final V3 checkpoint directory (default: REFERENCE/v3)")
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".npy", ".csv"}:
        parser.error("--output must end with .npy or .csv")
    coordinates = predict(np.load(args.input, allow_pickle=False, mmap_mode="r"),
                          args.models or args.reference / "v3", args.reference)
    if args.output.suffix.lower() == ".csv":
        np.savetxt(args.output, coordinates, fmt="%.10f", delimiter=",", header="latitude,longitude", comments="")
    else:
        with args.output.open("wb") as output:
            np.save(output, coordinates)
    print(f"Saved {len(coordinates)} WGS84 predictions to {args.output}")


if __name__ == "__main__":
    main()
