"""Inference from checkpoints created by literature.fit_cells and literature.fit_ranker."""
import numpy as np
import torch
from torch import nn

import literature


def _ranker(state, bundle, n, device):
    if bundle is None:
        raise ValueError("candidate_ranker requires the V3 features and experts bundle")
    features = np.asarray(bundle["features"])
    experts = np.asarray(bundle["experts"])
    if (features.ndim != 2 or features.shape[0] != n or features.shape[1] < 188
            or experts.shape != (n, 7, 3)):
        raise ValueError("Misaligned V3 candidate bundle")
    # fit_ranker saves a bare Sequential state_dict, with no model class in the checkpoint.
    width = state["0.weight"].shape[1]
    model = nn.Sequential(nn.Linear(width, 64), nn.LayerNorm(64), nn.GELU(),
                          nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)).to(device)
    model.load_state_dict(state)
    model.eval()
    result = np.empty((n, 3), dtype=np.float32)
    for start in range(0, n, 2048):
        rows = slice(start, start + 2048)
        points, values = literature.candidate_inputs({"features": features[rows], "experts": experts[rows]})
        # fit_ranker quantizes inputs to fp16 before passing them to its network.
        values = torch.as_tensor(values, dtype=torch.float16, device=device)
        if device.type == "cpu":
            values = values.float()
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            risk = model(values).squeeze(-1).float()
        weights = (-risk / .35).softmax(dim=1)
        xyz, _ = literature.row_mean(torch.as_tensor(points, device=device), weights)
        result[rows] = xyz.cpu().numpy()
    return result


def _cells(method, saved, x, device):
    if saved["method"] != method:
        raise ValueError(f"Expected {method}, found {saved['method']} in checkpoint")
    centers = saved["centers"]
    if method == "fourier_location":
        model = literature.ContinuousLocator(centers, int(saved["seed"]))
    else:
        model = literature.GeographicClassifier(len(centers), hierarchy=method == "hierarchical_cells")
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    centers = centers.to(device)
    result = np.empty((len(x), 3), dtype=np.float32)
    for start in range(0, len(x), 4096):
        stop = min(start + 4096, len(x))
        # Advanced indexing gives _features its own storage for its in-place scaling on CPU.
        values = literature._features(x, np.arange(start, stop), device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            fine, coarse = model(values)
        probability = literature.probabilities(fine.float(), coarse.float() if coarse is not None else None)
        mode = centers[probability.argmax(dim=1)]
        result[start:stop] = literature._spherical_mean(probability, centers, mode).cpu().numpy()
    return result


@torch.inference_mode()
def predict(method, path, X, bundle=None):
    """Predict unit ECEF directions for raw [n,1280] paired vectors in their original order."""
    if method not in literature.METHODS:
        raise ValueError(f"Unknown literature method {method!r}")
    x = np.asarray(X)
    if x.ndim != 2 or x.shape[1] != 1280:
        raise ValueError("Expected paired query vectors with shape [rows, 1280]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if method == "candidate_ranker":
        answer = _ranker(saved, bundle, len(x), device)
    else:
        answer = _cells(method, saved, x, device)
    norm = np.linalg.norm(answer, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm < 1e-7).any():
        raise ValueError("Literature checkpoint produced invalid directions")
    return answer / norm
