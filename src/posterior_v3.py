"""Fold-local cell posterior model with calibrated-temperature candidates."""

from math import sqrt

import numpy as np
import torch
from torch.nn import functional as F

import cell_v2 as base


TEMPERATURES = {"pred": 1.0, "t08": 0.8, "t12": 1.2, "t06": 0.6, "t16": 1.6}
TOP_CELLS = 32
FEATURE_CELLS = 10
FEATURE_NAMES = tuple(
    f"rank{rank + 1}_{name}" for rank in range(FEATURE_CELLS) for name in ("p", "x", "y", "z")
) + (
    "entropy", "normalized_entropy", "confidence_top1", "confidence_margin",
    "probability_top3", "probability_top8", "probability_top32", "resultant_length",
    "mean_angle_rad", "rms_angle_rad", "mode_mean_angle_rad",
    "posterior_weighted_prior", "mode_prior",
)


def _angular_distance(a, b):
    cosine = (a * b).sum(dim=-1)
    sine = torch.linalg.vector_norm(torch.linalg.cross(a, b, dim=-1), dim=-1)
    return torch.atan2(sine, cosine)


def _cell_radii(points, centers, labels, counts):
    distance = _angular_distance(points, centers[labels])
    squared = torch.zeros_like(counts, dtype=torch.float32).scatter_add_(0, labels, distance.square())
    return (squared / counts.clamp_min(1)).sqrt()


def _empty_predictions(n):
    result = {name: np.empty((n, 3), dtype=np.float32) for name in (*TEMPERATURES, "mode")}
    result.update(
        top_centers=np.empty((n, TOP_CELLS, 3), dtype=np.float32),
        top_probs=np.empty((n, TOP_CELLS), dtype=np.float32),
        top_radii=np.empty((n, TOP_CELLS), dtype=np.float32),
        top_counts=np.empty((n, TOP_CELLS), dtype=np.int32),
        features=np.empty((n, len(FEATURE_NAMES)), dtype=np.float32),
    )
    return result


def _ranked_cells(probs, centers, counts, radii, mode):
    ranked_probs, top_ids = probs.topk(min(TOP_CELLS, len(centers)), dim=1)
    top_centers, top_radii, top_counts = centers[top_ids], radii[top_ids], counts[top_ids]
    missing = TOP_CELLS - ranked_probs.shape[1]
    if missing:
        ranked_probs = F.pad(ranked_probs, (0, missing))
        top_radii = F.pad(top_radii, (0, missing))
        top_counts = F.pad(top_counts, (0, missing))
        top_centers = torch.cat((top_centers, mode[:, None, :].expand(-1, missing, -1)), dim=1)
    return ranked_probs, top_centers, top_radii, top_counts


def _posterior_features(probs, centers, priors, mode, mean, ranked_probs, top_centers):
    # Raw class identifiers differ across fold-local clusterings, whereas
    # ranks, probabilities and absolute center coordinates remain meaningful.
    p = ranked_probs[:, :FEATURE_CELLS]
    xyz = top_centers[:, :FEATURE_CELLS]
    xyz = torch.where(p[:, :, None] > 0, xyz, 0.0)
    rank_features = torch.cat((p[:, :, None], xyz), dim=2).flatten(1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    n_active = (priors > 0).sum().float().clamp_min(2)
    average = probs @ centers
    angles = (mean @ centers.T).clamp(-1, 1).acos()
    confidence = torch.stack((
        entropy,
        entropy / n_active.log(),
        ranked_probs[:, 0],
        ranked_probs[:, 0] - ranked_probs[:, 1],
        ranked_probs[:, :3].sum(dim=1),
        ranked_probs[:, :8].sum(dim=1),
        ranked_probs.sum(dim=1),
        torch.linalg.vector_norm(average, dim=1),
        (probs * angles).sum(dim=1),
        (probs * angles.square()).sum(dim=1).sqrt(),
        _angular_distance(mode, mean),
        probs @ priors,
        priors[probs.argmax(dim=1)],
    ), dim=1)
    return torch.cat((rank_features, confidence), dim=1)


def _infer(model, features, centers, counts, radii):
    results = _empty_predictions(len(features))
    priors = counts.float() / counts.sum()
    model.eval()
    with torch.inference_mode():
        # Keep all temperature reductions and geographic dot products in float32.
        for start in range(0, len(features), 2048):
            batch = features[start:start + 2048]
            with torch.autocast("cuda", enabled=features.is_cuda):
                logits = model(batch).float()
            logits = logits.masked_fill(counts == 0, -torch.inf)
            mode = centers[logits.argmax(dim=1)]
            probs = logits.softmax(dim=1)
            mean = base._spherical_mean(probs, centers, mode)
            rows = slice(start, start + len(batch))
            results["mode"][rows] = mode.cpu().numpy()
            results["pred"][rows] = mean.cpu().numpy()
            for name, temperature in TEMPERATURES.items():
                if name == "pred":
                    continue
                adjusted = (logits / temperature).softmax(dim=1)
                results[name][rows] = base._spherical_mean(adjusted, centers, mode).cpu().numpy()
            top_probs, top_centers, top_radii, top_counts = _ranked_cells(probs, centers, counts, radii, mode)
            results["top_probs"][rows] = top_probs.cpu().numpy()
            results["top_centers"][rows] = top_centers.cpu().numpy()
            results["top_radii"][rows] = top_radii.cpu().numpy()
            results["top_counts"][rows] = top_counts.cpu().numpy()
            results["features"][rows] = _posterior_features(
                probs, centers, priors, mode, mean, top_probs, top_centers,
            ).cpu().numpy()
    return results


def fit_posterior(X, Y, train_idx, query_idx, k, seed, save_path=None):
    """Fit a fold-local posterior; use only training labels to fit and cluster."""
    train_idx = np.asarray(train_idx, dtype=np.intp)
    query_idx = np.asarray(query_idx, dtype=np.intp)
    k, seed = int(k), int(seed)
    if not 2 <= k <= len(train_idx):
        raise ValueError("Expected 2 <= k <= training row count")
    if X.ndim != 2 or X.shape[1] != 1280:
        raise ValueError("Expected 512-dimensional GeoCLIP and 768-dimensional StreetCLIP")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(seed)
    points = torch.as_tensor(Y[train_idx], dtype=torch.float32, device=device)
    points = F.normalize(points, dim=1)
    centers, labels, counts = base._spherical_cells(points, k, rng)
    radii = _cell_radii(points, centers, labels, counts)
    model = base._fit_classifier(base._features(X, train_idx, device), labels, counts, rng)
    if save_path is not None:
        torch.save({
            "k": k, "seed": seed, "hidden": base._HIDDEN, "epochs": base._EPOCHS,
            "batch_size": base._BATCH_SIZE, "input_scale": sqrt(X.shape[1] / 2),
            "centers": centers.cpu(), "cluster_counts": counts.cpu(), "cell_radii_rad": radii.cpu(),
            "prior_probs": (counts.float() / counts.sum()).cpu(), "temperature_candidates": TEMPERATURES,
            "feature_names": FEATURE_NAMES,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        }, save_path)
    if not len(query_idx):
        return _empty_predictions(0)
    return _infer(model, base._features(X, query_idx, device), centers, counts, radii)
