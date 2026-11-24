"""Geographic prior and cell-posterior models for unit WGS84 ECEF directions."""

from math import pi, sqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


K_VALUES = (512, 1024)
_HIDDEN = (640, 384)
_BATCH_SIZE = 4096
_EPOCHS = 17
_CLUSTER_ITERATIONS = 13


class CellClassifier(nn.Module):
    def __init__(self, feature_count, k):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, _HIDDEN[0]), nn.LayerNorm(_HIDDEN[0]), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(_HIDDEN[0], _HIDDEN[1]), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(_HIDDEN[1], k),
        )

    def forward(self, features):
        return self.network(features)


def _initial_centers(points, count, rng):
    """Seed by k-means++ on a shuffled subsample, using squared chord distance."""
    sample_ids = torch.randperm(len(points), device=points.device, generator=rng)[:16384]
    sample = points[sample_ids]
    uniforms = torch.rand(count, device=points.device, generator=rng)
    sample_index = (uniforms[:1] * len(sample)).long()
    nearest = torch.full((len(sample),), float("inf"), device=points.device)
    centers = torch.empty((count, 3), device=points.device)
    for i in range(count):
        center = sample.index_select(0, sample_index)
        centers[i] = center[0]
        if i + 1 < count:
            chord_squared = (sample - center).square().sum(dim=1)
            nearest = torch.minimum(nearest, chord_squared)
            cumulative = nearest.clamp_min(1e-12).cumsum(dim=0)
            sample_index = torch.searchsorted(cumulative, uniforms[i + 1:i + 2] * cumulative[-1], right=True)
            sample_index.clamp_max_(len(sample) - 1)
    return centers


def _assign(points, centers):
    # Fractional differences near cosine 1 matter for adjacent geographic cells.
    # Keep this dot product and its reduction in float32.
    return torch.cat([(batch @ centers.T).argmax(dim=1) for batch in points.split(8192)])


def _counts(labels, k):
    counts = torch.zeros(k, dtype=torch.long, device=labels.device)
    return counts.scatter_add_(0, labels, torch.ones_like(labels))


def _spherical_cells(points, k, rng):
    centers = _initial_centers(points, k, rng)
    for _ in range(_CLUSTER_ITERATIONS):
        labels = _assign(points, centers)
        sums = torch.zeros_like(centers).index_add_(0, labels, points)
        norms = torch.linalg.vector_norm(sums, dim=1, keepdim=True)
        updated = sums / norms.clamp_min(1e-8)
        replacements = points[torch.randint(len(points), (k,), device=points.device, generator=rng)]
        centers = torch.where(norms > 1e-8, updated, replacements)
    labels = _assign(points, centers)
    return centers, labels, _counts(labels, k)


def _features(x, indices, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    values = torch.as_tensor(np.asarray(x[indices], dtype=np.float32), dtype=dtype, device=device)
    return values.mul_(sqrt(x.shape[1] / 2))


def _fit_classifier(features, labels, counts, rng):
    gpu = features.is_cuda
    model = CellClassifier(features.shape[1], len(counts)).to(features.device)
    with torch.no_grad():
        log_frequency = (counts.float() + 1).log()
        model.network[-1].bias.copy_(log_frequency - log_frequency.mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002, fused=gpu)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_EPOCHS, eta_min=0.00012)
    scaler = torch.amp.GradScaler("cuda", enabled=gpu)
    for _ in range(_EPOCHS):
        order = torch.randperm(len(features), device=features.device, generator=rng)
        for batch in order.split(_BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=gpu):
                logits = model(features[batch])
                loss = F.cross_entropy(logits, labels[batch], label_smoothing=0.012)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        schedule.step()
    return model


def _spherical_mean(probabilities, centers, mode):
    average = probabilities @ centers
    norm = torch.linalg.vector_norm(average, dim=1, keepdim=True)
    axes = F.one_hot(mode.abs().argmin(dim=1), num_classes=3).to(mode.dtype)
    perpendicular = F.normalize(torch.linalg.cross(mode, axes, dim=1), dim=1)
    mean = torch.where(norm > 1e-8, average / norm.clamp_min(1e-8), perpendicular)
    # Three intrinsic Fréchet updates. A matrix multiply and tangent projection
    # avoid allocating a [batch, cells, 3] tensor for the log map.
    for _ in range(3):
        dot = (mean @ centers.T).clamp(-1, 1)
        sin = (1 - dot.square()).clamp_min(1e-8).sqrt()
        scale = torch.atan2(sin, dot) / sin
        tangent = (probabilities * scale) @ centers
        tangent = tangent - (tangent * mean).sum(dim=1, keepdim=True) * mean
        step = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
        tangent = tangent * (1.0 / step.clamp_min(1.0))
        step = step.clamp_max(1.0)
        mean = F.normalize(step.cos() * mean + torch.sinc(step / pi) * tangent, dim=1)
    return mean


def _predict(model, features, centers, counts):
    modes, means = [], []
    model.eval()
    with torch.inference_mode():
        for batch in features.split(_BATCH_SIZE):
            with torch.autocast("cuda", enabled=features.is_cuda):
                logits = model(batch).float()
            logits = logits.masked_fill(counts == 0, -torch.inf)
            mode = centers[logits.argmax(dim=1)]
            probabilities = logits.softmax(dim=1)
            mean = _spherical_mean(probabilities, centers, mode)
            modes.append(mode.cpu().numpy())
            means.append(mean.cpu().numpy())
    return np.concatenate(modes), np.concatenate(means)


def fit_predict(X, Y, train_idx, query_idx, k, seed, save_path=None):
    """Fit cells, empirical prior, and conditional posterior using train_idx only."""
    train_idx = np.asarray(train_idx, dtype=np.intp)
    query_idx = np.asarray(query_idx, dtype=np.intp)
    k = int(k)
    if not 2 <= k <= len(train_idx):
        raise ValueError("Expected 2 <= k <= the number of training rows")
    if X.ndim != 2 or X.shape[1] != 1280:
        raise ValueError("Expected 512 GeoCLIP and 768 StreetCLIP input features")
    seed = int(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device).manual_seed(seed)
    points = torch.as_tensor(Y[train_idx], dtype=torch.float32, device=device)
    points = F.normalize(points, dim=1)
    centers, labels, counts = _spherical_cells(points, k, rng)
    model = _fit_classifier(_features(X, train_idx, device), labels, counts, rng)
    prior_probs = counts.float() / counts.sum()
    prior_mode = centers.index_select(0, counts.argmax().reshape(1))
    prior_mean = _spherical_mean(prior_probs[None, :], centers, prior_mode)
    if save_path is not None:
        torch.save({
            "k": k, "seed": seed, "hidden": _HIDDEN, "epochs": _EPOCHS, "batch_size": _BATCH_SIZE,
            "input_scale": sqrt(X.shape[1] / 2), "centers": centers.cpu(), "cluster_counts": counts.cpu(),
            "prior_probs": prior_probs.cpu(), "prior_mode": prior_mode[0].cpu(), "prior_mean": prior_mean[0].cpu(),
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        }, save_path)
    if not len(query_idx):
        return {name: np.empty((0, 3), dtype=np.float32) for name in ("prior", "prior_mean", "posterior_mode", "posterior_mean")}
    mode, mean = _predict(model, _features(X, query_idx, device), centers, counts)
    return {"prior": np.repeat(prior_mode.cpu().numpy(), len(query_idx), axis=0),
            "prior_mean": np.repeat(prior_mean.cpu().numpy(), len(query_idx), axis=0),
            "posterior_mode": mode, "posterior_mean": mean}
