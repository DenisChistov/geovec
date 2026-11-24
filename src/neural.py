"""Fold-local neural regression models for normalized image embeddings."""

from math import sqrt

import numpy as np
import torch
from pyproj import Geod
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F


NEURAL_CONFIGS = {
    "dnn": [
        {"name": "384-192", "widths": (384, 192), "dropout": 0.10, "epochs": 16,
         "batch_size": 2048, "lr": 0.0015, "weight_decay": 0.001},
        {"name": "768-384", "widths": (768, 384), "dropout": 0.15, "epochs": 22,
         "batch_size": 2048, "lr": 0.001, "weight_decay": 0.001},
    ],
    "siamese": [
        {"name": "384-192", "widths": (384, 192), "dropout": 0.05, "epochs": 16,
         "batch_size": 2048, "lr": 0.0012, "weight_decay": 0.001, "near_fraction": 0.25,
         "neighbors": 16},
        {"name": "768-384", "widths": (768, 384), "dropout": 0.10, "epochs": 22,
         "batch_size": 2048, "lr": 0.001, "weight_decay": 0.001, "near_fraction": 0.15,
         "neighbors": 32},
    ],
}

_MEAN_EARTH_RADIUS_M = 6371008.8
_WGS84_FLATTENING = 1.0 / 298.257223563
_WGS84_ONE_MINUS_E2 = (1.0 - _WGS84_FLATTENING) ** 2


class GeoEncoder(nn.Module):
    def __init__(self, features, widths, dropout):
        super().__init__()
        layers = []
        for width in widths:
            layers.extend((nn.Linear(features, width), nn.GELU(), nn.Dropout(dropout)))
            features = width
        self.network = nn.Sequential(*layers, nn.Linear(features, 3))

    def forward(self, features):
        return F.normalize(self.network(features).float(), dim=1, eps=1e-8)


def _geodetic_lonlat(ecef):
    x, y, z = np.asarray(ecef, dtype=np.float64).T
    return (
        np.degrees(np.arctan2(y, x)),
        np.degrees(np.arctan2(z, _WGS84_ONE_MINUS_E2 * np.hypot(x, y))),
    )


class GeographicPairs:
    def __init__(self, train_xyz, neighbors, near_fraction, seed):
        if len(train_xyz) < 2:
            raise ValueError("Siamese training requires at least two observations")
        self.n = len(train_xyz)
        self.rng = np.random.default_rng(seed)
        self.near_fraction = near_fraction
        self.lon, self.lat = _geodetic_lonlat(train_xyz)
        self.geod = Geod(ellps="WGS84")
        tree = cKDTree(train_xyz)
        _, indices = tree.query(train_xyz, k=min(neighbors + 1, self.n), workers=2)
        # With duplicate coordinates, the query result may omit the point itself.
        # Move any occurrence of self to the final slot before taking K neighbors.
        self.neighbors = np.sort(np.where(indices == np.arange(self.n)[:, None], self.n, indices), axis=1)[:, :-1]

    def sample(self):
        first = self.rng.permutation(self.n)
        second = self.rng.integers(self.n - 1, size=self.n)
        second += second >= first
        is_near = self.rng.random(self.n) < self.near_fraction
        near_cols = self.rng.integers(self.neighbors.shape[1], size=int(is_near.sum()))
        second[is_near] = self.neighbors[first[is_near], near_cols]
        _, _, meters = self.geod.inv(self.lon[first], self.lat[first], self.lon[second], self.lat[second])
        cosine = np.cos(meters / _MEAN_EARTH_RADIUS_M).astype(np.float32)
        return first, second, cosine


def _features(x, indices, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    subset = np.asarray(x[indices], dtype=np.float32)
    # Both source vectors have unit norm, so this yields unit RMS input scale.
    return torch.as_tensor(subset, dtype=dtype, device=device).mul_(sqrt(x.shape[1] / 2.0))


def _encode(model, features, batch_size):
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", enabled=features.is_cuda):
        return torch.cat([model(batch) for batch in features.split(batch_size)]).cpu().numpy()


def _orient_siamese(encoded, train_xyz, predictions, return_transform=False):
    cross = np.asarray(encoded, dtype=np.float64).T @ np.asarray(train_xyz, dtype=np.float64)
    u, _, vt = np.linalg.svd(cross)
    reflect = np.linalg.det(u @ vt) < 0
    if reflect:
        # The cosine loss cannot distinguish a mirror reflection from a rotation.
        # Canonicalize the learned handedness using only the training fold.
        cross[2] *= -1
        predictions[:, 2] *= -1
        u, _, vt = np.linalg.svd(cross)
    rotation = u @ vt
    projected = predictions @ rotation
    if return_transform:
        return projected, rotation, bool(reflect)
    return projected


def _normalize_predictions(predictions, train_xyz):
    norms = np.linalg.norm(predictions, axis=1)
    if not np.isfinite(norms).all():
        raise FloatingPointError("Neural model yielded nonfinite predictions")
    zero = norms <= 1e-8
    if zero.any():
        fallback = train_xyz.mean(axis=0)
        if np.linalg.norm(fallback) <= 1e-8:
            fallback = np.array([1.0, 0.0, 0.0])
        predictions[zero] = fallback
        norms[zero] = np.linalg.norm(fallback)
    return (predictions / norms[:, None]).astype(np.float32)


def train_predict(method, config, X, Y, train_idx, val_idx, seed):
    """Fit using train_idx alone and return unit ECEF vectors for val_idx."""
    if method not in NEURAL_CONFIGS:
        raise ValueError(f"Unknown neural method: {method}")
    train_idx = np.asarray(train_idx, dtype=np.intp)
    val_idx = np.asarray(val_idx, dtype=np.intp)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_gpu = device.type == "cuda"
    train_x = _features(X, train_idx, device)
    train_xyz = np.asarray(Y[train_idx], dtype=np.float32)
    model = GeoEncoder(X.shape[1], config["widths"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"], fused=on_gpu)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=0.1 * config["lr"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)
    batch_size = config["batch_size"]
    is_siamese = method == "siamese"
    if is_siamese:
        pairs = GeographicPairs(train_xyz, config["neighbors"], config["near_fraction"], seed)
    else:
        targets = torch.as_tensor(train_xyz, device=device)

    for _ in range(config["epochs"]):
        if is_siamese:
            left, right, similarities = (
                torch.as_tensor(array, device=device) for array in pairs.sample()
            )
        else:
            left = torch.randperm(len(train_idx), device=device)

        for start in range(0, len(train_idx), batch_size):
            batch = slice(start, start + batch_size)
            anchor = left[batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=on_gpu):
                if is_siamese:
                    joined = torch.cat((anchor, right[batch]))
                    first, second = model(train_x[joined]).chunk(2)
                    loss = F.mse_loss((first * second).sum(dim=1), similarities[batch])
                else:
                    loss = F.mse_loss(model(train_x[anchor]), targets[anchor])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    if not len(val_idx):
        return np.empty((0, 3), dtype=np.float32)
    val_features = _features(X, val_idx, device)
    predictions = _encode(model, val_features, batch_size)
    rotation, reflect = None, False
    if is_siamese:
        train_encoded = _encode(model, train_x, batch_size)
        predictions, rotation, reflect = _orient_siamese(train_encoded, train_xyz, predictions, return_transform=True)
    if config.get("save_path"):
        torch.save({
            "model_state": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
            "architecture": {"features": X.shape[1], "widths": config["widths"], "dropout": config["dropout"]},
            "rotation": torch.as_tensor(rotation) if rotation is not None else None,
            "reflect_z_first": reflect,
            "ecef": "WGS84",
        }, config["save_path"])
    return _normalize_predictions(predictions, train_xyz)
