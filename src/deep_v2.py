"""Direct regression from GeoCLIP/StreetCLIP to WGS84 ECEF directions."""

from math import pi, sqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CONFIGS = {
    "residual-angle": {
        "width": 640, "bottleneck": 320, "blocks": 2, "dropout": 0.10,
        "epochs": 32, "batch_size": 4096, "lr": 0.0020, "weight_decay": 0.002,
        "angle_weight": 0.90, "dual_stem": False, "aux_weight": 0.0,
    },
    "dual-stem-regions": {
        "width": 640, "bottleneck": 320, "blocks": 2, "dropout": 0.12,
        "epochs": 35, "batch_size": 4096, "lr": 0.0023, "weight_decay": 0.002,
        "angle_weight": 0.80, "dual_stem": True, "aux_weight": 0.12,
    },
}

_GEO_WIDTH = 512
_STREET_WIDTH = 768
_LAT_BINS = 12
_LON_BINS = 24
_WGS84_FLATTENING = 1 / 298.257223563
_AXIS_RATIO_TERM = 1 / (1 - _WGS84_FLATTENING)**2 - 1


class ResidualBlock(nn.Module):
    def __init__(self, width, bottleneck, dropout):
        super().__init__()
        self.transform = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, bottleneck), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, width),
        )

    def forward(self, x):
        return x + self.transform(x)


class GeoRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["width"]
        self.dual_stem = config["dual_stem"]
        if self.dual_stem:
            geo_hidden = width * 2 // 5
            self.geo = nn.Sequential(nn.Linear(_GEO_WIDTH, geo_hidden), nn.GELU())
            self.street = nn.Sequential(nn.Linear(_STREET_WIDTH, width - geo_hidden), nn.GELU())
        else:
            self.stem = nn.Sequential(nn.Linear(_GEO_WIDTH + _STREET_WIDTH, width), nn.GELU())
        self.body = nn.Sequential(*(
            ResidualBlock(width, config["bottleneck"], config["dropout"]) for _ in range(config["blocks"])
        ), nn.LayerNorm(width), nn.Dropout(config["dropout"]))
        self.position = nn.Linear(width, 3)
        self.region = nn.Linear(width, _LAT_BINS * _LON_BINS) if config["aux_weight"] else None

    def forward(self, x):
        if self.dual_stem:
            geo, street = x.split((_GEO_WIDTH, _STREET_WIDTH), dim=-1)
            features = torch.cat((self.geo(geo), self.street(street)), dim=-1)
        else:
            features = self.stem(x)
        hidden = self.body(features)
        xyz = F.normalize(self.position(hidden).float(), dim=-1, eps=1e-8)
        regional_logits = self.region(hidden) if self.training and self.region is not None else None
        return xyz, regional_logits


def _geographic_loss(predicted, target, angle_weight):
    dot = (predicted * target).sum(dim=-1)
    cross = torch.linalg.cross(predicted, target, dim=-1)
    angle = torch.atan2(torch.linalg.vector_norm(cross, dim=-1), dot)
    # Squared central angle with an effective WGS84 radius correction. The
    # evaluator uses the exact ellipsoidal geodesic independently of this loss.
    mean_z_squared = 0.5 * (predicted[:, 2].square() + target[:, 2].square())
    angular = angle.square() / (1 + _AXIS_RATIO_TERM * mean_z_squared)
    chord = (predicted - target).square().sum(dim=-1)
    return (angle_weight * angular + (1 - angle_weight) * chord).mean()


def _region_index(xyz):
    lat_band = ((xyz[:, 2] + 1) * (_LAT_BINS / 2)).long().clamp(0, _LAT_BINS - 1)
    lon = torch.atan2(xyz[:, 1], xyz[:, 0])
    lon_band = ((lon + pi) * (_LON_BINS / (2 * pi))).long().remainder(_LON_BINS)
    return lat_band * _LON_BINS + lon_band


def _features(x, indices, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    values = torch.as_tensor(np.asarray(x[indices], dtype=np.float32), device=device, dtype=dtype)
    return values.mul_(sqrt(x.shape[1] / 2))


def fit_predict(X, Y, train_idx, query_idx, config, seed, save_path=None):
    """Fit on train_idx and return normalized float32 predictions for query_idx."""
    if X.ndim != 2 or X.shape[1] != _GEO_WIDTH + _STREET_WIDTH:
        raise ValueError("Expected contiguous feature blocks GeoCLIP (512) and StreetCLIP (768)")
    train_idx = np.asarray(train_idx, dtype=np.intp)
    query_idx = np.asarray(query_idx, dtype=np.intp)
    if not len(train_idx):
        raise ValueError("No training rows")
    torch.manual_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_gpu = device.type == "cuda"
    features = _features(X, train_idx, device)
    target = torch.as_tensor(Y[train_idx], dtype=torch.float32, device=device)
    regions = _region_index(target) if config["aux_weight"] else None
    model = GeoRegressor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], fused=on_gpu)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)
    for epoch in range(config["epochs"]):
        angle_weight = config["angle_weight"] * min(1.0, (epoch + 1) / 3)
        shuffled = torch.randperm(len(train_idx), device=device)
        for batch in shuffled.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=on_gpu):
                predicted, logits = model(features[batch])
                loss = _geographic_loss(predicted, target[batch], angle_weight)
                if logits is not None:
                    loss = loss + config["aux_weight"] * F.cross_entropy(logits, regions[batch])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        schedule.step()

    if save_path is not None:
        state = {name: param.detach().cpu() for name, param in model.state_dict().items()}
        torch.save({"config": dict(config), "state_dict": state, "input_scale": sqrt(X.shape[1] / 2)}, save_path)
    if not len(query_idx):
        return np.empty((0, 3), dtype=np.float32)

    mean = target.mean(dim=0).cpu().numpy()
    del features, target, optimizer, schedule, scaler
    query = _features(X, query_idx, device)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", enabled=on_gpu):
        xyz = torch.cat([model(batch)[0] for batch in query.split(config["batch_size"])]).cpu().numpy()
    norms = np.linalg.norm(xyz, axis=1)
    if not np.isfinite(norms).all():
        raise FloatingPointError("Nonfinite geographic predictions")
    zero = norms <= 1e-8
    if zero.any():
        if np.linalg.norm(mean) <= 1e-8:
            mean = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        xyz[zero] = mean
        norms[zero] = np.linalg.norm(mean)
    return xyz / norms[:, None]
