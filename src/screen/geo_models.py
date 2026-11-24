"""Ten conditional geographic models for frozen image features and unit WGS84 ECEF targets."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geo_geometry import prepare


METHODS = {
    "geo_balanced_cells": "Классификатор ячеек равной численности, полученных медианными разрезами вдоль главных осей географии.",
    "geo_distance_targets": "Мягкие целевые вероятности ближайших сферических кластеров по эллипсоидальному расстоянию WGS84.",
    "geo_conditional_tree": "Отдельные головы для крупного региона и условного распределения по его дочерним ячейкам.",
    "geo_fourier_similarity": "Два кодировщика сопоставляют изображение с координатами опор через многочастотное представление Фурье.",
    "geo_ellipsoid_risk": "Кросс-энтропия классов с дополнительным ожидаемым квадратом расстояния между опорами WGS84.",
    "geo_landmark_rings": "Модель предсказывает попадание в кольцевые пороги вокруг опор, затем согласует их с картой ячеек.",
    "geo_local_offsets": "Классификатор с отдельными предсказаниями касательных смещений внутри каждой ячейки.",
    "geo_graph_spectrum": "Регрессия собственных функций географического графа с вероятностным декодированием по спектральным кодам.",
    "geo_directional_mixture": "Условная смесь распределений фон Мизеса — Фишера с обучаемыми направлениями, концентрациями и массами.",
    "geo_triangle_targets": "Классификация географических опор по барицентрическим меткам вершин сферического треугольника.",
}

EPOCHS = 6
BATCH = 2048
BALANCED = {"geo_balanced_cells", "geo_conditional_tree"}


def _geometry(method, prepared, device, seed):
    def tensor(key):
        return torch.as_tensor(prepared[key], device=device, dtype=torch.float32)

    centers = tensor("balanced_centers" if method in BALANCED else "centers")
    if method in BALANCED:
        labels = torch.as_tensor(prepared["balanced_labels"], device=device)
        prior = torch.bincount(labels.long(), minlength=len(centers)).float()
        prior /= prior.sum()
    else:
        prior = tensor("prior")
    geo = {"centers": centers, "log_prior": prior.clamp_min(1e-9).log()}
    if method == "geo_fourier_similarity":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        freq = torch.randn(3, 48, generator=gen) * torch.tensor([1., 4., 16., 48.]).repeat_interleave(12)
        phase = (2 * torch.pi) * (centers @ freq.to(device))
        geo["bank"] = torch.cat((centers, phase.sin(), phase.cos()), dim=1)
    if method == "geo_graph_spectrum":
        geo["bank"] = tensor("spectral_codes").clamp(-7, 7)
    if method == "geo_landmark_rings":
        geo["bank"] = tensor("ring_codes")
    if method == "geo_local_offsets":
        geo["radius"] = tensor("radius").clamp(.01, 1.25)
    if method == "geo_directional_mixture":
        geo["anchors"] = tensor("mixture_centers")
    return geo


class GeoHead(nn.Module):
    def __init__(self, method, width, geo):
        super().__init__()
        self.method = method
        self.backbone = nn.Sequential(
            nn.Linear(width, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.06),
            nn.Linear(192, 128), nn.GELU(),
        )
        k = len(geo["centers"])
        output = k
        if method == "geo_conditional_tree":
            output += k // min(8, k)
        if method == "geo_local_offsets":
            output = 4 * k
        if method == "geo_directional_mixture":
            output = 5 * len(geo["anchors"])
        if method in {"geo_graph_spectrum", "geo_landmark_rings"}:
            output = geo["bank"].shape[1]
        if method == "geo_fourier_similarity":
            output = 64
            self.location = nn.Sequential(nn.Linear(geo["bank"].shape[1], 96), nn.GELU(), nn.Linear(96, output))
            self.log_scale = nn.Parameter(torch.tensor(2.5))
        self.head = nn.Linear(128, output)
        if method not in {"geo_directional_mixture", "geo_fourier_similarity", "geo_landmark_rings", "geo_graph_spectrum"}:
            with torch.no_grad():
                self.head.bias[:k].copy_(geo["log_prior"] - geo["log_prior"].mean())

    def forward(self, x, geo):
        out = self.head(self.backbone(x))
        if self.method == "geo_fourier_similarity":
            v = F.normalize(out.float(), dim=1)
            t = F.normalize(self.location(geo["bank"]).float(), dim=1)
            return (v @ t.T).float() * self.log_scale.clamp(1, 4.2).exp()
        return out.float()


def _logits(method, out, geo):
    k = len(geo["centers"])
    if method == "geo_conditional_tree":
        children = min(8, k)
        weights = out[:, :k].reshape(len(out), k // children, children).log_softmax(dim=2)
        return (weights + out[:, k:].log_softmax(dim=1)[:, :, None]).flatten(1)
    if method == "geo_landmark_rings":
        codes = geo["bank"]
        # For a Bernoulli code the log likelihood differs across candidate
        # locations by the dot product of its bits with the predicted logits.
        return (out @ codes.T) / max(1., codes.shape[1] ** .5) + geo["log_prior"]
    if method == "geo_graph_spectrum":
        codes = geo["bank"]
        return (2 * out @ codes.T - codes.square().sum(dim=1)) / max(1., codes.shape[1] ** .5) + geo["log_prior"]
    return out[:, :k]


def _mixture(out, geo):
    parts = out.reshape(len(out), len(geo["anchors"]), 5)
    directions = F.normalize(geo["anchors"][None] + parts[:, :, 1:4], dim=2)
    kappa = (1 + 12 * F.softplus(parts[:, :, 4])).clamp(max=200)
    return parts[:, :, 0], directions, kappa


def _local_delta(raw, centers, radii):
    delta = 2 * torch.tanh(raw) * radii[..., None]
    return delta - (delta * centers).sum(dim=-1, keepdim=True) * centers


def _targets(method, prepared, device):
    label = "balanced_labels" if method in BALANCED else "labels"
    keys = {
        "geo_distance_targets": ("distance_ids", "distance_weights"),
        "geo_triangle_targets": ("tri_ids", "tri_weights"),
        "geo_ellipsoid_risk": ("risk_table",),
        "geo_landmark_rings": ("ring_targets",),
        "geo_local_offsets": ("y",),
        "geo_directional_mixture": ("y",),
    }
    names = (label,) + keys.get(method, ())
    arrays = {k: torch.as_tensor(prepared[k], device=device) for k in names}
    arrays["labels"] = arrays.pop(label).long()
    return arrays


def _loss(method, out, geo, data, idx):
    labels = data["labels"][idx]
    if method == "geo_directional_mixture":
        mass, direction, kappa = _mixture(out, geo)
        true = data["y"][idx]
        alignment = (true[:, None] * direction).sum(dim=2).clamp(-1, 1)
        log_density = kappa.log() - torch.log(-torch.expm1(-2 * kappa)) + kappa * (alignment - 1)
        return -(mass.log_softmax(dim=1) + log_density).logsumexp(dim=1).mean()
    logits = _logits(method, out, geo)
    if method in {"geo_distance_targets", "geo_triangle_targets"}:
        ids, weight = ("distance_ids", "distance_weights") if method == "geo_distance_targets" else ("tri_ids", "tri_weights")
        ll = logits.log_softmax(dim=1).gather(1, data[ids][idx].long())
        return -(ll * data[weight][idx]).sum(dim=1).mean()
    loss = F.cross_entropy(logits, labels)
    if method == "geo_ellipsoid_risk":
        loss += .22 * (logits.softmax(dim=1) * data["risk_table"][labels]).sum(dim=1).mean()
    if method == "geo_landmark_rings":
        loss += 1.5 * F.binary_cross_entropy_with_logits(out, data["ring_targets"][idx].float())
    if method == "geo_graph_spectrum":
        loss += .6 * F.smooth_l1_loss(out, geo["bank"][labels])
    if method == "geo_local_offsets":
        k = len(geo["centers"])
        raw = out[:, k:].reshape(len(out), k, 3)[torch.arange(len(out), device=out.device), labels]
        center = geo["centers"][labels]
        radius = geo["radius"][labels]
        y = data["y"][idx]
        dot = (center * y).sum(dim=1, keepdim=True).clamp(-1, 1)
        tangent = y - dot * center
        norm = tangent.norm(dim=1, keepdim=True)
        target = tangent * torch.atan2(norm, dot) / norm.clamp_min(1e-7)
        estimated = _local_delta(raw, center, radius)
        loss += .8 * F.smooth_l1_loss(estimated / radius[:, None], target / radius[:, None])
    return loss


def _mean(weights, centers):
    shared = centers.ndim == 2
    def average(w):
        if shared:
            return w @ centers
        return torch.bmm(w[:, None], centers)[:, 0]

    direction = average(weights)
    length = direction.norm(dim=1, keepdim=True)
    mode = weights.argmax(dim=1)
    fallback = centers[mode] if shared else centers[torch.arange(len(mode), device=mode.device), mode]
    mean = torch.where(length > 1e-7, direction / length.clamp_min(1e-7), fallback)
    for _ in range(2):
        dot = (mean @ centers.T if shared else (mean[:, None] * centers).sum(dim=2)).clamp(-1, 1)
        sine = (1 - dot.square()).clamp_min(1e-8).sqrt()
        tangent = average(weights * torch.atan2(sine, dot) / sine)
        tangent -= (tangent * mean).sum(dim=1, keepdim=True) * mean
        step = tangent.norm(dim=1, keepdim=True)
        tangent /= step.clamp_min(1.)
        step = step.clamp_max(1.)
        mean = F.normalize(step.cos() * mean + torch.sinc(step / torch.pi) * tangent, dim=1)
    return mean


def _decode(method, out, geo):
    if method == "geo_directional_mixture":
        mass, directions, kappa = _mixture(out, geo)
        # Use the mean resultant length of each vMF component to attenuate
        # the directional contribution of near-uniform components.
        weights = mass.softmax(dim=1) * (1 / kappa.tanh() - 1 / kappa)
        return _mean(weights / weights.sum(dim=1, keepdim=True), directions)
    weights = _logits(method, out, geo).softmax(dim=1)
    centers = geo["centers"]
    if method == "geo_local_offsets":
        k = len(centers)
        raw = out[:, k:].reshape(len(out), k, 3)
        delta = _local_delta(raw, centers[None], geo["radius"][None])
        angle = delta.norm(dim=2, keepdim=True)
        centers = F.normalize(angle.cos() * centers[None] + torch.sinc(angle / torch.pi) * delta, dim=2)
    return _mean(weights, centers)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _features(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)


@torch.inference_mode()
def _infer(model, x, scale, geo, device):
    output = np.empty((len(x), 3), dtype=np.float32)
    model.eval()
    for start in range(0, len(x), BATCH):
        rows = slice(start, start + BATCH)
        batch = _features(x[rows], device) * scale
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            raw = model(batch, geo)
        output[rows] = _decode(model.method, raw, geo).cpu().numpy()
    return output


def predict(method, X_train, Y_train, X_query, prepared, seed, save_dir=None):
    """Fit using the supplied fold-local geometry and return unit ECEF directions."""
    if method not in METHODS:
        raise ValueError(f"Unknown geographic model {method!r}")
    if (len(Y_train) != len(X_train) or len(prepared["y"]) != len(Y_train)
            or X_train.ndim != 2 or X_query.ndim != 2 or X_query.shape[1] != X_train.shape[1]):
        raise ValueError("Training and query arrays must match the prepared fold and feature width")
    device = _device()
    torch.manual_seed(int(seed))
    geo = _geometry(method, prepared, device, int(seed))
    data = _targets(method, prepared, device)
    x = _features(X_train, device)
    scale = float(x[:4096].float().square().mean().clamp_min(1e-8).rsqrt().clamp(max=512))
    x = x * scale
    width = x.shape[1]
    model = GeoHead(method, width, geo).to(device)
    gpu = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025, weight_decay=.002, fused=gpu)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=.00035)
    scaler = torch.amp.GradScaler("cuda", enabled=gpu, init_scale=1024.)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    for _ in range(EPOCHS):
        for ids in torch.randperm(len(x), device=device, generator=gen).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=gpu):
                raw = model(x[ids], geo)
            loss = _loss(method, raw, geo, data, ids)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        schedule.step()
    if save_dir is not None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        saved_geometry = {k: v.detach().cpu() for k, v in geo.items()}
        checkpoint = {"method": method, "width": width, "scale": scale, "geo": saved_geometry, "state": state}
        torch.save(checkpoint, path / "geo_model.tmp.pt")
        (path / "geo_model.tmp.pt").replace(path / "geo_model.pt")
    return _infer(model, X_query, scale, geo, device)


def predict_saved(X_query, save_dir):
    """Replay a single fitted configuration on identically transformed query features."""
    device = _device()
    saved = torch.load(Path(save_dir) / "geo_model.pt", map_location=device, weights_only=True)
    if X_query.ndim != 2 or X_query.shape[1] != saved["width"]:
        raise ValueError("The checkpoint expects the training feature width")
    model = GeoHead(saved["method"], saved["width"], saved["geo"]).to(device)
    model.load_state_dict(saved["state"])
    return _infer(model, X_query, saved["scale"], saved["geo"], device)
