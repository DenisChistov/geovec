"""Ten geographic retrieval decoders sharing one top-64 search per query.

The caller supplies one pre-fitted feature representation at a time. Features
are re-normalized here. All calibration, frequencies and reference coordinates
come exclusively from the supplied training rows.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


METHODS = {
    "ret_frechet": "Взвешенное внутреннее среднее на сфере с температурой по разбросу визуальных баллов.",
    "ret_geo_prior": "Поправка визуальных весов на фоновую плотность обучающих мест в равновеликих ячейках.",
    "ret_support_medoid": "Географический медоид из реальных соседей, минимизирующий взвешенную сумму квадратов углов.",
    "ret_huber": "Устойчивая сферическая оценка положения с весами функции Хьюбера.",
    "ret_two_basin": "Две мягкие географические группы с отдельными центрами и уточнёнными массами.",
    "ret_local_ridge": "Локальная регуляризованная регрессия касательного смещения на визуальные разности.",
    "ret_kernel_mode": "Наиболее плотный географический бассейн и локальное среднее внутри него.",
    "ret_hubness": "Коррекция визуального балла по близости референса к независимым пробам обучающей базы.",
    "ret_learned_kernel": "Обученное на поиске с исключением самого примера ядро визуального и географического согласия.",
    "ret_barycentric": "Энтропийно регуляризованные барицентрические веса для восстановления визуального запроса соседями.",
}
K = 64
BLOCK = 384


def _unit(value, fallback):
    length = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return torch.where(length > 1e-7, value / length.clamp_min(1e-7), fallback)


def _log(origin, value):
    dot = (origin * value).sum(dim=-1, keepdim=True).clamp(-1, 1)
    tangent = value - dot * origin
    length = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    axis = torch.zeros_like(origin).scatter_(-1, origin.abs().argmin(dim=-1, keepdim=True), 1)
    alt = F.normalize(torch.linalg.cross(origin, axis, dim=-1), dim=-1)
    direction = torch.where((length < 1e-7) & (dot < 0), alt, tangent / length.clamp_min(1e-7))
    return direction * torch.atan2(length, dot)


def _exp(origin, step):
    step = step - (step * origin).sum(-1, keepdim=True) * origin
    length = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
    return _unit(origin * length.cos() + step * torch.sinc(length / torch.pi), origin)


def _limit(step, cap):
    length = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
    return step * (cap / length.clamp_min(1e-7)).clamp(max=1)


def _mean(points, weight, initial=None, steps=2):
    row = torch.arange(len(points), device=points.device)
    if initial is None:
        fallback = points[row, weight.argmax(dim=1)]
        initial = _unit((weight[:, :, None] * points).sum(1), fallback)
    for _ in range(steps):
        step = (weight[:, :, None] * _log(initial[:, None], points)).sum(1)
        initial = _exp(initial, _limit(step, .85))
    return initial


def _temperature(score):
    index = min(16, score.shape[1] - 1)
    return (.004 + (score[:, :1] - score[:, index:index + 1]) / 3.2).clamp(.006, .080)


def _prior(y):
    # Equal-area cells use geocentric z and longitude for unit ECEF rays.
    band = ((y[:, 2].clamp(-1, 1) + 1) * 12).long().clamp(0, 23)
    lon = torch.atan2(y[:, 1], y[:, 0])
    col = ((lon + torch.pi) * (48 / (2 * torch.pi))).long().clamp(0, 47)
    cell = band * 48 + col
    counts = torch.bincount(cell, minlength=24 * 48).float().reshape(24, 48)
    horizontal = (counts.roll(1, 1) + 2 * counts + counts.roll(-1, 1)) / 4
    smoothed = (2 * horizontal + torch.cat((horizontal[:1], horizontal[:-1]))
                + torch.cat((horizontal[1:], horizontal[-1:]))) / 4
    return (smoothed.flatten()[cell] + 3).log()


def _probe(ref, generator):
    """Return LOO reference probes plus one hubness estimate per bank item."""
    n = len(ref)
    if n <= 1:
        return None, None, None, torch.zeros(n, device=ref.device)
    ids = torch.as_tensor(generator.choice(n, size=min(1536, n), replace=False), device=ref.device)
    neighbor_count = min(K, n - 1)
    hub_count = min(8, len(ids) - 1)
    hub = None
    rows, values = [], []
    for selected in ids.split(BLOCK):
        scores = ref[selected] @ ref.T
        scores[torch.arange(len(selected), device=ref.device), selected] = -torch.inf
        score, index = scores.topk(neighbor_count, dim=1)
        rows.append(index)
        values.append(score.float())
        top = scores.topk(min(hub_count, len(selected)), dim=0).values
        hub = top if hub is None else torch.cat((hub, top)).topk(
            min(hub_count, len(hub) + len(top)), dim=0).values
    return ids, torch.cat(rows), torch.cat(values), hub.float().mean(0)


def _features(scores, affinity, hub, prior):
    tau = _temperature(scores)
    log_visual = ((scores - scores[:, :1]) / tau).clamp(-25, 0)
    base = log_visual.softmax(1)
    support = (affinity @ base.unsqueeze(-1)).squeeze(-1).clamp_min(1e-7).log()
    ranks = torch.arange(1, 1 + scores.shape[1], device=scores.device).float().log()
    rank = ranks[None].expand_as(scores)
    hub = ((hub - hub.mean(1, keepdim=True)) / (tau + .02)).clamp(-4, 4)
    prior = (prior - prior.mean(1, keepdim=True)).clamp(-5, 5)
    return log_visual, torch.stack((log_visual, support, rank, hub, prior), dim=-1)


def _kernel(probe, neighbors, scores, y, hubs, prior):
    coeff = torch.zeros(5, device=y.device)
    if probe is None or len(probe) < 48:
        return coeff
    points = y[neighbors]
    cosine = points @ points.transpose(1, 2)
    affinity = ((cosine - 1) / .028).clamp(-50, 0).exp()
    logw, feat = _features(scores, affinity, hubs[neighbors], prior[neighbors])
    origin = _mean(points, logw.softmax(1), steps=1)
    offsets = _log(origin[:, None], points)
    target = _log(origin, y[probe])
    # Cross-fitted within the reference bank by explicitly masking each probe.
    # The L2 term and bounded coefficients regularize this five-parameter fit.
    theta = torch.zeros_like(coeff)
    first, second = torch.zeros_like(coeff), torch.zeros_like(coeff)
    for i in range(1, 46):
        bounded = theta.tanh()
        coef = 2 * bounded
        weights = (logw + (feat * coef).sum(-1)).softmax(1)
        position = (offsets * weights.unsqueeze(-1)).sum(1)
        residual = position - target
        grad_logit = 2 * weights * ((offsets - position[:, None]) * residual[:, None]).sum(-1)
        grad_coef = (grad_logit[:, :, None] * feat).sum(1).mean(0) + .0012 * coef
        grad = 2 * (1 - bounded.square()) * grad_coef
        first.mul_(.9).add_(grad, alpha=.1)
        second.mul_(.999).addcmul_(grad, grad, value=.001)
        theta.addcdiv_(first / (1 - .9**i),
                       (second / (1 - .999**i)).sqrt() + 1e-8, value=-.035)
    return 2 * theta.tanh()


def _basins(points, weight, distance, affinity, support):
    rows = torch.arange(len(weight), device=weight.device)
    dominant = support.argmax(dim=1)
    c0 = points[rows, dominant]
    other = (weight * distance[rows, dominant].square()).argmax(dim=1)
    c1 = points[rows, other]
    for _ in range(3):
        first_dot = (points * c0[:, None]).sum(-1)
        second_dot = (points * c1[:, None]).sum(-1)
        member = ((first_dot - second_dot) / .018).sigmoid()
        w0 = weight * member
        w1 = weight * (1 - member)
        c0 = _unit((w0[:, :, None] * points).sum(1), c0)
        c1 = _unit((w1[:, :, None] * points).sum(1), c1)
    masses = torch.stack((w0.sum(1), w1.sum(1)), dim=1).clamp_min(1e-8)
    mix = (1.28 * masses.log()).softmax(1)[:, 1:2]
    combined = _exp(c0, mix * _log(c0, c1))
    local = weight * affinity[rows, dominant]
    local = local / local.sum(1, keepdim=True).clamp_min(1e-10)
    return combined, _mean(points, local, initial=points[rows, dominant])


def _local_ridge(delta, offsets, base, weight):
    v = delta[:, :, :8]
    average_x = (weight[:, :, None] * v).sum(1)
    average_y = (weight[:, :, None] * offsets).sum(1)
    x = v - average_x[:, None]
    y = offsets - average_y[:, None]
    xtw = x.transpose(1, 2) * weight[:, None]
    gram = xtw @ x
    diag = gram.diagonal(dim1=1, dim2=2)
    penalty = .18 * diag.mean(1, keepdim=True) + .025
    diag.add_(penalty)
    coefs = torch.linalg.solve_ex(gram, xtw @ y).result
    correction = average_y - (average_x[:, None] @ coefs)[:, 0]
    return _exp(base, _limit(.75 * correction, .36))


def _barycentric(delta, base, logw, offsets):
    # Exponentiated gradient on squared visual reconstruction error plus
    # KL(w || visual_weights). These weights also apply to geographic offsets.
    weight = logw.softmax(1)
    scale = (delta.square().sum(2) * weight).sum(1, keepdim=True).clamp_min(.5)
    for _ in range(7):
        residual = (weight[:, :, None] * delta).sum(1)
        gradient = (delta * residual[:, None]).sum(2) / scale
        log_next = .70 * weight.clamp_min(1e-14).log() + .30 * logw - 1.25 * gradient
        weight = log_next.softmax(1)
    return _exp(base, (weight[:, :, None] * offsets).sum(1))


def _decode(scores, ids, points, delta, hubs, prior, coef):
    cosines = (points @ points.transpose(1, 2)).clamp(-1, 1)
    cosines.diagonal(dim1=1, dim2=2).fill_(1)
    distance = cosines.acos()
    affinity = ((cosines - 1) / .028).clamp(-50, 0).exp()
    ref_hub, ref_prior = hubs[ids], prior[ids]
    logw, features = _features(scores, affinity, ref_hub, ref_prior)
    weights = logw.softmax(1)
    base = _mean(points, weights)
    offsets = _log(base[:, None], points)
    rows = torch.arange(len(ids), device=ids.device)

    cost = (distance.square() @ weights.unsqueeze(-1)).squeeze(-1)
    medoid = points[rows, cost.argmin(1)]
    angular = offsets.norm(dim=2)
    threshold = (.045 + .42 * (weights * angular.square()).sum(1, keepdim=True).sqrt()).clamp(.045, .35)
    robust = medoid
    for _ in range(3):
        tangent = _log(robust[:, None], points)
        w = weights * (threshold / tangent.norm(dim=2).clamp_min(1e-7)).clamp(max=1)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-10)
        robust = _exp(robust, (w.unsqueeze(-1) * tangent).sum(1))
    support = (affinity @ weights.unsqueeze(-1)).squeeze(-1)
    two_basin, mode = _basins(points, weights, distance, affinity, support)

    def shifted(logits):
        # Initialize at the base intrinsic mean and re-evaluate tangents once
        # after the first shift, including when a distant group gains mass.
        w = logits.softmax(1)
        position = _exp(base, (w.unsqueeze(-1) * offsets).sum(1))
        return _mean(points, w, initial=position, steps=1)

    hub_delta = (ref_hub - ref_hub.mean(1, keepdim=True)) / _temperature(scores)
    values = (
        base,
        shifted(logw - .35 * ref_prior),
        medoid,
        robust,
        two_basin,
        _local_ridge(delta, offsets, base, weights),
        mode,
        shifted(logw - .5 * hub_delta),
        shifted(logw + (features * coef).sum(-1)),
        _barycentric(delta, base, logw, offsets),
    )
    return torch.stack(values, dim=0)


@torch.no_grad()
def predict(X_train, Y_train, X_query, seed, save_path=None, state_path=None):
    """Search a supplied reference bank, optionally persisting or replaying its calibration."""
    if len(X_train.shape) != 2 or len(X_query.shape) != 2 or X_train.shape[1] != X_query.shape[1]:
        raise ValueError("Training and query feature widths must match")
    if X_train.shape[1] < 1 or len(X_train) < 1 or tuple(Y_train.shape) != (len(X_train), 3):
        raise ValueError("Expected nonempty training vectors and aligned WGS84 XYZ")
    if not len(X_query):
        return {name: np.empty((0, 3), np.float32) for name in METHODS}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_type = torch.float16 if device.type == "cuda" else torch.float32

    def features(value):
        matrix = torch.as_tensor(value, device=device, dtype=torch.float32)
        if not torch.isfinite(matrix).all():
            raise ValueError("Feature transform produced nonfinite values")
        return F.normalize(matrix, dim=1).to(work_type)

    ref, query = features(X_train), features(X_query)
    y = torch.as_tensor(Y_train, dtype=torch.float32, device=device)
    norm = y.norm(dim=1, keepdim=True)
    if not torch.isfinite(norm).all() or (norm < 1e-8).any():
        raise ValueError("Reference coordinates must be finite nonzero XYZ")
    y = y / norm
    if state_path is None:
        generator = np.random.default_rng(int(seed))
        sampled, neighbor_ids, scores, hubs = _probe(ref, generator)
        prior = _prior(y)
        coef = _kernel(sampled, neighbor_ids, scores, y, hubs, prior)
        del sampled, neighbor_ids, scores
        projection = torch.as_tensor(generator.normal(size=(ref.shape[1], 16)),
                                     dtype=work_type, device=device)
        projected = (ref @ projection).float()
        scale = projected.std(dim=0, unbiased=False).clamp_min(.12)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp.pt")
            torch.save({"version": 1, "seed": int(seed), "reference_shape": list(ref.shape),
                        "hubs": hubs.cpu(), "prior": prior.cpu(), "coef": coef.cpu(),
                        "projection": projection.cpu(), "scale": scale.cpu()}, temp)
            temp.replace(path)
    else:
        state = torch.load(state_path, map_location=device, weights_only=True)
        if state["version"] != 1 or state["seed"] != int(seed) or state["reference_shape"] != list(ref.shape):
            raise ValueError("Retrieval state differs from the expected reference bank or seed")
        hubs, prior, coef = (state[k] for k in ("hubs", "prior", "coef"))
        projection = state["projection"].to(work_type)
        scale = state["scale"]
        projected = (ref @ projection).float()
    query_projected = (query @ projection).float()
    projected /= scale
    query_projected /= scale
    count = min(K, len(ref))
    output = np.empty((len(METHODS), len(query), 3), dtype=np.float32)
    for start in range(0, len(query), BLOCK):
        selection = slice(start, start + BLOCK)
        score, index = (query[selection] @ ref.T).topk(count, dim=1)
        delta = projected[index] - query_projected[selection, None]
        result = _decode(score.float(), index, y[index], delta, hubs, prior, coef)
        output[:, selection] = result.cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite geographic retrieval prediction")
    return dict(zip(METHODS, output))
