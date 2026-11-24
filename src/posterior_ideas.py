"""Sixteen fixed posterior-based geographic forecasts from saved V3 bundles."""

import numpy as np


DESCRIPTIONS = {
    "uncertainty_adaptive_temperature": "Температура апостериорного распределения определяется энтропией; интерполяция между сохранёнными прогнозами.",
    "prior_odds_reference_shrinkage": "Вес узкого прогноза определяется превышением апостериорной вероятности моды над её частотой в обучении.",
    "rank_entropy_weighted_mean": "Среднее трёх наборов топ-10 центров с весами уверенности и концентрации каждого распределения.",
    "rank_tail_completed_mean": "Объединение топ-10 центров с дополнением пропущенной массы глобальными прогнозами.",
    "rank_frechet_medoid": "Выбор центра из объединённых топ-10 с минимальным ожидаемым квадратом углового расстояния.",
    "rank_spherical_median": "Робастная взвешенная сферическая медиана объединённых топ-10 центров.",
    "rank_trimmed_frechet": "Среднее Фреше после удаления наиболее далёких 18% массы около глобального прогноза.",
    "rank_spatial_mode": "Плотностная мода объединённых центров, затем локальное среднее внутри ядра плотности.",
    "rank_connected_component": "Среднее географической компоненты, содержащей центр с наибольшей суммарной массой ближайших по угловому порогу соседей.",
    "rank_product_mode": "Географическая мода произведения локальных плотностей трёх разбиений.",
    "rank_minimax_consensus": "Центр, минимизирующий максимальный ожидаемый квадрат ошибки среди трёх разбиений.",
    "expert_geometric_median": "Геометрическая медиана семи прогнозов-экспертов на сфере.",
    "expert_consensus_reweight": "Среднее экспертов с весом по числу близких согласующихся прогнозов.",
    "posterior_scored_expert": "Выбор эксперта по ожидаемой квадратичной ошибке объединённого распределения с дополнением хвоста.",
    "posterior_local_adaptive": "Адаптивное смешивание глобального и локального прогнозов по их согласованию и апостериорной поддержке.",
    "confident_scale_routing": "Переход к наиболее уверенному масштабу разбиения при достаточном отрыве от остальных.",
}

_RANKS = 10
_POSTERIOR_WIDTH = 53
_BATCH = 2048


def _weights(values):
    total = values.sum(axis=-1, keepdims=True)
    return np.where(total > 1e-8, values / np.maximum(total, 1e-8), 1.0 / values.shape[-1])


def _unit(values, fallback):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(norm > 1e-7, values / np.maximum(norm, 1e-7), fallback)


def _blend(a, b, w):
    w = np.asarray(w, dtype=np.float32).reshape(-1, 1)
    dot = np.clip((a * b).sum(axis=1, keepdims=True), -1, 1)
    angle = np.arccos(dot)
    ordinary = (angle > 1e-4) & (angle < np.pi - 1e-4)
    sine = np.where(ordinary, np.sin(angle), 1.0)
    wa = np.where(ordinary, np.sin((1 - w) * angle) / sine, 1 - w)
    wb = np.where(ordinary, np.sin(w * angle) / sine, w)
    return _unit(wa * a + wb * b, a)


def _frechet(points, weights, fallback, *, median=False):
    weights = _weights(weights)
    mean = _unit(np.einsum("nk,nki->ni", weights, points), fallback)
    for _ in range(4):
        dot = np.clip(np.einsum("ni,nki->nk", mean, points), -1, 1)
        tangent = points - dot[:, :, None] * mean[:, None, :]
        length = np.linalg.norm(tangent, axis=2)
        angle = np.arctan2(length, dot)
        direction = tangent / np.maximum(length[:, :, None], 1e-7)
        if median:
            inverse = (weights / np.maximum(angle, 0.002)).sum(axis=1, keepdims=True)
            step = np.einsum("nk,nki->ni", weights, direction) / np.maximum(inverse, 1e-7)
        else:
            step = np.einsum("nk,nki->ni", weights * angle, direction)
        size = np.linalg.norm(step, axis=1, keepdims=True)
        step /= np.maximum(size / 0.9, 1.0)
        size = np.minimum(size, 0.9)
        mean = _unit(np.cos(size) * mean + np.sinc(size / np.pi) * step, mean)
    return mean


def _angles(a, b):
    """Pairwise central angles, for arrays of shape [row, candidates, 3]."""
    cosine = np.clip(a @ np.swapaxes(b, 1, 2), -1, 1)
    return np.arccos(cosine)


def _choose(points, index):
    return points[np.arange(len(points)), index]


def _best_region(points, raw_weights, affinity, score, fallback):
    valid = raw_weights > 0
    best = np.argmax(np.where(valid, score, -np.inf), axis=1)
    local = _choose(affinity, best) * raw_weights
    return _frechet(points, local, fallback)


def _adaptive_temperature(bundle, entropy):
    temperatures = np.array([0.6, 0.8, 1.0, 1.2, 1.6], dtype=np.float32)
    directions = np.stack([bundle[name] for name in ("temp_06", "temp_08", "reference_512", "temp_12", "temp_16")], axis=1)
    chosen = 0.6 + np.clip(entropy, 0, 1)
    left = np.clip(np.searchsorted(temperatures, chosen, side="right") - 1, 0, len(temperatures) - 2)
    fraction = (chosen - temperatures[left]) / (temperatures[left + 1] - temperatures[left])
    return _blend(_choose(directions, left), _choose(directions, left + 1), fraction)


def _trimmed_weights(points, weights, anchor, mass):
    angles = _angles(anchor[:, None, :], points)[:, 0]
    order = np.argsort(angles, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    previous = sorted_weights.cumsum(axis=1) - sorted_weights
    kept = np.minimum(sorted_weights, np.maximum(mass - previous, 0.0))
    result = np.zeros_like(weights)
    np.put_along_axis(result, order, kept, axis=1)
    return result


def _connected_mode(points, weights, angle, bandwidth, fallback):
    valid = weights > 0
    edge = (angle <= bandwidth[:, None, None]) & valid[:, :, None] & valid[:, None, :]
    density = np.einsum("nij,nj->ni", edge, weights)
    start = np.argmax(np.where(valid, density, -np.inf), axis=1)
    connected = _choose(edge, start)
    for _ in range(points.shape[1]):
        expanded = np.any(edge & connected[:, :, None], axis=1)
        if np.array_equal(connected, expanded):
            break
        connected = expanded
    return _frechet(points, weights * connected, fallback)


def _posterior_features(bundle):
    offset = bundle["experts"].shape[1] * 3
    end = offset + 3 * _POSTERIOR_WIDTH
    if bundle["features"].ndim != 2 or bundle["features"].shape[1] < end:
        raise ValueError("Expected the V3 bundle schema with three posterior feature blocks")
    block = bundle["features"][:, offset:end].reshape(-1, 3, _POSTERIOR_WIDTH)
    ranks = block[:, :, :4 * _RANKS].reshape(-1, 3, _RANKS, 4)
    prob = np.clip(ranks[:, :, :, 0], 0, 1)
    anchor = bundle["ensemble_multiscale"]
    xyz = _unit(ranks[:, :, :, 1:], anchor[:, None, None, :])
    entropy = np.clip(block[:, :, 41], 0, 1)
    resultant = np.clip(block[:, :, 47], 0, 1)
    rms = np.maximum(block[:, :, 49], 0)
    mode_prior = np.maximum(block[:, :, 52], 1e-7)
    return prob, xyz, entropy, resultant, rms, mode_prior


def _predict_batch(bundle):
    anchor = bundle["ensemble_multiscale"]
    reference = bundle["reference_512"]
    n = len(anchor)
    prob, xyz, entropy, resultant, rms, mode_prior = _posterior_features(bundle)
    points = xyz.reshape(n, -1, 3)
    mass = np.clip(prob.sum(axis=2), 0, 1)
    pooled = _weights(prob.reshape(n, -1))
    width = np.clip(rms.mean(axis=1) * .25, .045, .40)
    angle = _angles(points, points)
    angle2 = angle ** 2
    kernel = np.exp(-0.5 * angle2 / width[:, None, None] ** 2)
    result = {}

    result["uncertainty_adaptive_temperature"] = _adaptive_temperature(bundle, entropy[:, 0])
    odds = np.maximum(prob[:, 0, 0] / mode_prior[:, 0], 1.0)
    odds_confidence = np.clip(np.log(odds) / np.log(30.0), 0, 1)
    result["prior_odds_reference_shrinkage"] = _blend(anchor, bundle["temp_08"], odds_confidence * (1 - entropy[:, 0]))

    reliability = np.sqrt(np.maximum(0.01, (1 - entropy) * resultant))
    entropy_weights = (prob * reliability[:, :, None]).reshape(n, -1)
    result["rank_entropy_weighted_mean"] = _frechet(points, entropy_weights, anchor)

    tail_centers = np.stack([reference, anchor, anchor], axis=1)
    completed_points = np.concatenate([points, tail_centers], axis=1)
    completed_weights = _weights(np.column_stack([prob.reshape(n, -1), 1.0 - mass]))
    result["rank_tail_completed_mean"] = _frechet(completed_points, completed_weights, anchor)

    risk = np.einsum("nij,nj->ni", angle2, pooled)
    medoid = np.argmin(np.where(pooled > 0, risk, np.inf), axis=1)
    result["rank_frechet_medoid"] = _choose(points, medoid)
    result["rank_spherical_median"] = _frechet(points, pooled, anchor, median=True)
    trimmed = _trimmed_weights(points, pooled, anchor, 0.82)
    result["rank_trimmed_frechet"] = _frechet(points, trimmed, anchor)

    density = np.einsum("nij,nj->ni", kernel, pooled)
    result["rank_spatial_mode"] = _best_region(points, pooled, kernel, density, anchor)
    connectivity_radius = np.clip(1.4 * width, .07, .55)
    result["rank_connected_component"] = _connected_mode(points, pooled, angle, connectivity_radius, anchor)

    support = np.einsum("nism,nsm->nis", kernel.reshape(n, points.shape[1], 3, _RANKS), prob)
    product_score = np.log(0.015 + support).sum(axis=2)
    result["rank_product_mode"] = _best_region(points, pooled, kernel, product_score, anchor)

    balanced = _weights(prob).reshape(n, 3, _RANKS)
    scale_risk = np.einsum("nism,nsm->nis", angle2.reshape(n, points.shape[1], 3, _RANKS), balanced)
    max_risk = scale_risk.max(axis=2)
    minimax = np.argmin(np.where(pooled > 0, max_risk, np.inf), axis=1)
    result["rank_minimax_consensus"] = _choose(points, minimax)

    experts = bundle["experts"]
    expert_count = experts.shape[1]
    uniform = np.ones((n, expert_count), dtype=np.float32)
    result["expert_geometric_median"] = _frechet(experts, uniform, anchor, median=True)
    expert_angle = _angles(experts, experts)
    agreement_radius = np.maximum(np.median(expert_angle[:, np.triu_indices(expert_count, 1)[0], np.triu_indices(expert_count, 1)[1]], axis=1), 0.03)
    degrees = np.exp(-0.5 * (expert_angle / agreement_radius[:, None, None]) ** 2).sum(axis=2) - 1
    result["expert_consensus_reweight"] = _frechet(experts, np.maximum(degrees, 0.001), anchor)

    expert_risks = np.einsum("nij,nj->ni", _angles(experts, completed_points) ** 2, completed_weights)
    result["posterior_scored_expert"] = _choose(experts, expert_risks.argmin(axis=1))

    local = bundle["local_refinement"]
    alternative = np.stack([anchor, local], axis=1)
    support_angle = _angles(alternative, points)
    support_kernel = np.exp(-0.5 * (support_angle / width[:, None, None]) ** 2)
    local_support = np.einsum("nij,nj->ni", support_kernel, pooled)
    local_share = (local_support[:, 1] + .02) / (local_support.sum(axis=1) + .04)
    gap = _angles(anchor[:, None, :], local[:, None, :])[:, 0, 0]
    agreement = np.exp(-0.5 * (gap / np.maximum(rms.mean(axis=1), .12)) ** 2)
    result["posterior_local_adaptive"] = _blend(anchor, local, (0.15 + 0.70 * local_share) * agreement)

    scale_score = reliability * mass
    scale_order = np.argsort(scale_score, axis=1)
    scale = scale_order[:, -1]
    scale_points = _choose(xyz, scale)
    scale_weights = _choose(prob, scale)
    strongest = _frechet(scale_points, scale_weights, anchor)
    best_score = scale_score[np.arange(n), scale]
    next_score = scale_score[np.arange(n), scale_order[:, -2]]
    margin = np.clip((best_score - next_score) / np.maximum(best_score, 1e-5), 0, 1)
    result["confident_scale_routing"] = _blend(anchor, strongest, margin)
    return result


def predict(bundle):
    """Evaluate all sixteen label-free candidates in bounded-memory CPU batches."""
    anchor = np.asarray(bundle["ensemble_multiscale"], dtype=np.float32)
    if anchor.ndim != 2 or anchor.shape[1] != 3:
        raise ValueError("Expected [rows,3] geography predictions")
    source_names = ("ensemble_multiscale", "reference_512", "temp_06", "temp_08", "temp_12", "temp_16", "local_refinement", "experts", "features")
    source = {name: np.asarray(bundle[name], dtype=np.float32) for name in source_names}
    if any(len(value) != len(anchor) or not np.isfinite(value).all() for value in source.values()):
        raise ValueError("V3 prediction bundle contains missing or nonfinite data")
    out = {name: np.empty_like(anchor) for name in DESCRIPTIONS}
    for start in range(0, len(anchor), _BATCH):
        rows = slice(start, start + _BATCH)
        values = _predict_batch({name: value[rows] for name, value in source.items()})
        for name in DESCRIPTIONS:
            out[name][rows] = _unit(values[name], anchor[rows])
    return out
