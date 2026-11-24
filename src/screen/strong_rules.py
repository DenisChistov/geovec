"""Fold-local WGS84 fusion of [V4, meta, paper distance-risk, raw champion]."""
from itertools import combinations

import numpy as np
from pyproj import Geod
from scipy.optimize import minimize
from scipy.spatial import cKDTree


RULES = {
    "strong:balanced": "Равные веса V4, метамодели и модели географического риска в плоскости WGS84.",
    "strong:fixed_v4": "Фиксированная смесь V4, метамодели и модели риска с весами 0,55 / 0,30 / 0,15.",
    "strong:convex": "Неотрицательные веса четырёх экспертов с уточнением по квадрату точной геодезической WGS84.",
    "strong:signed": "Регуляризованная линейная комбинация геодезических смещений с весами обоих знаков.",
    "strong:anisotropic": "Раздельные веса экспертов для восточной и северной составляющих смещения.",
    "strong:disagreement": "Выпуклые веса сглаживаются между обученными уровнями расхождения экспертов.",
    "strong:regional": "Выпуклые веса зависят от региона прогноза V4 и приближаются к общему решению в редких регионах.",
    "strong:shrink": "Поправка двух специалистов к V4 ослабляется при расхождении направлений; усиление оценивается на обучении.",
}

_GEOD = Geod(ellps="WGS84")
_E2 = 1 - (1 - 1 / 298.257223563) ** 2
_RADIUS = 6371008.8
_PRIOR = np.array([.40, .36, .21, .03])
_FIXED = {
    "strong:balanced": np.array([1 / 3, 1 / 3, 1 / 3, 0.]),
    "strong:fixed_v4": np.array([.55, .30, .15, 0.]),
}


def _unit(values, dimensions):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != dimensions or values.shape[-1] != 3:
        raise ValueError("Expected ECEF directions with three components")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Expected finite nonzero ECEF directions")
    return values / norms


def _lonlat(points):
    x, y, z = np.moveaxis(points, -1, 0)
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, (1 - _E2) * np.hypot(x, y)))


def _inverse(origin, dest):
    azimuth, _, distance = _GEOD.inv(*origin, *dest)
    angle = np.radians(azimuth)
    return np.column_stack([np.sin(angle), np.cos(angle)]) * (distance / _RADIUS)[:, None]


def _direct(origin, step):
    distance = np.linalg.norm(step, axis=1) * _RADIUS
    bearing = np.degrees(np.arctan2(step[:, 0], step[:, 1]))
    lon, lat, _ = _GEOD.fwd(*origin, bearing, distance)
    return lon, lat


def _to_ecef(lonlat):
    lon, lat = np.radians(lonlat)
    values = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                              (1 - _E2) * np.sin(lat)])
    return _unit(values, 2).astype(np.float32)


def _geometry(experts):
    e = _unit(experts, 3)
    if e.shape[1] != 4:
        raise ValueError("Expected four experts ordered V4, meta, paper, raw champion")
    a = e[:, 0]
    origin = _lonlat(a)
    d = np.stack([_inverse(origin, _lonlat(e[:, i])) for i in (1, 2, 3)], axis=2)
    return a, origin, d


def _ridge(g, b, penalty, prior):
    return np.linalg.solve(g + penalty * np.eye(len(g)), b + penalty * prior)


def _simplex(d, y, prior, penalty):
    # Include V4 as a zero offset. Enumerating 15 active sets solves the
    # regularized four-weight quadratic program, including boundary optima.
    x = np.column_stack([np.zeros(2 * len(d)), d.reshape(-1, 3)])
    g = x.T @ x + penalty * np.eye(4)
    b = x.T @ y.ravel() + penalty * prior
    best, loss = prior.copy(), np.inf
    for k in range(1, 5):
        for face in combinations(range(4), k):
            ids = np.asarray(face)
            a = g[np.ix_(ids, ids)]
            system = np.ones((k + 1, k + 1))
            system[:-1, :-1] = a
            system[-1, -1] = 0.
            weights = np.linalg.solve(system, np.r_[b[ids], 1.])[:-1]
            if np.min(weights) < -1e-8:
                continue
            value = weights @ a @ weights - 2 * (b[ids] @ weights)
            if value < loss:
                best = np.zeros(4)
                best[ids] = np.maximum(weights, 0)
                best /= best.sum()
                loss = value
    return best


def _weighted(d, weights):
    if np.ndim(weights) == 1:
        return d @ weights[1:]
    return np.einsum("ndj,nj->nd", d, weights[:, 1:])


def _exact_convex(d, y, origin, truth_ll, initial, penalty, seed):
    # A fixed sample bounds geodesic calls on large folds. Subtract the
    # sampled tangent cost and add its full-fold expectation as a control
    # variate: the sampled residual corrects the planar geometry, while
    # every training row informs the main quadratic objective.
    n = len(d)
    sample = np.random.default_rng(seed).choice(n, min(16000, n), replace=False)
    part = d[sample]
    here = tuple(a[sample] for a in origin)
    target = tuple(a[sample] for a in truth_ll)
    response = y[sample]
    full = d.reshape(-1, 3)
    gram = full.T @ full / n
    cross = full.T @ y.ravel() / n
    scale = max(gram.diagonal().mean(), 1e-5)

    def objective(w):
        step = _weighted(part, w)
        pred = _direct(here, step)
        meters = _GEOD.inv(*pred, *target)[2]
        correction = np.mean((meters / _RADIUS) ** 2 - ((response - step) ** 2).sum(1))
        linear = w[1:] @ gram @ w[1:] - 2 * w[1:] @ cross
        return (linear + correction + penalty / n * np.sum((w - _PRIOR) ** 2)) / scale

    opt = minimize(objective, initial, method="SLSQP", bounds=[(0., 1.)] * 4,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                 "jac": lambda w: np.ones(4)}],
                   options={"ftol": 1e-8, "maxiter": 25, "eps": 1e-5})
    if opt.success and np.isfinite(opt.fun) and opt.fun <= objective(initial) + 1e-9:
        w = np.maximum(opt.x, 0.)
        return w / w.sum()
    return initial


def _disagreement(d):
    # Measure disagreement about location in radians on WGS84.
    sizes = np.linalg.norm(d, axis=1)
    return np.log1p(20 * sizes.max(axis=1)) + .5 * np.log1p(20 * sizes.mean(axis=1))


def _regions():
    i = np.arange(48)
    z = 1 - (2 * i + 1) / len(i)
    angle = i * np.pi * (3 - np.sqrt(5))
    rho = np.sqrt(1 - z * z)
    return np.column_stack([rho * np.cos(angle), rho * np.sin(angle), z])


def _shrink_basis(d):
    meta, paper = d[:, :, 0], d[:, :, 1]
    shared = .62 * meta + .38 * paper
    signal = np.sum(shared**2, axis=1)
    disagreement = np.sum((meta - paper)**2, axis=1)
    agreement = signal / (signal + .40 * disagreement + 1e-8)
    return shared, np.stack([shared, agreement[:, None] * shared], axis=2)


def fit_rule(name, train, y, seed=0):
    """Fit on aligned out-of-fold predictions and geocentric unit ECEF truth."""
    if name not in RULES:
        raise KeyError(name)
    if name in _FIXED:
        return {"weights": _FIXED[name].copy()}
    a, origin, d = _geometry(train)
    truth = _unit(y, 2)
    if not len(a) or truth.shape != a.shape:
        raise ValueError("Expected one training target per nonempty expert row")
    truth_ll = _lonlat(truth)
    target = _inverse(origin, truth_ll)
    n = len(a)
    strength = max(np.sum(d*d) / (3 * n), 1e-6)
    if name == "strong:shrink":
        _, basis = _shrink_basis(d)
        x = basis.reshape(-1, 2)
        g = x.T @ x
        slope = _ridge(g, x.T @ target.ravel(), .12 * max(np.trace(g) / 2, n * 1e-6),
                       np.array([.24, .38]))
        return {"slope": slope}
    if name == "strong:anisotropic":
        coefs = []
        for axis in range(2):
            x = d[:, axis, :]
            coefs.append(_ridge(x.T @ x, x.T @ target[:, axis], .045 * n * strength, _PRIOR[1:]))
        return {"coef": np.asarray(coefs).clip(-.4, 1.25)}
    if name == "strong:signed":
        x = d.reshape(-1, 3)
        coef = _ridge(x.T @ x, x.T @ target.ravel(), .025 * n * strength, _PRIOR[1:])
        return {"coef": np.clip(coef, -.45, 1.3)}
    penalty = .015 * n * strength
    global_w = _simplex(d, target, _PRIOR, penalty)
    if name == "strong:convex":
        return {"weights": _exact_convex(d, target, origin, truth_ll, global_w, penalty, seed)}
    state = {}
    if name == "strong:disagreement":
        spread = _disagreement(d)
        knots = np.unique(np.quantile(spread, [.05, .25, .5, .75, .90, .98]))
        ids = np.searchsorted((knots[:-1] + knots[1:]) * .5, spread)
        state["knots"] = knots
        count = len(knots)
    else:
        centers = _regions()
        ids = cKDTree(centers).query(a)[1]
        state["centers"] = centers
        count = len(centers)
    weights = np.tile(global_w, (count, 1))
    pseudo = max(80., .005 * n) * strength
    for j in range(count):
        mask = ids == j
        if mask.any():
            weights[j] = _simplex(d[mask], target[mask], global_w, pseudo + .025 * mask.sum() * strength)
    state["weights"] = weights
    return state


def apply_rule(name, state, query):
    """Return unit ECEF predictions using only the saved rule and four experts."""
    if name not in RULES:
        raise KeyError(name)
    a, origin, d = _geometry(query)
    if not len(a):
        return np.empty((0, 3), dtype=np.float32)
    if name == "strong:disagreement":
        spread = _disagreement(d)
        weights = np.column_stack([np.interp(spread, state["knots"], w) for w in state["weights"].T])
        step = _weighted(d, weights)
    elif name == "strong:regional":
        distance, ids = cKDTree(state["centers"]).query(a, k=4)
        kernel = np.exp(-18 * (distance**2 - distance[:, :1]**2))
        kernel /= kernel.sum(axis=1, keepdims=True)
        weights = np.einsum("nr,nrj->nj", kernel, state["weights"][ids])
        step = _weighted(d, weights)
    elif name == "strong:shrink":
        shared, basis = _shrink_basis(d)
        size = np.sum(shared**2, axis=1)
        gain = np.sum((basis @ state["slope"]) * shared, axis=1) / np.maximum(size, 1e-20)
        step = shared * np.clip(gain, 0, 1.15)[:, None]
    elif name == "strong:signed":
        step = d @ state["coef"]
    elif name == "strong:anisotropic":
        step = np.einsum("ndj,dj->nd", d, state["coef"])
    else:
        step = _weighted(d, state["weights"])
    # Signed or axis-specific extrapolation can exceed half the circumference
    # when one reference is almost antipodal. Use the shortest-path extent.
    size = np.linalg.norm(step, axis=1, keepdims=True)
    step *= np.minimum(1., np.pi / np.maximum(size, 1e-20))
    return _to_ecef(_direct(origin, step))
