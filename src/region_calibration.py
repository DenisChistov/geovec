"""Cross-fitted calibration of seven geographic experts in a local tangent plane."""

from itertools import combinations

import numpy as np


DESCRIPTIONS = {
    "cal_global_simplex": "Fit one convex mixture of the seven expert log maps about the multiscale forecast using ellipsoid-scaled squared error.",
    "cal_regional_simplex": "Fit convex tangent-plane mixtures by geographic region, with smooth memberships around 24 spherical centers and shrinkage toward the global fit.",
    "cal_spread_simplex": "Fit convex tangent-plane mixtures across seven training quantiles of expert disagreement and interpolate weights for each query.",
}

_COUNT = 7
_ANCHOR = 1
_FLATTENING = 1.0 / 298.257223563
_EPS = 1e-7


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(length).all() or np.any(length <= 1e-12):
        raise ValueError("Expected finite nonzero ECEF directions.")
    return x / length


def _log(points, anchor, east, north):
    """Resolve the spherical log map into two orthonormal basis directions."""
    e = np.einsum("nkd,nd->nk", points, east)
    n = np.einsum("nkd,nd->nk", points, north)
    dot = np.einsum("nkd,nd->nk", points, anchor)
    sine = np.hypot(e, n)
    angle = np.arctan2(sine, dot)
    scale = np.divide(angle, sine, out=np.ones_like(sine), where=sine > _EPS)
    e *= scale
    n *= scale
    # At an exact antipode every great circle has the same length.
    antipodal = (sine <= _EPS) & (dot < 0)
    e[antipodal], n[antipodal] = np.pi, 0
    return np.stack((e, n), axis=-1)


def _geometry(experts):
    if np.ndim(experts) != 3 or np.shape(experts)[1:] != (_COUNT, 3):
        raise ValueError("Expected [rows, 7, 3] experts in the V3 expert order.")
    points = _unit(experts)
    anchor = points[:, _ANCHOR]
    lon = np.arctan2(anchor[:, 1], anchor[:, 0])
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=1)
    north = np.cross(anchor, east)
    tangent = _log(points, anchor, east, north)
    tangent[:, _ANCHOR] = 0
    return anchor, east, north, tangent


def _regional_centers():
    i = np.arange(24, dtype=np.float64)
    z = 1 - 2 * (i + 0.5) / len(i)
    lon = np.pi * (3 - np.sqrt(5.0)) * i
    xy = np.sqrt(1 - z * z)
    return np.stack((xy * np.cos(lon), xy * np.sin(lon), z), axis=1).astype(np.float32)


def _spread(tangent):
    return np.sqrt(np.square(tangent).sum(axis=(1, 2)) / _COUNT)


def _memberships(name, centers, anchor, spread):
    if name == "cal_regional_simplex":
        score = (anchor @ centers.T) * 16.0
        score -= score.max(axis=1, keepdims=True)
        w = np.exp(score)
        return w / w.sum(axis=1, keepdims=True)
    if len(centers) == 1:
        return np.ones((len(anchor), 1), dtype=np.float32)
    right = np.searchsorted(centers, spread).clip(1, len(centers) - 1)
    left = right - 1
    t = np.clip((spread - centers[left]) / (centers[right] - centers[left]), 0, 1)
    rows = np.arange(len(anchor))
    w = np.zeros((len(rows), len(centers)), dtype=np.float32)
    w[rows, left], w[rows, right] = 1 - t, t
    return w


def _solve(grams, moments):
    """Solve the small simplex QPs by enumerating the 127 nonempty faces."""
    groups = len(grams)
    anchor_weights = np.eye(_COUNT, dtype=np.float64)[_ANCHOR]
    # Among indistinguishable expert mixtures prefer the multiscale anchor.
    jitter = 1e-9 * np.maximum(np.trace(grams, axis1=1, axis2=2) / _COUNT, 1e-10)
    grams = grams + jitter[:, None, None] * np.eye(_COUNT)
    moments = moments + jitter[:, None] * anchor_weights
    best = np.full(groups, np.inf)
    weights = np.zeros((groups, _COUNT), dtype=np.float64)
    for size in range(1, _COUNT + 1):
        for subset in combinations(range(_COUNT), size):
            ix = np.asarray(subset)
            gram = grams[:, ix][:, :, ix]
            moment = moments[:, ix]
            if size == 1:
                solution = np.ones_like(moment)
            else:
                # Solve the equality-constrained least-squares system on this face.
                kkt = np.empty((groups, size + 1, size + 1), dtype=np.float64)
                kkt[:, :-1, :-1] = gram
                kkt[:, -1, :-1] = kkt[:, :-1, -1] = 1
                kkt[:, -1, -1] = 0
                rhs = np.pad(moment, ((0, 0), (0, 1)), constant_values=1)
                solution = np.linalg.solve(kkt, rhs[:, :, None])[:, :-1, 0]
            value = 0.5 * np.einsum("gi,gij,gj->g", solution, gram, solution)
            value -= (moment * solution).sum(axis=1)
            better = (solution.min(axis=1) >= -1e-8) & (value < best)
            rows = np.flatnonzero(better)
            weights[rows] = 0
            weights[np.ix_(rows, ix)] = solution[rows].clip(0)
            best[rows] = value[rows]
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def _statistics(anchor, tangent, target):
    # A unit ECEF ray intersects WGS84 at rho * a. Differentiating that
    # surface point gives the east and north length scales (in units of
    # the semimajor radius) for the geocentric tangent frame.
    z = anchor[:, 2]
    ratio = (1 - _FLATTENING) ** -2 - 1
    rho = 1 / np.sqrt(1 + ratio * z * z)
    radial_slope = ratio * z * np.hypot(anchor[:, 0], anchor[:, 1]) * np.square(rho)
    scales = np.stack((rho, rho * np.sqrt(1 + np.square(radial_slope))), axis=1)
    x = tangent * scales[:, None, :]
    y = target * scales
    # Only the upper triangle is needed by the regional reductions.
    i, j = np.triu_indices(_COUNT)
    gram = (x[:, i] * x[:, j]).sum(axis=2)
    moments = (x * y[:, None, :]).sum(axis=2)
    return np.concatenate((gram, moments), axis=1)


def _solve_stats(stats):
    i, j = np.triu_indices(_COUNT)
    gram = np.empty((len(stats), _COUNT, _COUNT), dtype=np.float64)
    gram[:, i, j] = gram[:, j, i] = stats[:, :len(i)]
    return _solve(gram, stats[:, len(i):])


def fit_many(names, experts_train, y_train):
    """Fit requested calibrations with shared training-only geometry."""
    if isinstance(names, str):
        names = [names]
    names = tuple(dict.fromkeys(names))
    if any(name not in DESCRIPTIONS for name in names):
        raise ValueError(f"Unknown calibration in {names!r}")
    anchor, east, north, tangent = _geometry(experts_train)
    if np.shape(y_train) != anchor.shape or not len(anchor):
        raise ValueError("Training labels and expert rows must agree and be nonempty.")
    target = _log(_unit(y_train)[:, None], anchor, east, north)[:, 0]
    stats = _statistics(anchor, tangent, target)
    global_stats = stats.mean(axis=0, dtype=np.float64)
    spread = _spread(tangent)
    result = {}
    for name in names:
        if name == "cal_global_simplex":
            result[name] = {"weights": _solve_stats(global_stats[None])}
            continue
        if name == "cal_regional_simplex":
            centers = _regional_centers()
        else:
            centers = np.unique(np.quantile(spread, np.linspace(0.05, 0.95, 7)).astype(np.float32))
        membership = _memberships(name, centers, anchor, spread)
        # Double precision keeps the small least-squares systems well
        # conditioned when near-identical experts generate correlated columns.
        weighted = membership.T.astype(np.float64) @ stats.astype(np.float64)
        weighted += max(64.0, len(anchor) * 0.003) * global_stats[None]
        result[name] = {"weights": _solve_stats(weighted), "centers": centers}
    return result


def fit(name, experts_train, y_train):
    return fit_many([name], experts_train, y_train)[name]


def apply_many(models, experts_eval):
    """Predict every supplied calibration, sharing query geometry."""
    anchor, east, north, tangent = _geometry(experts_eval)
    spread = _spread(tangent)
    predictions = {}
    for name, params in models.items():
        if name not in DESCRIPTIONS:
            raise ValueError(f"Unknown calibration {name!r}")
        weights = params["weights"]
        if name != "cal_global_simplex":
            weights = _memberships(name, params["centers"], anchor, spread) @ weights
        step = (tangent * weights[:, :, None]).sum(axis=1)
        angle = np.linalg.norm(step, axis=1, keepdims=True)
        vector = step[:, :1] * east + step[:, 1:] * north
        pred = np.cos(angle) * anchor + np.sinc(angle / np.pi) * vector
        predictions[name] = _unit(pred)
    return predictions


def apply(name, params, experts_eval):
    return apply_many({name: params}, experts_eval)[name]
