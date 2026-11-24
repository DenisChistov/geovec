"""Absolute-risk-weighted geometric medians in a WGS84 azimuthal plane."""
from itertools import product

import numpy as np

from tune_rules import E2, GEOD, R, lonlat, unit


def grid():
    specs = {}
    for pool, risk, power, location in product(
            ("core", "all"), ("global", "spread", "agreement"), (2, 6), ("median", "blend")):
        name = f"mae:median/{pool}/{risk}/p{power}/{location}"
        specs[name] = dict(kind="mae_median", pool=pool, risk=risk, power=power, location=location)
    return specs


def _points(g, pool):
    # A geometry dictionary belongs to a single fold/split. Reuse solely
    # its query-derived offsets when many grid entries evaluate that split.
    cache = g.setdefault("_mae_median_points", {})
    if pool not in cache:
        if pool not in ("core", "all"):
            raise ValueError("Expected core or all expert pool")
        d = g["d"][:, :, :3] if pool == "core" else g["d"]
        n = len(d)
        east = np.concatenate([np.zeros((n, 1), np.float32), d[:, 0]], axis=1)
        north = np.concatenate([np.zeros((n, 1), np.float32), d[:, 1]], axis=1)
        width = np.hypot(east, north).mean(axis=1)
        med_e, med_n = np.median(east, axis=1), np.median(north, axis=1)
        isolation = np.hypot(east - med_e[:, None], north - med_n[:, None])
        radius = np.median(isolation, axis=1, keepdims=True)
        peer = .10 + .90 / (1 + (isolation / (.004 + radius))**2)
        cache[pool] = east, north, np.log1p(20*width), peer
    return cache[pool]


def _dest(here, east, north):
    bearing = np.degrees(np.arctan2(east, north)).astype(np.float64)
    distance = np.hypot(east, north).astype(np.float64) * R
    return GEOD.fwd(*here, bearing, distance)[:2]


def _errors(g):
    """Exact geodesic distances to training truth in km, recovered from OOF offsets."""
    if "_mae_median_errors" not in g:
        if "target" not in g or not len(g["target"]):
            raise ValueError("Fitting a median requires nonempty training targets")
        here = lonlat(g["anchor"])
        target = g["target"]
        truth = _dest(here, target[:, 0], target[:, 1])
        d = g["d"]
        errors = np.empty((len(target), 1 + d.shape[2]), np.float32)
        errors[:, 0] = np.hypot(target[:, 0], target[:, 1]) * (R / 1000.)
        for j in range(d.shape[2]):
            pred = _dest(here, d[:, 0, j], d[:, 1, j])
            errors[:, j+1] = GEOD.inv(*pred, *truth)[2] * .001
        g["_mae_median_errors"] = errors
    return g["_mae_median_errors"]


def _risk_tables(g, pool):
    cache = g.setdefault("_mae_median_tables", {})
    if pool not in cache:
        east, _, spread, _ = _points(g, pool)
        errors = _errors(g)[:, :east.shape[1]]
        global_mae = errors.mean(axis=0, dtype=np.float64)
        knots = np.unique(np.quantile(spread, [.06, .25, .50, .75, .94]))
        regions = np.searchsorted((knots[1:] + knots[:-1]) * .5, spread)
        risk = np.tile(global_mae, (len(knots), 1))
        shrink = max(32., len(spread) * .003)
        for i in range(len(knots)):
            group = regions == i
            if group.any():
                risk[i] = (errors[group].sum(axis=0, dtype=np.float64) + shrink*global_mae) / (group.sum()+shrink)
        cache[pool] = global_mae[None, :], knots, risk
    return cache[pool]


def fit_one(spec, g):
    """Fit absolute-risk weights with OOF training predictions and targets."""
    if spec.get("kind") != "mae_median":
        raise ValueError("Expected a median grid specification")
    global_risk, knots, conditional_risk = _risk_tables(g, spec["pool"])
    risk = conditional_risk if spec["risk"] == "spread" else global_risk
    rates = (1. + risk.min(axis=1, keepdims=True)) / (1. + risk)
    weights = rates**spec["power"]
    weights /= weights.sum(axis=1, keepdims=True)
    return {**spec, "knots": knots.copy() if spec["risk"] == "spread" else np.empty(0),
            "weights": weights.astype(np.float32)}


def _location(east, north, weight):
    # Work in the V4-anchored WGS84 azimuthal equidistant plane. Six
    # safeguarded Weiszfeld iterations approximate its weighted L1 optimum.
    # A majority mass makes that observed point an exact minimizer.
    w = np.broadcast_to(weight, east.shape)
    mean_e, mean_n = (w * east).sum(1), (w * north).sum(1)
    ye, yn = mean_e, mean_n
    for _ in range(6):
        se = east - ye[:, None]
        sn = north - yn[:, None]
        length = np.sqrt(se*se + sn*sn)
        nearby = np.maximum(length, 2e-8)
        updated = w / nearby
        z = updated.sum(1)
        ye = (updated*east).sum(1) / z
        yn = (updated*north).sum(1) / z
    idx = w.argmax(axis=1)
    rows = np.arange(len(east))
    fixed = w[rows, idx] >= .5
    return np.where(fixed, east[rows, idx], ye), np.where(fixed, north[rows, idx], yn), mean_e, mean_n


def predict_geometry(g, state):
    """Apply trained weights without labels and return unit geocentric ECEF."""
    if not len(g["anchor"]):
        return np.empty((0, 3), np.float32)
    east, north, spread, peer = _points(g, state["pool"])
    table = state["weights"]
    if table.shape[1] != east.shape[1]:
        raise ValueError("Median checkpoint uses a different expert count")
    if state["risk"] == "spread":
        weight = np.column_stack([np.interp(spread, state["knots"], col) for col in table.T]).astype(np.float32)
    elif state["risk"] == "agreement":
        weight = table * peer
        weight /= weight.sum(axis=1, keepdims=True)
    else:
        weight = table
    # Both readouts use the same fitted weights. A bounded one-entry cache
    # reuses the median iterations for the adjacent 'median'/'blend' pair.
    tag = (state["pool"], state["risk"], table.tobytes(), state["knots"].tobytes())
    previous = g.get("_mae_median_last")
    if previous is None or previous[0] != tag:
        previous = (tag, _location(east, north, weight))
        g["_mae_median_last"] = previous
    ye, yn, mean_e, mean_n = previous[1]
    if state["location"] == "blend":
        ye, yn = .8 * ye + .2 * mean_e, .8 * yn + .2 * mean_n
    lo, la = np.radians(_dest(lonlat(g["anchor"]), ye, yn))
    return unit(np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), (1-E2)*np.sin(la)]))
