"""Robust fold-local ensembles for mean WGS84 distance.

The first forecast is the geographic origin. Input dictionaries come from
tune_rules.geometry; the origin may be replaced and the expert count may vary.
Each rule emits geocentric unit ECEF. Rank by the mean of exact WGS84 inverse
distances on held-out predictions. Training uses a smoothed vector L1 loss in
the origin's WGS84 azimuthal equidistant plane.
"""
from collections import defaultdict
from itertools import product

import numpy as np
from threadpoolctl import threadpool_limits

import mae_median
import tune_rules


def grid():
    specs = {}
    for pool, mode, context, alpha, smooth, gain in product(
        ("core", "all"), ("shared", "separate"), ("constant", "spread", "geo_spread"),
        (.015, .14), (8., 70.), (.70, 1.),
    ):
        name = f"mae:irls/{pool}/{mode}/{context}/a{alpha:g}/d{smooth:g}/gain{gain:g}"
        specs[name] = dict(kind="irls", pool=pool, mode=mode, context=context,
                           alpha=alpha, smooth_km=smooth, gain=gain)
    for name, spec in mae_median.grid().items():
        if name in specs:
            raise ValueError(f"Duplicate MAE configuration {name}")
        specs[name] = spec
    return specs


def _features(g, pool, kind, center=None, scale=None):
    d = np.asarray(g["d"], dtype=np.float32)
    if d.ndim != 3 or d.shape[1] != 2 or d.shape[2] < 1:
        raise ValueError("Expected WGS84 offsets [rows, 2, non-anchor experts]")
    if pool == "core":
        d = d[:, :, :3]
    n, _, m = d.shape
    if kind == "constant":
        c = np.ones((n, 1), np.float32)
        center = scale = np.empty(0, np.float32)
    else:
        radii = np.linalg.norm(d, axis=1)
        # A single available alternative has no inter-expert disagreement.
        pair = np.linalg.norm(d[:, :, 0] - d[:, :, 1], axis=1) if m > 1 else np.zeros(n, "f")
        spread = np.log1p(12 * np.column_stack((radii.mean(1), radii.max(1), pair)))
        if kind == "geo_spread":
            east_x, east_y, north_z = np.asarray(g["anchor"]).T
            raw = np.column_stack((spread, east_x, east_y, north_z)).astype("f")
        elif kind == "spread":
            raw = spread
        else:
            raise ValueError(f"Unknown MAE context {kind}")
        if center is None:
            center = raw.mean(axis=0, dtype=np.float64).astype("f")
            scale = np.maximum(raw.std(axis=0, dtype=np.float64), .025).astype("f")
        z = np.clip((raw - center) / scale, -4, 4)
        c = np.column_stack((np.ones(n, dtype="f"), z)).astype("f", copy=False)
    x = (d[..., None] * c[:, None, None, :]).reshape(n, 2, m * c.shape[1])
    return x, center, scale, c.shape[1]


def _prior(width, n_context):
    p = np.zeros(width, dtype=np.float64)
    # A weak anchor-centered correction is independent of expert identities.
    p[::n_context] = .25 / (width // n_context)
    return p


def _penalty(diagonal, alpha):
    return alpha * np.maximum(diagonal, max(float(diagonal.mean()) * .04, 1e-7))


class _IRLS:
    """Reuse one feature design for every regularizer, robust scale and gain."""
    def __init__(self, x, y, n_context):
        import torch
        self.torch = torch
        self.gpu = torch.cuda.is_available()
        self.n_context = n_context
        axes = np.ascontiguousarray(np.moveaxis(x, 1, 0))
        targets = np.ascontiguousarray(np.asarray(y, dtype="f").T)
        self.x = torch.as_tensor(axes, device="cuda") if self.gpu else axes
        self.y = torch.as_tensor(targets, device="cuda") if self.gpu else targets

    def fit(self, spec):
        delta = spec["smooth_km"] * 1000 / tune_rules.R
        prior = _prior(self.x.shape[2], self.n_context)
        if self.gpu:
            return self._fit_torch(spec, prior, delta)
        return self._fit_cpu(spec, prior, delta)

    def _fit_cpu(self, spec, prior, delta):
        x, y = self.x, self.y
        coef = np.tile(prior.astype("f"), (2, 1))
        penalty = None
        shared = spec["mode"] == "shared"
        for _ in range(6):
            err = np.einsum("cnp,cp->cn", x, coef, optimize=False) - y
            # MM weights for sqrt(||error||^2 + delta^2); the same weight
            # applies to east and north even when their coefficients differ.
            w = delta / np.sqrt(np.sum(err * err, axis=0) + delta**2)
            wx = x * w[None, :, None]
            gram = x.swapaxes(1, 2) @ wx
            moment = (wx.swapaxes(1, 2) @ y[..., None])[..., 0]
            if shared:
                gram, moment = gram.sum(0, keepdims=True), moment.sum(0, keepdims=True)
            gram, moment = gram.astype("d"), moment.astype("d")
            if penalty is None:
                penalty = np.stack([_penalty(g.diagonal(), spec["alpha"]) for g in gram])
            ids = np.arange(len(prior))
            gram[:, ids, ids] += penalty
            rhs = (moment + penalty * prior)[..., None]
            updated = np.linalg.solve(gram, rhs)[..., 0].astype("f")
            coef = np.repeat(updated, 2, axis=0) if shared else updated
        return np.ascontiguousarray(coef)

    def _fit_torch(self, spec, prior_np, delta):
        torch = self.torch
        x, y = self.x, self.y
        with torch.inference_mode():
            prior = torch.as_tensor(prior_np, device=x.device)
            coef = prior.float().expand(2, -1)
            penalty = None
            shared = spec["mode"] == "shared"
            for _ in range(6):
                err = torch.bmm(x, coef.unsqueeze(-1)).squeeze(-1) - y
                w = delta * torch.rsqrt((err * err).sum(0) + delta**2)
                wx = x * w[None, :, None]
                gram = torch.bmm(x.transpose(1, 2), wx)
                moment = torch.bmm(wx.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)
                if shared:
                    gram, moment = gram.sum(0, keepdim=True), moment.sum(0, keepdim=True)
                gram, moment = gram.double(), moment.double()
                if penalty is None:
                    diagonal = gram.diagonal(dim1=1, dim2=2)
                    floors = (.04 * diagonal.mean(1, keepdim=True)).clamp_min(1e-7)
                    penalty = spec["alpha"] * torch.maximum(diagonal, floors)
                gram = gram + torch.diag_embed(penalty)
                rhs = (moment + penalty * prior).unsqueeze(-1)
                coef = torch.linalg.solve(gram, rhs).squeeze(-1).float()
                if shared:
                    coef = coef.expand(2, -1)
            return coef.contiguous().cpu().numpy()


def _is_median(spec):
    return spec["kind"] != "irls"


def _forward(g, step):
    lon, lat = tune_rules.lonlat(g["anchor"])
    bearing = np.degrees(np.arctan2(step[:, 0], step[:, 1]))
    meters = np.minimum(np.linalg.norm(step, axis=1), np.pi) * tune_rules.R
    lo, la, _ = tune_rules.GEOD.fwd(lon, lat, bearing, meters)
    lo, la = np.radians(lo), np.radians(la)
    return tune_rules.unit(np.column_stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo),
                                           (1 - tune_rules.E2)*np.sin(la))))


def fit_one(spec, train):
    """Fit on training geometry only; state consists of scalars, strings and NumPy arrays."""
    if _is_median(spec):
        return mae_median.fit_one(spec, train)
    x, center, scale, n_context = _features(train, spec["pool"], spec["context"])
    with threadpool_limits(limits=3, user_api="blas"):
        coef = _IRLS(x, train["target"], n_context).fit(spec)
    return {**spec, "center": center, "scale": scale, "coef": coef}


def predict_geometry(g, state):
    if not len(g["anchor"]):
        return np.empty((0, 3), "f")
    if _is_median(state):
        return mae_median.predict_geometry(g, state)
    x, _, _, _ = _features(g, state["pool"], state["context"], state["center"], state["scale"])
    step = np.einsum("ncp,cp->nc", x, state["coef"], optimize=False)
    return _forward(g, step * state["gain"])


def grouped_predictions(train, query):
    """Fit once per IRLS design/scale/regularizer and share coefficients across gains."""
    groups = defaultdict(lambda: defaultdict(list))
    medians = []
    for name, spec in grid().items():
        if _is_median(spec):
            medians.append((name, spec))
        else:
            groups[spec["pool"], spec["context"]][spec["mode"], spec["alpha"], spec["smooth_km"]].append((name, spec))
    for (pool, kind), settings in groups.items():
        x, center, scale, n_context = _features(train, pool, kind)
        model = _IRLS(x, train["target"], n_context)
        del x
        q, _, _, _ = _features(query, pool, kind, center, scale)
        for pairs in settings.values():
            with threadpool_limits(limits=3, user_api="blas"):
                coef = model.fit(pairs[0][1])
            step = np.einsum("ncp,cp->nc", q, coef, optimize=False)
            for name, spec in pairs:
                yield name, _forward(query, step * spec["gain"]) if len(step) else np.empty((0,3), "f")
        del model
    if hasattr(mae_median, "grouped_predictions"):
        yield from mae_median.grouped_predictions(train, query)
    else:
        for name, spec in medians:
            state = mae_median.fit_one(spec, train)
            yield name, mae_median.predict_geometry(query, state)
