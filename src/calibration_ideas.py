"""Cross-fitted calibrators for the V3 posterior and retrieval experts.

Targets and predictions use unit geocentric ECEF, as in experiment.decode.
Each method models offsets from the multiscale expert in its WGS84 azimuthal
equidistant plane. The inverse map follows a WGS84 geodesic.
"""
from pathlib import Path

import numpy as np
from pyproj import Geod
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


DESCRIPTIONS = {
    "cal_atlas_prior": "Learn a geographic label-density prior and calibrate motion along its gradients at two uncertainty-dependent scales.",
    "cal_signed": "Fit a ridge-regularized, signed combination of expert offsets with extrapolation.",
    "cal_compass": "Fit separate east/north coefficients, cross-axis terms and a global bias.",
    "cal_rotation": "Estimate a proper 3D rotation of the signed stack by iteratively reweighted spherical Procrustes.",
    "cal_robust": "Fit an uncertainty-adaptive Huber stack with bounded influence for extreme training misses.",
    "cal_harmonics": "Correct a signed stack with a smooth degree-two geographic bias field conditioned on uncertainty.",
    "cal_local_bias": "Estimate a spatial bias from training forecast neighbors of similar uncertainty.",
    "cal_posterior_rank": "Learn directional corrections for probability-weighted posterior ranks at two cell resolutions.",
    "cal_interaction": "Regress expert directions on continuous posterior, retrieval and disagreement indicators.",
    "cal_kernel": "Learn a random Fourier kernel ridge correction from geographic, uncertainty and visual-PCA features.",
}

_GEOD = Geod(ellps="WGS84")
_E2 = 1 - (1 - 1 / 298.257223563) ** 2
_RADIUS = 6371008.8
_EXPERTS = (0, 2, 3, 4, 5, 6)
_POST = (21, 74, 127)  # Each V3 posterior holds 10 * (p, x, y, z) + 13 summaries.
_REF = 180
_BASE_FEATURES = 246


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm < 1e-7):
        raise ValueError("Directions must be finite and nonzero")
    return a / norm


def _lonlat(a):
    x, y, z = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arctan2(z, (1 - _E2) * np.hypot(x, y)))


def _xyz(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return _unit(np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                           (1 - _E2) * np.sin(lat)), axis=-1))


def _log(origin, points):
    lon, lat = _lonlat(points)
    bearing, _, meters = _GEOD.inv(origin[0], origin[1], lon, lat)
    az = np.radians(bearing)
    return (np.column_stack((np.sin(az), np.cos(az))) * (meters / _RADIUS)[:, None]).astype("f")


def _exp(origin, offset):
    length = np.linalg.norm(offset, axis=-1)
    bearing = np.degrees(np.arctan2(offset[:, 0], offset[:, 1]))
    lon, lat, _ = _GEOD.fwd(origin[0], origin[1], bearing, _RADIUS * np.minimum(length, np.pi))
    return _xyz(lon, lat)


class _Inputs:
    def __init__(self, features, experts):
        self.f = np.asarray(features, dtype=np.float32)
        e = _unit(experts)
        if e.ndim != 3 or e.shape[1:] != (7, 3) or self.f.ndim != 2 or len(e) != len(self.f):
            raise ValueError("Expected row-aligned V3 features and seven XYZ experts")
        if self.f.shape[1] < _BASE_FEATURES or not np.isfinite(self.f).all():
            raise ValueError("Expected the original 246 V3 columns, optionally followed by PCA")
        self.anchor = e[:, 1, :]
        self.origin = _lonlat(self.anchor)
        self.d = np.stack([_log(self.origin, e[:, j]) for j in _EXPERTS], axis=-1)
        self.spread = np.linalg.norm(self.d, axis=1).max(axis=1)
        entropy = np.mean([self.f[:, o + 41] for o in _POST], axis=0)
        confidence = np.mean([self.f[:, o + 42] for o in _POST], axis=0)
        rms = np.mean([self.f[:, o + 49] for o in _POST], axis=0)
        self.u = np.column_stack((entropy, 1 - confidence, rms, self.spread,
                                  1 - self.f[:, _REF + 5], self.f[:, _REF + 18],
                                  self.f[:, _REF + 7])).astype("f")


def _solve(g, b, penalty, prior=None):
    g = np.asarray(g, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    g.flat[::len(g) + 1] += penalty
    if prior is not None:
        b += penalty * prior
    return np.linalg.solve(g, b).astype("f")


def _stats(x, y):
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(len(x), -1)
    return x.T @ x, (x.T @ y)


def _ridge(x, y, strength):
    g, b = _stats(x, y)
    penalty = strength * np.maximum(g.diagonal(), len(x) * .0003)
    return _solve(g, b, penalty)


def _apply(x, w):
    return np.einsum("ntk,k->nt", x, w, optimize=False)


def _normalized_covariates(x, model, fitting):
    if fitting:
        model["center"] = x.mean(axis=0, dtype=np.float64).astype("f")
        model["scale"] = np.maximum(x.std(axis=0, dtype=np.float64), .015).astype("f")
    return np.clip((x - model["center"]) / model["scale"], -4, 4)


def _sphere_grid(count=128):
    i = np.arange(count)
    z = (2 * i + 1) / count - 1
    t = i * (np.pi * (3 - np.sqrt(5)))
    r = np.sqrt(1 - z * z)
    return np.column_stack((r * np.cos(t), r * np.sin(t), z)).astype("f")


def _fit_atlas(truth, model):
    grid = _sphere_grid(384)
    ids = cKDTree(grid).query(truth)[1]
    counts = np.bincount(ids, minlength=len(grid))
    sums = np.column_stack([np.bincount(ids, weights=truth[:, j], minlength=len(grid))
                            for j in range(3)])
    model["sites"] = _unit(sums[counts > 0])
    model["log_mass"] = np.log(counts[counts > 0]).astype("f")


def _atlas_vectors(t, model):
    sites, mass = model["sites"], model["log_mass"]
    basis = _basis(t)
    output = np.empty((len(t.d), 2, 2), dtype="f")
    for start in range(0, len(output), 2048):
        s = slice(start, start + 2048)
        similarity = t.anchor[s] @ sites.T
        uncertainty = np.clip(t.u[s, 2:3], .01, 1.1)
        for j, width in enumerate((.05 + .3 * uncertainty, .13 + .75 * uncertainty)):
            logits = (similarity - 1) / width**2 + mass
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            barycenter = (weights @ sites) / weights.sum(axis=1, keepdims=True)
            radial = np.einsum("nd,nd->n", barycenter, t.anchor[s])
            step = barycenter - radial[:, None] * t.anchor[s]
            output[s, :, j] = np.einsum("nd,ntd->nt", step, basis[s])
    return output


def _fit_rotation(source, truth):
    weight = np.ones(len(source))
    for _ in range(4):
        matrix = source.T @ (truth * weight[:, None]) + np.eye(3) * (.001 * len(source))
        left, _, right = np.linalg.svd(matrix)
        left[:, -1] *= np.sign(np.linalg.det(left @ right))
        rotation = left @ right
        rotated = source @ rotation
        angle = np.arctan2(np.linalg.norm(np.cross(rotated, truth), axis=1),
                           np.einsum("ij,ij->i", rotated, truth))
        weight = np.clip(1 / np.maximum(np.sinc(angle / np.pi), 1e-4), 1, 8)
    return rotation.astype("f")


def _compass(data):
    return np.column_stack((data.d[:, 0], data.d[:, 1], np.ones(len(data.d), dtype="f")))


def _harmonics(data):
    x, y, z = data.anchor.T
    basis = np.column_stack((np.ones_like(x), x, y, z, x*y, x*z, y*z, x*x-y*y, 3*z*z-1))
    # A nearly certain posterior can still have a small, persistent geographic bias.
    uncertainty = np.clip(data.u[:, 2], .015, .9)[:, None]
    return np.column_stack((basis, basis * uncertainty)).astype("f")


def _neighbor_keys(data):
    return np.column_stack((data.anchor, .18 * np.log1p(8 * np.maximum(data.u[:, 2], 0)),
                            .07 * np.log1p(12 * data.spread))).astype("f")


def _basis(data):
    lon, lat = np.radians(data.origin)
    east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)))
    north = np.column_stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)))
    return np.stack((east, north), axis=1).astype("f")


def _local(data, model):
    tree = cKDTree(model["keys"])
    keys = _neighbor_keys(data)
    basis = _basis(data)
    out = np.empty((len(keys), 2), dtype="f")
    k = min(96, len(model["keys"]))
    for start in range(0, len(out), 2048):
        part = slice(start, start + 2048)
        distances, ids = tree.query(keys[part], k=list(range(1, k + 1)), workers=2)
        weights = np.exp(-1.5 * (distances / np.maximum(distances[:, -1:], 1e-6))**2)
        # Shrink distant and sparsely supported queries toward the global stack.
        mass = weights.sum(axis=1) + 20 + 80 * np.minimum(distances[:, -1], 1)
        bias = np.einsum("ij,ijk->ik", weights, model["errors"][ids]) / mass[:, None]
        out[part] = np.einsum("nik,nk->ni", basis[part], bias)
    return out


def _posterior_design(data):
    # The posterior cells store geocentric directions. Their probability-weighted
    # spherical log directions are covariates; all response offsets are WGS84.
    frame = _basis(data)
    rank_sets = ((0, 1), (1, 2), (2, 4), (4, 7), (7, 10))
    output = []
    for family in ((_POST[0], _POST[2]), (_POST[1],)):
        by_rank = np.zeros((len(data.f), len(rank_sets), 2), dtype="f")
        for start in range(0, len(data.f), 8192):
            s = slice(start, start + 8192)
            for offset in family:
                cells = data.f[s, offset:offset + 40].reshape(-1, 10, 4)
                p, xyz = cells[..., 0], cells[..., 1:]
                dot = np.einsum("nrc,nc->nr", xyz, data.anchor[s])
                tangent = xyz - dot[:, :, None] * data.anchor[s, None]
                norm = np.linalg.norm(tangent, axis=2)
                factor = np.arctan2(norm, dot) / np.maximum(norm, 1e-7)
                directions = np.einsum("nrc,ntc->nrt", tangent, frame[s])
                directions *= (p * factor)[:, :, None]
                for j, (lo, hi) in enumerate(rank_sets):
                    by_rank[s, j] += directions[:, lo:hi].sum(axis=1) / len(family)
        output.append(by_rank.transpose(0, 2, 1))
    return np.concatenate((data.d, *output), axis=2)


def _interaction(data, model, fitting):
    u = _normalized_covariates(data.u, model, fitting)
    u = np.column_stack((np.ones(len(u), dtype="f"), u))
    return (data.d[..., None] * u[:, None, None, :]).reshape(len(u), 2, -1)


def _kernel(data, model, fitting, seed=0):
    raw = np.column_stack((data.u, data.f[:, _BASE_FEATURES:_BASE_FEATURES + 16]))
    z = _normalized_covariates(raw, model, fitting)
    context = np.column_stack((2 * data.anchor, z[:, :7] / np.sqrt(7),
                               z[:, 7:] * (.65 / np.sqrt(max(1, z.shape[1] - 7))))).astype("f")
    if fitting:
        rng = np.random.default_rng(seed)
        widths = np.repeat(np.array([.65, 1.6], dtype="f"), 80)
        model["omega"] = rng.normal(size=(context.shape[1], len(widths))).astype("f") * widths
        model["phase"] = rng.uniform(0, 2 * np.pi, size=len(widths)).astype("f")
    return np.cos(context @ model["omega"] + model["phase"])


def _fit(name, t, y, truth, w, seed):
    model = {"name": np.array(name), "features": np.int32(t.f.shape[1]), "weights": w}
    if name == "cal_atlas_prior":
        _fit_atlas(truth, model)
        model["coef"] = _ridge(_atlas_vectors(t, model), y - _apply(t.d, w), .05)[:, 0]
    elif name == "cal_compass":
        model["coef"] = _ridge(_compass(t), y, .025)
    elif name == "cal_rotation":
        model["rotation"] = _fit_rotation(_exp(t.origin, _apply(t.d, w)), truth)
    elif name == "cal_robust":
        threshold = np.clip(.12 + 2.5 * t.u[:, 2] + t.spread, .12, np.pi)
        robust = w
        for _ in range(4):
            error = np.linalg.norm(y - _apply(t.d, robust), axis=1)
            scale = np.sqrt(np.clip(threshold / np.maximum(error, 1e-6), .15, 1))
            robust = _ridge(t.d * scale[:, None, None], y * scale[:, None], .008)[:, 0]
        model["weights"] = robust
    elif name == "cal_harmonics":
        model["coef"] = _ridge(_harmonics(t), y - _apply(t.d, w), .035)
    elif name == "cal_local_bias":
        model["keys"] = _neighbor_keys(t)
        model["errors"] = np.einsum("nd,ndc->nc", y - _apply(t.d, w), _basis(t))
    elif name == "cal_posterior_rank":
        model["coef"] = _ridge(_posterior_design(t), y, .012)[:, 0]
    elif name == "cal_interaction":
        model["coef"] = _ridge(_interaction(t, model, True), y, .024)[:, 0]
    elif name == "cal_kernel":
        model["coef"] = _ridge(_kernel(t, model, True, seed), y - _apply(t.d, w), .05)
    return model


def _predict(t, model):
    name = str(model["name"])
    if int(model["features"]) != t.f.shape[1]:
        raise ValueError("Feature count differs from the fitted calibration")
    base = _apply(t.d, model["weights"])
    if name == "cal_atlas_prior":
        offset = base + _apply(_atlas_vectors(t, model), model["coef"])
    elif name == "cal_compass":
        offset = _compass(t) @ model["coef"]
    elif name == "cal_rotation":
        return _unit(_exp(t.origin, base) @ model["rotation"])
    elif name == "cal_harmonics":
        offset = base + _harmonics(t) @ model["coef"]
    elif name == "cal_local_bias":
        offset = base + _local(t, model)
    elif name == "cal_posterior_rank":
        offset = _apply(_posterior_design(t), model["coef"])
    elif name == "cal_interaction":
        offset = _apply(_interaction(t, model, False), model["coef"])
    elif name == "cal_kernel":
        offset = base + _kernel(t, model, False) @ model["coef"]
    elif name in ("cal_robust", "cal_signed"):
        offset = base
    else:
        raise ValueError(f"Unknown calibration {name}")
    return _exp(t.origin, offset)


def predict(features_train, y_train, features_eval, experts_train, experts_eval,
            seed, save_dir=None, only=None):
    """Fit all candidates, or a named subset, on inner out-of-fold examples."""
    names = list(DESCRIPTIONS) if only is None else [only] if isinstance(only, str) else list(only)
    if not names or any(name not in DESCRIPTIONS for name in names):
        raise ValueError(f"Invalid calibration selection {only!r}")
    t = _Inputs(features_train, experts_train)
    q = _Inputs(features_eval, experts_eval)
    truth = _unit(y_train)
    if truth.shape != t.anchor.shape or not len(truth):
        raise ValueError("Training truth must contain one ECEF direction per training row")
    y = _log(t.origin, truth)
    predictions = {}
    with threadpool_limits(limits=3, user_api="blas"):
        g, targets = _stats(t.d, y)
        b = targets[:, 0]
        w = _solve(g, b, .004 * np.maximum(g.diagonal(), len(t.d) * .0003))
        for name in names:
            model = _fit(name, t, y, truth, w, int(seed))
            predictions[name] = _predict(q, model)
            if save_dir is not None:
                directory = Path(save_dir)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / (name + ".npz")
                tmp = directory / (name + ".tmp.npz")
                np.savez(tmp, **model)
                tmp.replace(path)
    return predictions


def predict_saved(features_eval, experts_eval, path):
    """Apply a previously fitted <method>.npz checkpoint."""
    with np.load(path, allow_pickle=False) as archive, threadpool_limits(limits=3, user_api="blas"):
        return _predict(_Inputs(features_eval, experts_eval), dict(archive))
