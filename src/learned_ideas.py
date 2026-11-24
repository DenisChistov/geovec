"""Cross-fitted geographic stacks. All fitted quantities use training rows only."""

import json
from pathlib import Path

import numpy as np

import region_calibration as calibration


LEARNED = {
    "stack_ecef_residual": "Ridge regression of Cartesian residuals from the multiscale posterior mean.",
    "stack_wgs84_tangent": "Regress east/north ellipsoidal displacement and apply the correction by a WGS84 direct geodesic.",
    "stack_angular_risk": "Estimate each expert's excess squared-angular risk per unit of displacement and form a soft risk gate.",
    "stack_pair_tournament": "Learn the probabilities of pairwise expert wins and combine predictions by tournament score.",
    "stack_local_trust": "Learn the signed geographic displacement toward the retrieval-refined mean, constrained to their common arc.",
    "stack_disagreement_frame": "Regress two tangent components in a row-specific frame pointing toward the DNN/retrieval expert.",
    "stack_landmark_distances": "Regress angular-distance differences to 16 global landmarks and recover a local position by multilateration.",
    "stack_tail_mixture": "Fit separate typical-error and long-distance-error regressions, combined by a learned probability of an error above 1,500 km.",
    "stack_covariance_fusion": "Learn the expert error cross-products and minimize estimated squared error over regularized simplex weights.",
}
DESCRIPTIONS = {**LEARNED, **calibration.DESCRIPTIONS}
PRIOR = np.array([.08, .40, .14, .12, .13, .035, .095], dtype=np.float32)
I, J = np.triu_indices(7, k=1)
CI, CJ = np.triu_indices(7)
RADIUS_KM = 6371.0088
E2 = 1 - (1 - 1 / 298.257223563) ** 2
PHI = (1 + np.sqrt(5)) / 2
Z = (np.arange(16, dtype=np.float64) + .5) / 8 - 1
LON = np.arange(16) * 2 * np.pi / PHI
LANDMARKS = np.column_stack((np.sqrt(1 - Z**2) * np.cos(LON),
                             np.sqrt(1 - Z**2) * np.sin(LON), Z)).astype(np.float32)
FORMAT_VERSION = 1


def _norm(x):
    return np.linalg.norm(x, axis=-1, keepdims=True)


def _unit(x, name):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 2 or x.shape[-1] != 3:
        raise ValueError(f"{name} must have three XYZ components")
    sizes = _norm(x)
    if not np.isfinite(sizes).all() or (sizes < 1e-7).any():
        raise ValueError(f"{name} contains a nonfinite or zero direction")
    return x / sizes


def _inputs(features, experts):
    x = np.asarray(features, dtype=np.float32)
    e = _unit(experts, "experts")
    if x.ndim != 2 or x.shape[1] < 1 or e.shape != (len(x), 7, 3):
        raise ValueError("Expected [rows, features] and seven aligned [rows, 7, 3] expert vectors")
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")
    return x, e


def _project(v, a):
    return v - np.sum(v * a, axis=-1, keepdims=True) * a


def _east(a):
    horizontal = np.hypot(a[..., 0], a[..., 1])
    lon = np.arctan2(a[..., 1], a[..., 0])
    lon = np.where(horizontal > 1e-8, lon, 0)
    return np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)


def _log(a, v):
    # The tangent direction of an exactly antipodal point is indeterminate.
    # A deterministic eastward convention keeps the representation defined.
    dot = np.clip((a * v).sum(axis=-1, keepdims=True), -1, 1)
    p = v - dot * a
    size = _norm(p)
    direction = p / np.maximum(1e-7, size)
    direction = np.where((size < 1e-7) & (dot < 0), _east(a), direction)
    return direction * np.arctan2(size, dot)


def _exp(a, v):
    v = _project(v, a)
    size = _norm(v)
    return _unit(a * np.cos(size) + v * np.sinc(size / np.pi), "corrected vectors")


def _cap(v, limit):
    return v * np.minimum(1, limit / np.maximum(1e-7, _norm(v)))


def _geometry(e):
    a = e[:, 1]
    tangent = _log(a[:, None, :], e)
    tangent[:, 1] = 0
    sizes = _norm(tangent)[..., 0]
    return a, tangent, sizes


def _raw_design(x, t, sizes):
    # Angular diagnostics retain detail when expert dot products round to one.
    return np.column_stack((x, t.reshape(len(x), 21), sizes, sizes.max(1),
                            np.mean(sizes**2, axis=1))).astype(np.float32, copy=False)


def _design_setup(raw, seed):
    rng = np.random.default_rng(int(seed))
    rows = rng.choice(len(raw), min(24000, len(raw)), replace=False)
    sample = raw[rows]
    center = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(sample.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
    weight = rng.normal(0, raw.shape[1] ** -.5, (raw.shape[1], 160)).astype(np.float32)
    bias = rng.uniform(-.9, .9, 160).astype(np.float32)
    return dict(center=center, scale=std, random_weight=weight, random_bias=bias)


class _Design:
    """Evaluate the same fixed design on CPU and CUDA, in bounded batches."""

    def __init__(self, params, rows):
        self.torch = None
        if rows >= 16000:
            try:
                import torch
                if torch.cuda.is_available():
                    self.torch = torch
            except ImportError:
                pass
        self.center, self.std, self.weight, self.bias = [
            self.array(params[k]) for k in ("center", "scale", "random_weight", "random_bias")
        ]

    def array(self, value):
        if self.torch is None:
            return np.asarray(value, dtype=np.float32)
        return self.torch.as_tensor(value, dtype=self.torch.float32, device="cuda")

    def numpy(self, value):
        if self.torch is None:
            return np.asarray(value)
        return value.cpu().numpy()

    def __call__(self, raw):
        z = ((self.array(raw) - self.center) / self.std).clip(-6, 6)
        nonlinear = (z @ self.weight + self.bias).clip(0, 6)
        parts = (z[:, :1] * 0 + 1, z, nonlinear)
        if self.torch is None:
            return np.concatenate(parts, axis=1)
        return self.torch.cat(parts, dim=1)


def _lonlat(p):
    x, y, z = p.T
    return (np.rad2deg(np.arctan2(y, x)).astype(np.float64),
            np.rad2deg(np.arctan2(z, (1 - E2) * np.hypot(x, y))).astype(np.float64))


def _geodesic_target(a, y):
    from pyproj import Geod
    azimuth, _, meters = Geod(ellps="WGS84").inv(*_lonlat(a), *_lonlat(y))
    azimuth = np.deg2rad(azimuth)
    return (np.column_stack((np.sin(azimuth), np.cos(azimuth))) *
            (meters / (RADIUS_KM * 1000))[:, None]).astype(np.float32)


def _geodesic_apply(a, direction):
    from pyproj import Geod
    direction = _cap(.8 * direction, .50)
    azimuth = np.rad2deg(np.arctan2(direction[:, 0], direction[:, 1])).astype(np.float64)
    longitude, latitude, _ = Geod(ellps="WGS84").fwd(
        *_lonlat(a), azimuth, _norm(direction)[:, 0].astype(np.float64) * RADIUS_KM * 1000,
    )
    lon, lat = np.deg2rad(longitude), np.deg2rad(latitude)
    v = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                         (1 - E2) * np.sin(lat)))
    return _unit(v, "WGS84 correction")


def _frame(a, t, sizes):
    v = np.where(sizes[:, 6:7] > 1e-5, t[:, 6], _east(a))
    forward = _unit(v, "local frame")
    return forward, np.cross(a, forward)


def _targets(names, e, y, a, t, sizes):
    heads = {}
    loc = _log(a, y)
    tail = _norm(loc)[:, 0] > 1500 / RADIUS_KM
    if set(names).intersection(("stack_angular_risk", "stack_pair_tournament")):
        # atan2 using chord lengths stays accurate for both close and far points.
        error = 2 * np.arctan2(_norm(e - y[:, None])[:, :, 0],
                               _norm(e + y[:, None])[:, :, 0])
    for name in names:
        if name == "stack_ecef_residual":
            out = y - a
        elif name == "stack_wgs84_tangent":
            out = _geodesic_target(a, y)
        elif name == "stack_angular_risk":
            out = (error**2 - error[:, 1:2]**2) / (.025 + sizes)
        elif name == "stack_pair_tournament":
            out = .5 + .5 * np.sign(error[:, J] - error[:, I])
        elif name == "stack_local_trust":
            v = t[:, 4] / np.maximum(sizes[:, 4:5], 1e-6)
            out = (v * loc).sum(axis=1, keepdims=True)
        elif name == "stack_disagreement_frame":
            forward, side = _frame(a, t, sizes)
            out = np.column_stack(((forward * loc).sum(1), (side * loc).sum(1)))
        elif name == "stack_landmark_distances":
            out = np.arccos(np.clip(y @ LANDMARKS.T, -1, 1)) - np.arccos(
                np.clip(a @ LANDMARKS.T, -1, 1))
        elif name == "stack_tail_mixture":
            heads["body"] = loc
            heads["tail"] = loc
            out = tail[:, None]
        elif name == "stack_covariance_fusion":
            # Remove the common ||truth||^2 term. Under simplex weights it has
            # no effect on the optimizer and otherwise adds large label noise.
            dots = (t * loc[:, None]).sum(axis=2)
            out = (t[:, CI] * t[:, CJ]).sum(axis=2) - dots[:, CI] - dots[:, CJ]
        else:
            raise KeyError(name)
        heads[name] = np.asarray(out, dtype=np.float32)
    return heads, tail


def _train_linear(raw, params, heads, tail):
    all_names = [name for name in heads if name not in ("body", "tail")]
    widths = np.cumsum([heads[name].shape[1] for name in all_names])[:-1]
    targets = np.column_stack([heads[name] for name in all_names])
    groups = {"all": (targets, None)}
    if "body" in heads:
        groups.update(body=(heads["body"], ~tail), tail=(heads["tail"], tail))
    backend = _Design(params, len(raw))
    accum = {}
    for start in range(0, len(raw), 8192):
        sl = slice(start, start + 8192)
        phi = backend(raw[sl])
        for group, (label, mask) in groups.items():
            part = phi
            y = label[sl]
            if mask is not None:
                selected = mask[sl]
                if not selected.any():
                    continue
                ids = np.flatnonzero(selected)
                if backend.torch is not None:
                    ids = backend.torch.as_tensor(ids, device="cuda")
                part = phi[ids]
                y = y[selected]
            xy = part.T @ backend.array(y)
            xx = part.T @ part
            if group in accum:
                accum[group][0] += xx
                accum[group][1] += xy
            else:
                accum[group] = [xx, xy]
    learned = {}
    dimension = 1 + len(params["center"]) + len(params["random_bias"])
    for group, (target, mask) in groups.items():
        if group in accum:
            xx, xy = (backend.numpy(z).astype(np.float64) for z in accum[group])
            n = len(raw) if mask is None else int(mask.sum())
            penalty = max(2.0, n * (.008 if group == "all" else .023))
            xx.flat[::dimension + 1] += penalty
            xx[0, 0] -= .995 * penalty
            coef = np.linalg.solve(xx, xy).astype(np.float32)
        else:
            coef = np.zeros((dimension, target.shape[1]), dtype=np.float32)
        if group == "all":
            learned.update(zip(all_names, np.split(coef, widths, axis=1)))
        else:
            learned[group] = coef
    return learned


def _raw_predictions(raw, params, weights):
    names = list(weights)
    widths = np.cumsum([weights[name].shape[1] for name in names])[:-1]
    backend = _Design(params, len(raw))
    all_weights = backend.array(np.column_stack([weights[name] for name in names]))
    if not len(raw):
        return {name: np.empty((0, w.shape[1]), np.float32) for name, w in weights.items()}
    blocks = []
    for start in range(0, len(raw), 8192):
        blocks.append(backend.numpy(backend(raw[start:start + 8192]) @ all_weights))
    return dict(zip(names, np.split(np.concatenate(blocks), widths, axis=1)))


def _softweights(logits):
    z = logits + np.log(PRIOR)
    w = np.exp(np.maximum(z - z.max(1, keepdims=True), -40))
    return w / w.sum(1, keepdims=True)


def _landmark_step(a, delta):
    basis = np.stack((_east(a), np.cross(a, _east(a))), axis=1)
    dot = np.clip(a @ LANDMARKS.T, -1, 1)
    inv_sine = 1 / np.sqrt(np.maximum(1e-5, 1 - dot**2))
    # A damped local least-squares step toward the predicted landmark ranges.
    jac = -np.matmul(basis, LANDMARKS.T) * inv_sine[:, None]
    gram = jac @ np.swapaxes(jac, 1, 2)
    gram[:, 0, 0] += .7
    gram[:, 1, 1] += .7
    rhs = jac @ delta[..., None]
    step = np.linalg.solve(gram, rhs)[:, :, 0]
    return _cap(.8 * (step[:, :, None] * basis).sum(1), .42)


def _covariance_weights(moments, sizes):
    n = len(sizes)
    q = np.empty((n, 7, 7), np.float32)
    q[:, CI, CJ] = moments
    q[:, CJ, CI] = moments
    # Clip negative curvature on the simplex plane and shrink toward the prior.
    # The constant direction does not participate in simplex updates.
    mean = q.mean(axis=2, keepdims=True)
    centered = q - mean - mean.transpose(0, 2, 1) + q.mean(axis=(1, 2), keepdims=True)
    values, vectors = np.linalg.eigh(centered)
    psd = (vectors * values.clip(0)[:, None, :]) @ vectors.transpose(0, 2, 1)
    forcing = mean[:, :, 0] - mean[:, :, 0].mean(1, keepdims=True)
    strength = .0015 + .40 * np.mean(sizes**2, axis=1, keepdims=True)
    w = np.broadcast_to(PRIOR, (n, 7)).copy()
    # Frank-Wolfe with exact segment minimization for this quadratic objective.
    for _ in range(12):
        grad = (psd @ w[:, :, None])[:, :, 0] + forcing + strength * (w - PRIOR)
        vertex = np.argmin(grad, axis=1)
        step = -w.copy()
        step[np.arange(n), vertex] += 1
        curv = (step * (psd @ step[:, :, None])[:, :, 0]).sum(1) + strength[:, 0] * (step**2).sum(1)
        rate = np.clip(-(grad * step).sum(1) / np.maximum(1e-9, curv), 0, 1)
        w += rate[:, None] * step
    return w


def _decode(name, predictions, a, t, sizes):
    p = predictions[name]
    if not len(a):
        return np.empty((0, 3), dtype=np.float32)
    if name == "stack_ecef_residual":
        return _unit(a + _cap(.85 * p, .55), "Cartesian correction")
    if name == "stack_wgs84_tangent":
        return _geodesic_apply(a, p)
    if name == "stack_angular_risk":
        risk = p * (.025 + sizes)
        temp = .0015 + .20 * np.mean(sizes**2, axis=1, keepdims=True)
        weights = _softweights(-risk / temp)
        step = (weights[:, :, None] * t).sum(1)
    elif name == "stack_pair_tournament":
        wins = np.zeros((len(p), 7), np.float32)
        probs = np.clip(p, 0, 1)
        for k, (i, j) in enumerate(zip(I, J)):
            wins[:, i] += probs[:, k]
            wins[:, j] += 1 - probs[:, k]
        step = (_softweights(1.4 * wins)[:, :, None] * t).sum(1)
    elif name == "stack_local_trust":
        fraction = np.clip(p / np.maximum(.015, sizes[:, 4:5]), 0, 1)
        step = t[:, 4] * fraction
    elif name == "stack_disagreement_frame":
        f, side = _frame(a, t, sizes)
        step = _cap(.78 * (f * p[:, 0:1] + side * p[:, 1:2]), .45)
    elif name == "stack_landmark_distances":
        step = _landmark_step(a, p)
    elif name == "stack_tail_mixture":
        tail_rate = np.clip(p, 0, .95)
        step = _cap(.85 * ((1 - tail_rate) * predictions["body"] + tail_rate * predictions["tail"]), .55)
    elif name == "stack_covariance_fusion":
        w = _covariance_weights(p, sizes)
        step = (w[:, :, None] * t).sum(1)
    else:
        raise KeyError(name)
    return _exp(a, step)


def _save(root, name, feature_width, seed, **arrays):
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    metadata = dict(format_version=FORMAT_VERSION, name=name, seed=int(seed),
                    description=DESCRIPTIONS[name], feature_width=int(feature_width),
                    expert_order=["single_512", "multiscale", "sharper", "softer",
                                  "refined_mean", "refined_mode", "dnn_retrieval"])
    # Opening the temporary file avoids np.savez adding another extension.
    temp = path / f".{name}.tmp.npz"
    np.savez(temp, metadata=np.asarray(json.dumps(metadata)), **arrays)
    temp.replace(path / f"{name}.npz")


def _thread_limit():
    # Small normal-equation solves slow down under highly threaded BLAS.
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=4, user_api="blas")
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


def predict(features_train, y_train, features_eval, experts_train, experts_eval,
            seed, save_dir=None, only=None):
    """Fit the selected model(s) and predict. Seven experts follow v3's order."""
    if only is None:
        selected = list(DESCRIPTIONS)
    else:
        selected = [only] if isinstance(only, str) else list(dict.fromkeys(only))
    unknown = set(selected) - DESCRIPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown learned stack(s): {sorted(unknown)}")
    train_x, train_e = _inputs(features_train, experts_train)
    eval_x, eval_e = _inputs(features_eval, experts_eval)
    truth = _unit(y_train, "y_train")
    if truth.shape != (len(train_x), 3) or not len(train_x):
        raise ValueError("Training features, experts and nonempty targets must align")
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("Training and evaluation feature widths differ")
    results = {}
    with _thread_limit():
        calibrated = [name for name in selected if name in calibration.DESCRIPTIONS]
        if calibrated:
            models = calibration.fit_many(calibrated, train_e, truth)
            if len(eval_e):
                results.update(calibration.apply_many(models, eval_e))
            else:
                results.update({name: np.empty((0, 3), np.float32) for name in calibrated})
            if save_dir is not None:
                for name in calibrated:
                    _save(save_dir, name, train_x.shape[1], seed, **models[name])
        learned = [name for name in selected if name in LEARNED]
        if learned:
            a, tangent, sizes = _geometry(train_e)
            heads, tail = _targets(learned, train_e, truth, a, tangent, sizes)
            raw = _raw_design(train_x, tangent, sizes)
            del a, tangent, sizes
            design = _design_setup(raw, seed)
            weights = _train_linear(raw, design, heads, tail)
            del raw, heads, tail
            a, tangent, sizes = _geometry(eval_e)
            raw_eval = _raw_design(eval_x, tangent, sizes)
            out = _raw_predictions(raw_eval, design, weights)
            for name in learned:
                results[name] = _decode(name, out, a, tangent, sizes)
                if save_dir is not None:
                    parts = [name, "body", "tail"] if name == "stack_tail_mixture" else [name]
                    _save(save_dir, name, train_x.shape[1], seed, **design,
                          **{f"coef_{k}": weights[k] for k in parts})
    return {name: results[name] for name in selected}


def predict_saved(features_eval, experts_eval, path):
    """Apply a <method>.npz checkpoint without training rows or labels."""
    x, e = _inputs(features_eval, experts_eval)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {k: archive[k] for k in archive.files if k != "metadata"}
    name = metadata["name"]
    if metadata.get("format_version") != FORMAT_VERSION or name not in DESCRIPTIONS:
        raise ValueError("Unknown geographic stack checkpoint format or method")
    if metadata["feature_width"] != x.shape[1]:
        raise ValueError("Saved stack expects a different feature width")
    if not len(e):
        return np.empty((0, 3), np.float32)
    with _thread_limit():
        if name in calibration.DESCRIPTIONS:
            return calibration.apply(name, arrays, e)
        a, t, sizes = _geometry(e)
        raw = _raw_design(x, t, sizes)
        weights = {k.removeprefix("coef_"): v for k, v in arrays.items() if k.startswith("coef_")}
        out = _raw_predictions(raw, arrays, weights)
        return _decode(name, out, a, t, sizes)
