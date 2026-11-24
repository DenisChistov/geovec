"""GPU CatBoost fusion of cross-fitted geography experts.

Every training feature, including the expert predictions, must come from an
inner holdout. The caller owns that split and the completely separate outer
validation and final test.
"""

import json

import numpy as np
from catboost import CatBoostRegressor
from pyproj import Geod

EARTH_KM = 6371.0088
WGS84 = Geod(ellps="WGS84")
E2 = 2 / 298.257223563 - (1 / 298.257223563) ** 2
CONFIGS = {
    "risk_gate": {
        "mode": "risk_gate",
        "depth": 5,
        "iterations": 360,
        "learning_rate": 0.05,
        "shrink": 0.12,
    },
    "oracle_gate": {
        "mode": "oracle_gate",
        "depth": 6,
        "iterations": 360,
        "learning_rate": 0.05,
        "shrink": 0.12,
    },
    "tangent_residual": {
        "mode": "tangent_residual",
        "depth": 6,
        "iterations": 400,
        "learning_rate": 0.05,
        "gain": 0.65,
        "max_step_km": 1700,
    },
}


def _normalized(vectors, name):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1] != 3 or not np.isfinite(vectors).all():
        raise ValueError(f"{name} must contain finite XYZ directions")
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if (length < 1e-10).any():
        raise ValueError(f"{name} contains zero directions")
    return vectors / length


def _coordinates(xyz):
    """Convert unit WGS84 ECEF directions to geodetic degrees."""
    x, y, z = xyz.T
    return (
        np.rad2deg(np.arctan2(y, x)),
        np.rad2deg(np.arctan2(z, (1 - E2) * np.hypot(x, y))),
    )


def _distance(xyz, truth_lon, truth_lat):
    lon, lat = _coordinates(xyz)
    return WGS84.inv(lon, lat, truth_lon, truth_lat)[2] * 0.001


def _spherical_log(anchor, vectors):
    dot = np.clip((vectors * anchor).sum(axis=-1, keepdims=True), -1, 1)
    tangent = vectors - dot * anchor
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    angle = np.arctan2(length, dot)
    direction = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 1e-10)
    return angle * direction


def _spherical_exp(anchor, tangent):
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    direction = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 1e-10)
    return _normalized(np.cos(length) * anchor + np.sin(length) * direction, "exp map")


def _blend(experts, weights, steps=3):
    """Weighted spherical mean with a stable strongest-expert fallback."""
    p = np.sum(experts * weights[..., None], axis=1)
    length = np.linalg.norm(p, axis=-1)
    fallback = experts[np.arange(len(experts)), np.argmax(weights, axis=1)]
    p[length < 1e-8] = fallback[length < 1e-8]
    p = _normalized(p, "weighted mean")
    # Minimize an intrinsic squared-angular loss. Damped updates avoid
    # overshoots for experts predicting points on opposite sides of Earth.
    for _ in range(steps):
        shift = np.sum(_spherical_log(p[:, None, :], experts) * weights[..., None], axis=1)
        step_length = np.linalg.norm(shift, axis=1, keepdims=True)
        shift = shift * np.minimum(0.75, 0.65 / np.maximum(step_length, 1e-12))
        p = _spherical_exp(p, shift)
    return p


def _risk_weights(distances):
    """Smoothed target weights derived from WGS84 squared-distance regrets."""
    loss = distances ** 2
    regret = loss - loss.min(axis=1, keepdims=True)
    # A relative temperature allows softly blending similarly accurate
    # experts, including when all the candidates miss the true location.
    temperature = 350.0 ** 2 + 0.20 * loss.min(axis=1, keepdims=True)
    soft = np.exp(-np.minimum(regret / temperature, 80.0))
    return soft / soft.sum(axis=1, keepdims=True)


def _oracle_weights(experts, truth, truth_lon, truth_lat, distances):
    """Find an oracle single expert or two-expert great-circle blend.

    The best position on each pair's short spherical arc follows analytically
    from its dot product with truth. Candidate distances are then evaluated
    on WGS84. These oracle weights are training targets only.
    """
    rows, count, _ = experts.shape
    best = distances.min(axis=1).copy()
    target = np.eye(count, dtype=np.float64)[np.argmin(distances, axis=1)]
    row_ids = np.arange(rows)
    for i in range(count):
        p = experts[:, i]
        a = (p * truth).sum(axis=1)
        for j in range(i + 1, count):
            q = experts[:, j]
            cosine = np.clip((p * q).sum(axis=1), -1, 1)
            tangent = q - cosine[:, None] * p
            sine = np.linalg.norm(tangent, axis=1)
            theta = np.arctan2(sine, cosine)
            axis = tangent / np.maximum(sine[:, None], 1e-12)
            phi = np.arctan2((axis * truth).sum(axis=1), a)
            # Endpoints were already considered; skip rows with no interior
            # optimum and numerically ambiguous coincident/antipodal pairs.
            valid = (sine > 1e-8) & (phi > 0) & (phi < theta)
            ids = row_ids[valid]
            if not len(ids):
                continue
            pos = np.cos(phi[ids, None]) * p[ids] + np.sin(phi[ids, None]) * axis[ids]
            d = _distance(pos, truth_lon[ids], truth_lat[ids])
            better = d < best[ids]
            ids = ids[better]
            best[ids] = d[better]
            t = phi[ids] / theta[ids]
            target[ids] = 0
            target[ids, i] = 1 - t
            target[ids, j] = t
    # A modest amount of risk-based smoothing prevents arbitrarily assigning
    # nearly identical expert predictions to only one model.
    return 0.85 * target + 0.15 * _risk_weights(distances)


def _simplex_prediction(raw, shrink):
    raw = np.maximum(raw, 0.0)
    total = raw.sum(axis=1, keepdims=True)
    uniform = np.full_like(raw, 1 / raw.shape[1])
    weight = np.divide(raw, total, out=uniform.copy(), where=total > 1e-10)
    return (1 - shrink) * weight + shrink * uniform


def _anchors(experts, weights):
    weights = np.broadcast_to(weights[None, :], experts.shape[:2])
    return _blend(experts, weights)


def _fuse(model, features, experts, mode, shrink, gain, max_step, anchor_weights):
    prediction = np.asarray(model.predict(features), dtype=np.float64)
    if mode == "tangent_residual":
        # Project the learned ambient vector onto the new anchor's tangent
        # plane, clip its angular magnitude, and return to the sphere.
        anchor = _anchors(experts, anchor_weights)
        tangent = gain * (prediction - (prediction * anchor).sum(axis=-1, keepdims=True) * anchor)
        length = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent *= np.minimum(1.0, max_step / np.maximum(length, 1e-12))
        fused = _spherical_exp(anchor, tangent)
    elif mode in {"risk_gate", "oracle_gate"}:
        fused = _blend(experts, _simplex_prediction(prediction, shrink))
    else:
        raise ValueError(f"Unknown fusion mode {mode!r}")
    return _normalized(fused, "fusion prediction").astype(np.float32)


def predict_saved(features, experts, path):
    """Predict with a saved fusion CBM and its embedded inference settings.

    Features and expert directions must have the same column/order convention
    as during fitting. No training labels or GPU are needed for the CatBoost
    inference step.
    """
    x = np.asarray(features, dtype=np.float32)
    experts = _normalized(experts, "evaluation experts")
    if x.ndim != 2 or experts.ndim != 3 or len(x) != len(experts):
        raise ValueError("Misaligned fusion features or experts")
    if np.isinf(x).any():
        raise ValueError("Infinite fusion features")

    model = CatBoostRegressor()
    model.load_model(str(path))
    metadata = model.get_metadata()
    mode = metadata["geocv_mode"]
    shrink = float(metadata["geocv_shrink"])
    gain = float(metadata["geocv_gain"])
    max_step = float(metadata["geocv_max_step_radians"])
    anchor_weights = np.asarray(json.loads(metadata["geocv_anchor_weights"]), dtype=np.float64)
    if (
        anchor_weights.shape != (experts.shape[1],)
        or experts.shape[1] < 2
        or len(model.feature_names_) != x.shape[1]
    ):
        raise ValueError("Saved fusion model expects different features or expert count")
    if not len(x):
        return np.empty((0, 3), dtype=np.float32)
    return _fuse(model, x, experts, mode, shrink, gain, max_step, anchor_weights)


def fit_predict(
    features_train,
    y_train,
    features_eval,
    experts_train,
    experts_eval,
    config,
    seed,
    save_path=None,
):
    """Fit one fusion candidate and return finite float32 unit XYZ directions."""
    x = np.asarray(features_train, dtype=np.float32)
    x_eval = np.asarray(features_eval, dtype=np.float32)
    train_experts = _normalized(experts_train, "training experts")
    eval_experts = _normalized(experts_eval, "evaluation experts")
    truth = _normalized(y_train, "training labels")
    if x.ndim != 2 or x_eval.ndim != 2 or x.shape[1] != x_eval.shape[1]:
        raise ValueError("Training and evaluation features must have identical columns")
    if np.isinf(x).any() or np.isinf(x_eval).any():
        raise ValueError("Infinite fusion features")
    if train_experts.ndim != 3 or eval_experts.ndim != 3:
        raise ValueError("Experts must have shape [rows, experts, 3]")
    count = train_experts.shape[1]
    if (
        count < 2
        or eval_experts.shape[1] != count
        or len(train_experts) != len(x)
        or len(eval_experts) != len(x_eval)
        or truth.shape != (len(x), 3)
    ):
        raise ValueError("Misaligned fusion experts or targets")

    options = dict(config)
    mode = options.pop("mode")
    shrink = float(options.pop("shrink", 0.12))
    gain = float(options.pop("gain", 0.65))
    max_step = float(options.pop("max_step_km", 1700)) / EARTH_KM
    anchor_weights = np.asarray(options.pop("anchor_weights", np.ones(count)), dtype=np.float64)
    if (
        anchor_weights.shape != (count,)
        or not np.isfinite(anchor_weights).all()
        or (anchor_weights < 0).any()
        or anchor_weights.sum() <= 0
    ):
        raise ValueError("anchor_weights must be nonnegative and match expert count")
    anchor_weights /= anchor_weights.sum()
    if not 0 <= shrink <= 1 or not 0 < gain <= 1 or not 0 < max_step < np.pi / 2:
        raise ValueError("Invalid shrink, gain, or maximum tangent step")

    sample_weight = None
    if mode in {"risk_gate", "oracle_gate"}:
        lon, lat = _coordinates(truth)
        distances = np.column_stack([_distance(train_experts[:, i], lon, lat) for i in range(count)])
        if mode == "risk_gate":
            target = _risk_weights(distances)
        else:
            target = _oracle_weights(train_experts, truth, lon, lat, distances)
        # Variation in potential squared error makes the gate more consequential.
        spread = np.sqrt(np.maximum(distances.max(axis=1) ** 2 - distances.min(axis=1) ** 2, 0))
        sample_weight = 0.25 + np.clip(spread / 2000.0, 0, 4.0)
    elif mode == "tangent_residual":
        anchor_train = _anchors(train_experts, anchor_weights)
        target = _spherical_log(anchor_train, truth)
        # Shrink extreme training misses that are beyond a useful local step.
        length = np.linalg.norm(target, axis=1, keepdims=True)
        target *= np.minimum(1.0, (4500 / EARTH_KM) / np.maximum(length, 1e-12))
    else:
        raise ValueError(f"Unknown fusion mode {mode!r}")

    model_config = {
        "task_type": "GPU",
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.75,
        "l2_leaf_reg": 12.0,
        "border_count": 32,
        "thread_count": 4,
        "allow_writing_files": False,
        "verbose": False,
        **options,
        "loss_function": "MultiRMSE",
        "random_seed": int(seed),
        "metadata": {
            "geocv_mode": mode,
            "geocv_shrink": str(shrink),
            "geocv_gain": str(gain),
            "geocv_max_step_radians": str(max_step),
            "geocv_anchor_weights": json.dumps(anchor_weights.tolist()),
        },
    }
    if model_config["task_type"] == "GPU":
        model_config.setdefault("devices", "0")
    model = CatBoostRegressor(**model_config)
    model.fit(x, target.astype(np.float32), sample_weight=sample_weight)
    fused = _fuse(model, x_eval, eval_experts, mode, shrink, gain, max_step, anchor_weights)
    if save_path is not None:
        model.save_model(str(save_path))
    return fused
