"""CatBoost stacking of cross-fitted geographic predictions and diagnostics."""

import numpy as np
from catboost import CatBoostRegressor


EARTH_RADIUS_KM = 6371.0088

CONFIGS = {
    "gate": {
        "mode": "gate", "depth": 5, "iterations": 300, "learning_rate": 0.05,
    },
    "residual_guarded": {
        "mode": "residual", "depth": 5, "iterations": 300, "learning_rate": 0.05,
        "gain": 0.55, "max_step_km": 850,
    },
    "residual": {
        "mode": "residual", "depth": 7, "iterations": 420, "learning_rate": 0.05,
        "gain": 0.80, "max_step_km": 1700,
    },
}


def _unit_xyz(vectors, rows, name):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape != (rows, 3) or not np.isfinite(vectors).all():
        raise ValueError(f"{name} must contain {rows} finite XYZ vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero XYZ vector")
    return vectors / norms


def _gate_targets(dnn, retrieval, truth):
    # For q(t) = unit(dnn + t*(retrieval-dnn)), maximize q(t)·truth
    # over t in [0, 1]. A stationary point is a maximum only if a+b > 0.
    c = np.clip(np.einsum("ij,ij->i", dnn, retrieval), -1.0, 1.0)
    a = np.einsum("ij,ij->i", dnn, truth)
    b = np.einsum("ij,ij->i", retrieval, truth)
    target = (b > a).astype(float)
    interior_possible = (a + b > 0) & (c < 1 - 1e-12) & (c > -1 + 1e-12)
    target[interior_possible] = np.clip(
        (b[interior_possible] - c[interior_possible] * a[interior_possible])
        / ((1 - c[interior_possible]) * (a[interior_possible] + b[interior_possible])),
        0.0, 1.0,
    )
    target[c >= 1 - 1e-12] = 0.5
    # The choice of blend matters less when both predictors are almost identical.
    weight = np.clip(np.sqrt(2 * (1 - c)), 0.08, 1.0)
    return target, weight


def _tangent(vector, anchor):
    return vector - np.einsum("ij,ij->i", vector, anchor)[:, None] * anchor


def fit_predict(features_train, y_train, features_eval, baseline_train, baseline_eval,
                config, seed, save_path=None):
    """Fit a stage-2 candidate, returning float32 unit vectors.

    The gate reads DNN XYZ from columns 0–2 and retrieval XYZ from columns 3–5.
    Additional columns may hold cross-fitted diagnostics and fold-fitted PCA.
    Targets and baseline vectors are Cartesian XYZ with nonzero norm.
    """
    train = np.asarray(features_train, dtype=np.float32)
    evaluation = np.asarray(features_eval, dtype=np.float32)
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]:
        raise ValueError("Training and evaluation features must have the same columns")
    if not len(train) or not len(evaluation):
        raise ValueError("Training and evaluation features must both contain rows")
    truth = _unit_xyz(y_train, len(train), "y_train")
    anchor_eval = _unit_xyz(baseline_eval, len(evaluation), "baseline_eval")

    options = dict(config)
    mode = options.pop("mode")
    weights = None
    metadata = {"geocv_mode": mode}
    if mode == "gate":
        if train.shape[1] < 6:
            raise ValueError("Gate features must begin with DNN XYZ and retrieval XYZ")
        dnn_train = _unit_xyz(train[:, :3], len(train), "DNN training predictions")
        retrieval_train = _unit_xyz(train[:, 3:6], len(train), "Retrieval training predictions")
        dnn_eval = _unit_xyz(evaluation[:, :3], len(evaluation), "DNN evaluation predictions")
        retrieval_eval = _unit_xyz(evaluation[:, 3:6], len(evaluation), "Retrieval evaluation predictions")
        target, weights = _gate_targets(dnn_train, retrieval_train, truth)
        loss = "RMSE"
    elif mode == "residual":
        anchor_train = _unit_xyz(baseline_train, len(train), "baseline_train")
        gain = float(options.pop("gain"))
        max_step_km = float(options.pop("max_step_km"))
        if not 0 < gain <= 1 or not 0 < max_step_km < np.pi * EARTH_RADIUS_KM / 2:
            raise ValueError("Residual gain or maximum step is outside its valid range")
        metadata.update(geocv_gain=str(gain), geocv_max_step_km=str(max_step_km))
        target = _tangent(truth, anchor_train)
        loss = "MultiRMSE"
    else:
        raise ValueError(f"Unknown stack mode {mode!r}")

    params = {
        "task_type": "GPU",
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.75,
        "l2_leaf_reg": 8.0,
        "border_count": 32,
        "thread_count": 4,
        "allow_writing_files": False,
        "verbose": False,
        **options,
        "loss_function": loss,
        "random_seed": int(seed),
        "metadata": metadata,
    }
    if params["task_type"] == "GPU":
        params.setdefault("devices", "0")
    # CatBoost rejects a constant scalar target. This occurs when the base
    # directions coincide everywhere or the same predictor always wins.
    if mode == "gate" and np.all(target == target[0]):
        params["allow_const_label"] = True
    model = CatBoostRegressor(**params)
    model.fit(train, target, sample_weight=weights)
    prediction = model.predict(evaluation)

    if mode == "gate":
        blend = np.clip(prediction, 0.0, 1.0)[:, None]
        corrected = dnn_eval + blend * (retrieval_eval - dnn_eval)
        degenerate = np.linalg.norm(corrected, axis=1) <= 1e-12
        corrected[degenerate] = anchor_eval[degenerate]
    else:
        step = gain * _tangent(prediction, anchor_eval)
        max_tangent = np.tan(max_step_km / EARTH_RADIUS_KM)
        length = np.linalg.norm(step, axis=1, keepdims=True)
        corrected = anchor_eval + step * (max_tangent / np.maximum(length, max_tangent))

    prediction_xyz = _unit_xyz(corrected, len(evaluation), "Stage-2 predictions").astype(np.float32)
    if save_path is not None:
        model.save_model(str(save_path))
    return prediction_xyz
