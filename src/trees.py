"""CatBoost regression of unit Cartesian coordinates for geographic CV."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from catboost import CatBoostRegressor, Pool


# Train the longer depth-seven model first so its first 280 trees can be
# evaluated as a second candidate without fitting another booster.
CONFIGS = [
    {"depth": 6, "iterations": 260, "learning_rate": 0.12},
    {"depth": 7, "iterations": 480, "learning_rate": 0.10},
    {"depth": 7, "iterations": 280, "learning_rate": 0.10},
    {"depth": 8, "iterations": 1000, "learning_rate": 0.08},
    {"depth": 8, "iterations": 650, "learning_rate": 0.08},
]

_fold = None
_booster = None


def _pools(X, Y, train_idx, val_idx, border_count, thread_count):
    global _fold, _booster
    if _fold is not None:
        (
            previous_X, previous_Y, previous_train_idx, previous_val_idx,
            previous_border_count, previous_thread_count, train_pool, val_pool,
        ) = _fold
        if (
            previous_X is X
            and previous_Y is Y
            and previous_border_count == border_count
            and previous_thread_count == thread_count
            and np.array_equal(previous_train_idx, train_idx)
            and np.array_equal(previous_val_idx, val_idx)
        ):
            return train_pool, val_pool

    _fold = _booster = None
    train_pool = Pool(X[train_idx], label=Y[train_idx], thread_count=thread_count)
    train_pool.quantize(border_count=border_count)
    # Applying training borders prevents any use of validation data when
    # choosing the thresholds used to discretize the features.
    with TemporaryDirectory() as temp_dir:
        borders = str(Path(temp_dir) / "catboost-borders.tsv")
        train_pool.save_quantization_borders(borders)
        val_pool = Pool(X[val_idx], thread_count=thread_count)
        val_pool.quantize(input_borders=borders)

    _fold = (
        X, Y, np.array(train_idx, copy=True), np.array(val_idx, copy=True),
        border_count, thread_count, train_pool, val_pool,
    )
    return train_pool, val_pool


def train_predict(method, config, X, Y, train_idx, val_idx, seed):
    """Fit on train_idx and return unit XYZ for val_idx in its given order."""
    global _booster
    options = dict(config)
    save_path = options.pop("save_path", None)
    iterations = int(options.pop("iterations"))
    border_count = int(options.pop("border_count", 32))
    thread_count = int(options.pop("thread_count", 4))
    task_type = options.pop("task_type", "GPU")
    params = {
        "loss_function": "MultiRMSE",
        "task_type": task_type,
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.67,
        "random_seed": int(seed),
        "thread_count": thread_count,
        "allow_writing_files": False,
        "verbose": 25,
        **options,
    }
    if task_type == "GPU":
        params.setdefault("devices", "0")

    train_pool, val_pool = _pools(X, Y, train_idx, val_idx, border_count, thread_count)
    if _booster is not None and _booster[0] == params and _booster[1].tree_count_ >= iterations:
        model = _booster[1]
    else:
        model = CatBoostRegressor(iterations=iterations, **params)
        model.fit(train_pool)
        _booster = (params, model)

    if save_path:
        model.save_model(save_path)

    xyz = np.asarray(model.predict(val_pool, ntree_end=iterations), dtype=np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError("CatBoost returned nonfinite coordinates")
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    if zero.any():
        xyz[zero] = Y[train_idx[0]]
        norms[zero] = np.linalg.norm(xyz[zero], axis=1, keepdims=True)
    return xyz / norms
