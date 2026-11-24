"""Fit and replay the ten ensemble_models fusion rules on ordered expert directions."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import ensemble_models as ensemble


def _experts(value):
    data = np.asarray(value)
    if data.ndim != 3 or data.shape[1] < 1 or data.shape[2] != 3:
        raise ValueError("Expected [rows, experts, 3] directions")
    if not np.isfinite(data).all():
        raise ValueError("Expert directions must be finite")
    return data


def save_rule(path, method, train, truth):
    """Save fitted parameters for one rule; train and truth must be training OOF rows."""
    if method not in ensemble.METHODS:
        raise ValueError(f"Unknown fusion rule {method!r}")
    train = _experts(train)
    truth = np.asarray(truth)
    n, m, _ = train.shape
    if n == 0 or truth.shape != (n, 3) or not np.isfinite(truth).all():
        raise ValueError("Expected one finite training direction per row")

    state = {"version": 1, "method": method, "n_experts": m}
    distance = ensemble.risk_matrix(train, truth)
    mse = np.mean(distance**2, axis=0)
    weights = 1 / (mse + 1e-8)
    weights /= weights.sum()
    state["weights"] = weights
    if method == "uniform":
        state["weights"] = np.ones(m) / m
    elif method == "soft_risk":
        soft = np.exp(-(mse - mse.min()) / max(mse.mean() * .15, 1))
        state["weights"] = soft / soft.sum()
    elif method == "simplex":
        vectors = train.transpose(0, 2, 1).reshape(-1, m)
        gram = vectors.T @ vectors / len(vectors)
        cross = vectors.T @ truth.reshape(-1) / len(vectors)
        result = minimize(lambda w: float(w @ gram @ w - 2 * cross @ w),
                          weights.astype(float), bounds=[(0, 1)] * m,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                          method="SLSQP")
        state["weights"] = result.x / result.x.sum()
    elif method in ("signed_wgs84", "regional_risk"):
        anchor = ensemble.mean(train, weights)
        if method == "regional_risk":
            state.update(anchor=anchor, squared_distance=distance**2, mse=mse)
        else:
            offsets = np.stack([ensemble.logmap(anchor, train[:, i]) for i in range(m)], axis=2)
            design = offsets.reshape(-1, m)
            target = ensemble.logmap(anchor, truth).reshape(-1)
            state["coef"] = np.linalg.solve(design.T @ design + np.eye(m) * len(design) * .002,
                                            design.T @ target)
    elif method == "context_gate":
        _, context = ensemble._features(train)
        center, scale = context.mean(0), context.std(0).clip(.01)
        centered = (context - center) / scale
        design = np.column_stack([centered, np.ones(n)])
        gram = design.T @ design + np.eye(design.shape[1]) * n * .02
        coef = np.linalg.solve(gram, design.T @ np.log1p(distance / 250))
        state.update(center=center, scale=scale, coef=coef)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = file.name
        try:
            np.savez(file, **state)
            file.close()
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def predict_saved(path, query):
    """Replay a fitted fusion rule; query experts must retain their training column order."""
    query = _experts(query)
    with np.load(path, allow_pickle=False) as state:
        method = str(state["method"].item())
        m = int(state["n_experts"].item())
        if int(state["version"].item()) != 1 or method not in ensemble.METHODS:
            raise ValueError("Unsupported fusion checkpoint")
        if query.shape[1] != m:
            raise ValueError(f"Fusion checkpoint expects {m} experts")
        if not len(query):
            return np.empty((0, 3), dtype=np.float32)
        weights = state["weights"]
        if method in ("uniform", "inverse_risk", "soft_risk", "simplex", "median"):
            return ensemble.mean(query, weights, median=method == "median")
        if method == "medoid":
            dot = np.einsum("nmd,nkd->nmk", query, query).clip(-1, 1)
            cost = (np.arccos(dot)**2 * weights[None, None, :]).sum(2)
            return query[np.arange(len(query)), cost.argmin(1)]
        if method == "context_gate":
            _, context = ensemble._features(query)
            centered = (context - state["center"]) / state["scale"]
            design = np.column_stack([centered, np.ones(len(query))])
            return ensemble.mean(query, ensemble._softmax(-(design @ state["coef"]) * 2))

        anchor = ensemble.mean(query, weights)
        if method == "trim_consensus":
            gap = ensemble.angular(query, anchor[:, None])
            scale = np.maximum(np.median(gap, axis=1, keepdims=True), .02)
            local = np.exp(-gap / scale) * weights
            return ensemble.mean(query, local / local.sum(1, keepdims=True))
        if method == "signed_wgs84":
            offsets = np.stack([ensemble.logmap(anchor, query[:, i]) for i in range(m)], axis=2)
            return ensemble.expmap(anchor, (offsets @ state["coef"]).astype(np.float32))
        training_anchor = state["anchor"]
        squared_distance = state["squared_distance"]
        mse = state["mse"]
        tree = cKDTree(training_anchor)
        k = min(128, len(training_anchor))
        output = np.empty((len(query), 3), dtype=np.float32)
        for start in range(0, len(query), 4096):
            rows = slice(start, start + 4096)
            _, ids = tree.query(anchor[rows], k=k, workers=2)
            ids = ids.reshape(-1, k)
            local_mse = squared_distance[ids].mean(axis=1)
            local = 1 / (.7 * local_mse + .3 * mse + 1e-8)
            output[rows] = ensemble.mean(query[rows], local / local.sum(1, keepdims=True))
        return output
