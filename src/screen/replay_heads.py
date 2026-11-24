"""Query-only replay for checkpoints written by neural_models and meta_models."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import meta_models
import neural_models


def _load(path, query, methods, scale_key):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device, weights_only=True)
    method = state["method"]
    if method not in methods:
        raise ValueError(f"Unexpected checkpoint method {method!r}")
    values = np.asarray(query)
    mean, scale = state["mean"], state[scale_key]
    if (values.ndim != 2 or values.shape[1] != mean.numel() or
            mean.ndim != 1 or mean.shape != scale.shape or not torch.all(scale > 0)):
        raise ValueError("Query feature width or checkpoint standardization is invalid")
    return state, method, device, values


def _batches(values, state, scale_key, device, batch_size):
    for start in range(0, len(values), batch_size):
        rows = slice(start, start + batch_size)
        block = np.ascontiguousarray(values[rows], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Query features must be finite")
        x = torch.as_tensor(block, device=device)
        yield rows, ((x - state["mean"]) / state[scale_key]).clamp(-12, 12)


def _unit(xyz):
    lengths = np.linalg.norm(xyz, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or (lengths < 1e-8).any():
        raise ValueError("Checkpoint returned a nonfinite or zero direction")
    return np.ascontiguousarray(xyz / lengths, dtype=np.float32)


@torch.inference_mode()
def neural(path, X_query):
    """Apply a saved neural head to its already-transformed query features."""
    state, method, device, values = _load(path, X_query, neural_models.METHODS, "std")
    out = np.empty((len(values), 3), dtype=np.float32)
    matrix = method == "random_kernel"
    if not matrix:
        model = neural_models.Model(state["dim"], method).to(device)
        model.load_state_dict(state["state"])
        model.eval()
    batch_size = 1024 if method == "token_attention" else 4096
    for rows, z in _batches(values, state, "std", device, batch_size):
        # prepare_xy rounds all neural inputs to fp16, including the Fourier
        # ridge head whose matrix products are subsequently evaluated in fp32.
        z = z.half()
        if matrix:
            prediction = neural_models.norm((z.float() @ state["W"] + state["phase"]).cos() @ state["coef"])
        else:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                prediction = model(z if device.type == "cuda" else z.float())[0]
        out[rows] = prediction.float().cpu().numpy()
    return _unit(out)


def _meta_network(width, state, device):
    # meta_models.fit_predict builds this sequential architecture inline.
    out_width = state["state"]["6.weight"].shape[0]
    model = nn.Sequential(nn.Linear(width, 128), nn.LayerNorm(128), nn.GELU(),
                          nn.Dropout(.12), nn.Linear(128, 64), nn.GELU(),
                          nn.Linear(64, out_width)).to(device)
    model.load_state_dict(state["state"])
    return model.eval()


@torch.inference_mode()
def meta(path, X_view, experts):
    """Apply a saved meta head to its feature view and ordered seven V3 experts."""
    state, method, device, values = _load(path, X_view, meta_models.METHODS, "sd")
    experts = np.asarray(experts)
    if experts.shape != (len(values), 7, 3):
        raise ValueError("Expected seven row-aligned XYZ experts per query")
    out = np.empty((len(values), 3), dtype=np.float32)
    cat = method.startswith("cat_")
    mlp = method.startswith("mlp_")
    if cat:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(path) + ".cbm")
    elif mlp:
        model = _meta_network(values.shape[1], state, device)
    for rows, z in _batches(values, state, "sd", device, 4096):
        e = np.ascontiguousarray(experts[rows], dtype=np.float32)
        if not np.isfinite(e).all():
            raise ValueError("Expert directions must be finite")
        anchor = e[:, 1]
        if cat:
            prediction = np.asarray(model.predict(z.cpu().numpy(), thread_count=4), dtype=np.float32)
        elif mlp:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                rounded = z.half()
                value = model(rounded if device.type == "cuda" else rounded.float()).float()
                if method == "mlp_gate":
                    directions = torch.as_tensor(e, device=device)
                    value = F.normalize((value.softmax(1)[:, :, None] * directions).sum(1), dim=1)
            prediction = value.cpu().numpy()
        else:
            if method == "ridge_interaction":
                a = torch.as_tensor(anchor, device=device)
                z = torch.cat([z, (z[:, :, None] * a[:, None, :]).flatten(1)], dim=1)
            elif method == "kernel_ridge":
                z = (z @ state["W"] + state["phase"]).cos()
            z = torch.cat([z, torch.ones((len(z), 1), dtype=z.dtype, device=device)], dim=1)
            prediction = (z @ state["coefficients"]).cpu().numpy()
        if method.endswith("_wgs84"):
            prediction = meta_models.expmap(anchor, prediction)
        elif method in ("ridge_residual", "ridge_interaction"):
            prediction = prediction + anchor
        elif method == "mlp_xyz":
            zero = np.linalg.norm(prediction, axis=1) < 1e-8
            prediction[zero] = anchor[zero]
        out[rows] = prediction
    return _unit(out)
