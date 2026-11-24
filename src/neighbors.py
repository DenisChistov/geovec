import numpy as np
import torch

CONFIGS = [
    {'k': 12, 'temperature': 0.025, 'label': '12 соседей, вес 0.025'},
    {'k': 32, 'temperature': 0.04, 'label': '32 соседа, вес 0.04'},
    {'k': 96, 'temperature': 0.07, 'label': '96 соседей, вес 0.07'},
]
_CACHE = {}

def _neighbors(x, train, queries):
    # Each block was normalized on its own. The dot product of concatenated
    # vectors is twice the mean of their cosine similarities.
    dev = 'cuda'
    references = torch.as_tensor(x[train], device=dev, dtype=torch.float16)
    queries = torch.as_tensor(x[queries], device=dev, dtype=torch.float16)
    count = min(max(c['k'] for c in CONFIGS), len(references))
    values, ids = [], []
    for q in queries.split(192):
        score, index = torch.topk(q @ references.T, count, dim=1)
        values.append(score.cpu().numpy())
        ids.append(index.cpu().numpy())
    return np.concatenate(values), np.concatenate(ids)

def train_predict(method, config, X, Y, train_idx, val_idx, seed):
    # Reuse only within the same fold and input arrays.
    key = (id(X), id(train_idx), id(val_idx))
    if _CACHE.get('key') != key:
        _CACHE.clear()
        _CACHE.update(key=key, arrays=(X, train_idx, val_idx), neighbors=_neighbors(X, train_idx, val_idx))
    values, index = _CACHE['neighbors']
    k = config['k']
    out = []
    for start in range(0, len(val_idx), 1024):
        cos = torch.as_tensor(values[start:start+1024, :k], device='cuda').float() * .5
        y = torch.as_tensor(Y[train_idx[index[start:start+1024, :k]]], device='cuda')
        weights = torch.softmax(cos / config['temperature'], dim=1).unsqueeze(-1)
        mean = torch.nn.functional.normalize((weights * y).sum(dim=1), dim=1)
        # Refine the weighted spherical mean for squared angular distance.
        for _ in range(3):
            dot = (y * mean[:, None, :]).sum(dim=-1, keepdim=True)
            tangent = y - dot * mean[:, None, :]
            norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
            angle = torch.atan2(norm, dot)
            step = (weights * tangent * (angle / norm.clamp_min(1e-7))).sum(dim=1)
            length = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
            mean = torch.nn.functional.normalize(
                torch.cos(length) * mean + torch.sinc(length / torch.pi) * step, dim=-1)
        out.append(mean.cpu().numpy())
    return np.concatenate(out)
