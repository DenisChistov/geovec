"""GPU retrieval and reliability features from the two unit embeddings."""
import numpy as np
import torch
import torch.nn.functional as F


def weighted_mean(y, similarities, count, temperature):
    y = y[:, :count]
    s = similarities[:, :count]
    w = torch.softmax(s / temperature, dim=1).unsqueeze(-1)
    resultant = (y * w).sum(dim=1)
    mean = F.normalize(resultant, dim=1)
    concentration = resultant.norm(dim=1)
    for _ in range(3):
        dot = (y * mean[:, None]).sum(dim=-1, keepdim=True)
        tangent = y - dot * mean[:, None]
        norm = tangent.norm(dim=-1, keepdim=True)
        angle = torch.atan2(norm, dot)
        step = (w * tangent * (angle / norm.clamp_min(1e-7))).sum(dim=1)
        length = step.norm(dim=-1, keepdim=True)
        mean = F.normalize(torch.cos(length) * mean + torch.sinc(length / torch.pi) * step, dim=1)
    scatter = torch.sqrt(2 * (1 - concentration.clamp(max=1)))
    return mean, scatter


@torch.inference_mode()
def retrieve(X, Y, train_idx, query_idx):
    """Return 6 geographic estimates and a matrix of query-only confidence statistics."""
    dev = 'cuda'
    ref = torch.as_tensor(X[train_idx], device=dev, dtype=torch.float16)
    query = torch.as_tensor(X[query_idx], device=dev, dtype=torch.float16)
    coord = torch.as_tensor(Y[train_idx], device=dev)
    results = {name: [] for name in ('joint', 'short', 'broad', 'geo', 'street', 'favor_geo', 'statistics')}

    def summarize(key, vals, ids, count, tau, stats):
        vals = vals.float()
        estimate, scatter = weighted_mean(coord[ids[:, :count]], vals, count, tau)
        results[key].append(estimate.cpu().numpy())
        column = min(count - 1, vals.shape[1] - 1)
        stats.extend([vals[:, 0], vals[:, column], vals[:, 0] - vals[:, column], scatter])
        return estimate

    for q in query.split(256):
        g = q[:, :512] @ ref[:, :512].T
        s = q[:, 512:] @ ref[:, 512:].T
        stats = []
        count = min(40, len(ref))
        vals, ids = torch.topk((g + s) * .5, count, dim=1)
        joint = summarize('joint', vals, ids, 12, .025, stats)
        summarize('short', vals, ids, 5, .017, stats)
        summarize('broad', vals, ids, 40, .045, stats)
        # Compute cosine retrieval independently to reveal cross-model agreement.
        vals_g, ids_g = torch.topk(g, count, dim=1)
        vals_s, ids_s = torch.topk(s, count, dim=1)
        eg = summarize('geo', vals_g, ids_g, 16, .032, stats)
        es = summarize('street', vals_s, ids_s, 16, .032, stats)
        vals_f, ids_f = torch.topk(g * .70 + s * .30, count, dim=1)
        ef = summarize('favor_geo', vals_f, ids_f, 12, .025, stats)
        stats.extend([(eg * es).sum(dim=1), (ef * joint).sum(dim=1),
                      (ids_g[:, :1] == ids_s[:, :1]).squeeze(1).float()])
        results['statistics'].append(torch.stack(stats,dim=1).cpu().numpy())
    return {name:np.concatenate(values).astype(np.float32) for name,values in results.items()}
