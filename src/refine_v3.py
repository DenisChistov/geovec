"""Posterior-conditioned retrieval with local spherical density refinement."""

import numpy as np
import torch
from torch.nn import functional as F


NAMES = ("joint", "short", "broad", "geo", "street", "favor_geo")
REFINED_NAMES = ("refined_mean", "refined_mode", "refined_blend")
FEATURE_NAMES = (
    "posterior_top_mass", "posterior_top_probability", "posterior_concentration",
    "posterior_radius_rad", "retrieval_concentration", "conditioned_concentration",
    "local_concentration", "local_mode_mass", "posterior_support", "mean_log_evidence",
    "effective_retrieval_count", "effective_conditioned_count", "mean_joint_agreement",
    "mode_joint_agreement", "mean_posterior_agreement", "mode_posterior_agreement",
    "blend_joint_agreement", "local_blend_weight",
)

_CACHE = None


def _unit_or(vector, fallback):
    norm = vector.norm(dim=1, keepdim=True)
    return torch.where(norm > 1e-8, vector / norm.clamp_min(1e-8), fallback)


def _spherical_mean(points, weight):
    weight = weight.unsqueeze(-1)
    resultant = (points * weight).sum(dim=1)
    best = points[torch.arange(len(points), device=points.device), weight.squeeze(-1).argmax(dim=1)]
    mean = _unit_or(resultant, best)
    concentration = resultant.norm(dim=1).clamp(0, 1)
    for _ in range(3):
        dot = (points * mean[:, None]).sum(dim=-1, keepdim=True)
        tangent = points - dot * mean[:, None]
        norm = tangent.norm(dim=-1, keepdim=True)
        angle = torch.atan2(norm, dot)
        step = (weight * tangent * (angle / norm.clamp_min(1e-7))).sum(dim=1)
        length = step.norm(dim=1, keepdim=True)
        mean = _unit_or(torch.cos(length) * mean + torch.sinc(length / torch.pi) * step, mean)
    return mean, concentration


def _retrieval_candidates(X, Y, train, query, device):
    global _CACHE
    if _CACHE is not None:
        old_x, old_y, old_train, old_query, result = _CACHE
        if old_x is X and old_y is Y and np.array_equal(old_train, train) and np.array_equal(old_query, query):
            return result

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    ref = torch.as_tensor(X[train], device=device, dtype=dtype)
    queries = torch.as_tensor(X[query], device=device, dtype=dtype)
    coordinates = torch.as_tensor(Y[train], device=device, dtype=torch.float32)
    coordinates = F.normalize(coordinates, dim=1)
    count = min(64, len(ref))
    result = {name: [] for name in (*NAMES, "statistics", "candidate_ids", "candidate_scores")}

    for q in queries.split(256):
        geo_scores = q[:, :512] @ ref[:, :512].T
        street_scores = q[:, 512:] @ ref[:, 512:].T
        stats = []

        def summarize(name, values, ids, k, tau):
            k = min(k, count)
            values = values[:, :k].float()
            points = coordinates[ids[:, :k]]
            mean, concentration = _spherical_mean(points, torch.softmax(values / tau, dim=1))
            result[name].append(mean.cpu().numpy())
            stats.extend([values[:, 0], values[:, -1], values[:, 0] - values[:, -1],
                          (2 * (1 - concentration)).sqrt()])
            return mean

        scores, ids = torch.topk((geo_scores + street_scores) * .5, count, dim=1)
        result["candidate_ids"].append(ids.cpu().numpy().astype(np.int32))
        result["candidate_scores"].append(scores.cpu().numpy().astype(np.float32))
        joint = summarize("joint", scores, ids, 12, .025)
        summarize("short", scores, ids, 5, .017)
        summarize("broad", scores, ids, 40, .045)
        separate_count = min(16, count)
        g, gi = torch.topk(geo_scores, separate_count, dim=1)
        s, si = torch.topk(street_scores, separate_count, dim=1)
        geo = summarize("geo", g, gi, 16, .032)
        street = summarize("street", s, si, 16, .032)
        fav, fi = torch.topk(geo_scores * .70 + street_scores * .30, min(12, count), dim=1)
        favor_geo = summarize("favor_geo", fav, fi, 12, .025)
        stats.extend([(geo * street).sum(dim=1), (favor_geo * joint).sum(dim=1),
                      (gi[:, 0] == si[:, 0]).float()])
        result["statistics"].append(torch.stack(stats, dim=1).cpu().numpy())

    result = {name: np.concatenate(parts) for name, parts in result.items()}
    _CACHE = (X, Y, np.array(train, copy=True), np.array(query, copy=True), result)
    return result


def _local_evidence(points, centers, probs, radii, counts, train_size):
    mass = probs.sum(dim=1).clamp(0, 1)
    total_radius = (probs * radii.square()).sum(dim=1) / mass.clamp_min(1e-8)
    # Regularize individual radii toward the posterior-weighted cell radius.
    # A spherical cluster with a single point may have a measured radius of 0.
    # 1e-8 is a numeric floor for cosine differences in float32.
    variance = (.5 * (radii.square() + total_radius[:, None])).clamp_min(1e-8)
    dot = (points @ centers.transpose(1, 2)).clamp(-1, 1)
    affinity = torch.exp((dot - 1) / variance[:, None, :])
    # Use a spherical kernel normalized relative to uniform surface density.
    # The omitted posterior mass is modeled as a uniform background.
    scale = variance * (-torch.expm1(-2 / variance))
    kernel = affinity / scale[:, None, :].clamp_min(1e-8)
    posterior_density = (kernel * probs[:, None]).sum(dim=2) + .5 * (1 - mass[:, None])
    prior = counts / train_size
    prior_mass = prior.sum(dim=1).clamp(0, 1)
    prior_density = (kernel * prior[:, None]).sum(dim=2) + .5 * (1 - prior_mass[:, None])
    # Dividing by the training prior avoids rewarding a dense training cell
    # twice. Use a tempered likelihood because both estimators use the same X.
    evidence = posterior_density.clamp_min(1e-8).log() - prior_density.clamp_min(1e-8).log()
    return evidence, affinity, total_radius.clamp_min(1e-8), mass


def _refine_batch(points, scores, joint, centers, probs, radii, counts, global_pred, train_size):
    visual_logits = scores / .035
    visual_weight = torch.softmax(visual_logits, dim=1)
    evidence, affinity, radius, mass = _local_evidence(points, centers, probs, radii, counts, train_size)
    local_logits = visual_logits + .5 * evidence
    combined_weight = torch.softmax(local_logits, dim=1)
    mean, conditioned_concentration = _spherical_mean(points, combined_weight)

    # Candidate-centred spatial KDE. The bandwidth is learned from the cells'
    # training-fold geographic diameters. Select the strongest coherent region,
    # then refine inside it with a softly weighted geographic mean.
    local_dot = (points @ points.transpose(1, 2)).clamp(-1, 1)
    regional_logits = (local_dot - 1) / radius[:, None, None]
    regional_mass = (regional_logits.exp() * combined_weight[:, None]).sum(dim=2)
    rows = torch.arange(len(points), device=points.device)
    best_region = regional_mass.argmax(dim=1)
    mode_mass = regional_mass[rows, best_region]
    mode_weight = torch.softmax(local_logits + regional_logits[rows, best_region], dim=1)
    mode, mode_concentration = _spherical_mean(points, mode_weight)

    posterior_concentration = (centers * probs[:, :, None]).sum(dim=1).norm(dim=1).clamp(0, 1)
    visual_concentration = (points * visual_weight[:, :, None]).sum(dim=1).norm(dim=1).clamp(0, 1)
    posterior_support = (visual_weight * (affinity * probs[:, None]).sum(dim=2)).sum(dim=1)
    local_strength = conditioned_concentration * ((scores[:, 0] + 1) * .5).clamp(0, 1)
    blend_weight = local_strength / (local_strength + posterior_concentration).clamp_min(1e-8)
    blend = _unit_or(mean * blend_weight[:, None] + global_pred * (1 - blend_weight[:, None]), joint)

    features = torch.stack((
        mass, probs[:, 0], posterior_concentration, radius.sqrt(), visual_concentration,
        conditioned_concentration, mode_concentration, mode_mass, posterior_support,
        (visual_weight * evidence).sum(dim=1), visual_weight.square().sum(dim=1).reciprocal(),
        combined_weight.square().sum(dim=1).reciprocal(), (mean * joint).sum(dim=1),
        (mode * joint).sum(dim=1), (mean * global_pred).sum(dim=1),
        (mode * global_pred).sum(dim=1), (blend * joint).sum(dim=1), blend_weight,
    ), dim=1)
    return mean, mode, blend, features


@torch.inference_mode()
def refine(X, Y, train_idx, query_idx, posterior, seed):
    """Return baseline retrieval, three refinements, and fixed-width features.

    Posterior arrays (`pred`, `top_centers`, `top_probs`, `top_radii`,
    `top_counts`) must come from a classifier fitted only on `train_idx`.
    Radii are RMS angular distances in radians to cells' training points.
    Only training-point coordinates are read by the retrieval path.
    """
    train = np.asarray(train_idx)
    query = np.asarray(query_idx)
    if X.ndim != 2 or X.shape[1] != 1280 or not len(train) or not len(query):
        raise ValueError("Refinement needs 1280 input features and nonempty train/query indices")
    if np.intersect1d(train, query).size:
        raise ValueError("Retrieval reference and query sets must be disjoint")
    if any(len(posterior[name]) != len(query) for name in
           ("pred", "top_centers", "top_probs", "top_radii", "top_counts")):
        raise ValueError("Posterior rows must match query_idx")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retrieval = _retrieval_candidates(X, Y, train, query, device)
    result = {name: retrieval[name].astype(np.float32, copy=False) for name in (*NAMES, "statistics")}
    refinements = {name: [] for name in (*REFINED_NAMES, "features")}
    for start in range(0, len(query), 256):
        selection = slice(start, start + 256)
        ids = train[retrieval["candidate_ids"][selection]]
        points = F.normalize(torch.as_tensor(Y[ids], device=device, dtype=torch.float32), dim=2)
        scores = torch.as_tensor(retrieval["candidate_scores"][selection], device=device, dtype=torch.float32)
        joint = torch.as_tensor(result["joint"][selection], device=device)
        params = [torch.as_tensor(posterior[name][selection], device=device, dtype=torch.float32)
                  for name in ("top_centers", "top_probs", "top_radii", "top_counts", "pred")]
        for name, values in zip(refinements, _refine_batch(points, scores, joint, *params, len(train))):
            refinements[name].append(values.cpu().numpy().astype(np.float32, copy=False))
    result.update({name: np.concatenate(parts) for name, parts in refinements.items()})
    return result
