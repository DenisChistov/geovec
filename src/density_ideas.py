"""Fold-local spherical estimators using the saved V3 ranked cell posteriors."""

import numpy as np


DESCRIPTIONS = {
    "density_pooled_frechet": "Pool all three ranked cell posteriors and reconstruct each omitted tail from its full spherical mean.",
    "density_peer_kernel": "Reweight ranked cells by geographic support from the other two independently clustered classifiers.",
    "density_regional_median": "Build three regional basins per image and replace pooled basin probabilities with the across-classifier median.",
    "density_local_offset": "Use retrieval to adjust locations within nearby cells while preserving each cell's posterior probability.",
    "density_retrieval_likelihood": "Condition the multimodal cell posterior on a bounded geographic likelihood from visual retrieval.",
    "density_precision_fusion": "Fuse retrieval and classifier means by their respective posterior dispersions and visual modality agreement.",
    "density_anisotropic_update": "Apply a tangent-plane Kalman update using the geographic posterior covariance and retrieval dispersion.",
    "density_prior_temperature": "Adapt classifier temperature to modal-cell prior prevalence and agreement across independently fitted grids.",
    "density_tail_calibration": "Recalibrate omitted tail mass using independent grids and retrieval to check the leading cell's region.",
    "density_coarse_fine_transport": "Transport coarse cell locations toward supported neighbors in the finer grid while preserving coarse mass.",
    "density_modality_recovery": "Recover joint retrieval and direct-regression directions from saved agreement statistics, then compare regional support.",
    "density_classifier_reliability": "Weight complete classifier posteriors by how well their regional probability distributions match the other grids.",
}


def _unit(value, fallback):
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    return np.where(length > 1e-7, value / np.maximum(length, 1e-7), fallback)


def _logmap(origin, target):
    dot = np.clip(np.sum(origin * target, axis=-1, keepdims=True), -1, 1)
    tangent = target - dot * origin
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    step = tangent * (np.arctan2(length, dot) / np.maximum(length, 1e-7))
    # An exact antipode has multiple shortest paths. Pick a deterministic one.
    singular = (length[..., 0] <= 1e-7) & (dot[..., 0] < 0)
    if singular.any():
        vectors = np.broadcast_to(origin, step.shape)[singular]
        axes = np.eye(3, dtype=np.float32)[np.argmin(abs(vectors), axis=-1)]
        perp = np.cross(vectors, axes)
        step[singular] = np.pi * perp / np.linalg.norm(perp, axis=-1, keepdims=True)
    return step


def _expmap(origin, tangent, max_step=np.pi):
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    angle = np.minimum(length, max_step)
    return _unit(np.cos(angle) * origin + tangent * (np.sin(angle) / np.maximum(length, 1e-9)), origin)


def _toward(a, b, fraction):
    return _expmap(a, np.asarray(fraction, dtype=a.dtype)[..., None] * _logmap(a, b))


def _prob(weights):
    return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-15)


def _mean(points, weights, initial):
    weights = _prob(weights)[..., None]
    mean = initial
    for _ in range(3):
        direction = (weights * _logmap(mean[:, None, :], points)).sum(axis=1)
        mean = _expmap(mean, direction, max_step=1.0)
    return mean


def _atoms(features, experts, coarse_midpoint):
    n = len(features)
    post = features[:, 21:180].reshape(n, 3, 53)
    rank = post[:, :, :40].reshape(n, 3, 10, 4)
    base = experts[:, 1]
    # Multi-scale is one third of the way from the 512-cell midpoint to the
    # fine-grid mean. The order below matches the saved A, B, C feature blocks.
    means = np.stack((experts[:, 0], _toward(coarse_midpoint, base, 3.0),
                      _toward(experts[:, 0], coarse_midpoint, 2.0)), axis=1)
    mass = np.clip(rank[..., 0], 0, 1).copy()
    mass /= np.maximum(1, mass.sum(axis=2, keepdims=True))
    coords = _unit(rank[..., 1:], means[:, :, None, :])
    tail_mass = np.maximum(0, 1 - mass.sum(axis=2))
    # At a Fréchet mean the posterior-weighted log vectors sum to zero.
    # Represent the missing cells with one point that supplies their inferred
    # first moment. A finite cap handles full-grid iteration error near zero mass.
    gradient = (mass[..., None] * _logmap(means[:, :, None, :], coords)).sum(axis=2)
    tail = _expmap(means, -gradient / np.maximum(tail_mass[..., None], 1e-6), max_step=np.pi * .98)
    points = np.concatenate((coords, tail[:, :, None, :]), axis=2)
    probs = np.concatenate((mass, tail_mass[..., None]), axis=2)
    return points, probs, post[:, :, 40:]


def _basin_weights(points, weights, seed):
    """Three weighted farthest-first seeds and their nearest-center partition."""
    rows = np.arange(len(points))
    centers = [seed]
    nearest = np.sum(points * seed[:, None, :], axis=-1)
    for _ in range(2):
        selected = np.argmax(weights * (1 - nearest).clip(0, 2), axis=1)
        center = points[rows, selected]
        centers.append(center)
        nearest = np.maximum(nearest, np.sum(points * center[:, None, :], axis=-1))
    assignments = np.argmax(points @ np.stack(centers, axis=-1), axis=2)
    indicators = np.eye(3, dtype=np.float32)[assignments]
    regional = (weights[..., None] * indicators).reshape(len(points), 3, 11, 3).sum(axis=2)
    regional = _prob(regional)
    median = np.median(regional, axis=1)
    pooled = regional.mean(axis=1)
    # A three-way disagreement can give a zero median to every region.
    target = _prob(.80 * median + .20 * pooled)
    ratios = target / np.maximum(pooled, 1e-8)
    return weights * np.sum(indicators * ratios[:, None, :], axis=2)


def _recover_modalities(experts, ref):
    local, mode, hybrid = experts[:, 4], experts[:, 5], experts[:, 6]
    blend_weight = ref[:, 17:18]
    blended = _unit(blend_weight * local + (1 - blend_weight) * experts[:, 0], local)
    basis = np.stack((local, mode, blended), axis=1)
    target = ref[:, [12, 13, 16], None]
    # These three measured dot products constrain the original retrieval
    # direction. In near-singular geometry regularize toward the saved hybrid.
    transposed = np.swapaxes(basis, 1, 2)
    ridge = 2e-5 * np.eye(3, dtype=np.float32)
    residual = target - basis @ hybrid[..., None]
    delta = np.linalg.solve(transposed @ basis + ridge, transposed @ residual)
    joint = _unit(hybrid + delta[..., 0], hybrid)
    direct = _unit(2 * np.sum(joint * hybrid, axis=1, keepdims=True) * hybrid - joint, hybrid)
    return joint, direct


def _batch(features, experts, coarse_midpoint):
    n = len(features)
    points, probs, confidence = _atoms(features, experts, coarse_midpoint)
    ref, stats = features[:, 180:198], features[:, 198:225]
    base, local, mode, hybrid = (experts[:, i] for i in (1, 4, 5, 6))
    atoms, weights = points.reshape(n, 33, 3), probs.reshape(n, 33) / 3
    pooled = _mean(atoms, weights, base)
    out = {"density_pooled_frechet": pooled}

    # Bandwidth in squared angular radians. A floor accommodates different
    # cell tessellations, and the upper cap keeps the peer test regional.
    width = np.clip(2 * ref[:, 3], .045, .18)
    dots = np.clip(atoms @ atoms.swapaxes(1, 2), -1, 1)
    kernels = np.exp((dots - 1) / width[:, None, None]**2)
    support = (kernels.reshape(n, 33, 3, 11) * probs[:, None]).sum(axis=3)
    grids = np.repeat(np.arange(3), 11)
    own = support[:, np.arange(33), grids]
    peer = .5 * (support.sum(axis=2) - own).clip(0)
    factor = np.sqrt(.04 + peer)
    # The omitted mass represents many unobserved locations, so it cannot
    # legitimately gain or lose probability from this pointwise kernel.
    ranked = (probs * factor.reshape(n, 3, 11))[:, :, :10]
    ranked = _prob(ranked) * probs[:, :, :10].sum(axis=2, keepdims=True)
    endorsed = np.concatenate((ranked, probs[:, :, 10:]), axis=2)
    out["density_peer_kernel"] = _mean(atoms, endorsed.reshape(n, 33), pooled)

    consensus_mode = np.argmax(weights * (.08 + peer), axis=1)
    seed = atoms[np.arange(n), consensus_mode]
    region_weights = _basin_weights(atoms, weights, seed)
    out["density_regional_median"] = _mean(atoms, region_weights, pooled)

    at_mode = np.clip(np.sum(atoms * mode[:, None, :], axis=2), -1, 1)
    local_width = np.clip(2 * ref[:, 3], .018, .12)
    inside = np.exp((at_mode - 1) / local_width[:, None]**2)
    inside[:, 10::11] = 0
    offset = inside * (.25 + .65 * ref[:, 7:8].clip(0, 1))
    adjusted = _toward(atoms, mode[:, None, :], offset)
    out["density_local_offset"] = _mean(adjusted, weights, pooled)

    agreement = ((stats[:, 24].clip(-1, 1) + 1) / 2)**2
    concentration = ref[:, 4].clip(0, 1)
    near_local = np.clip(np.sum(atoms * local[:, None, :], axis=-1), -1, 1)
    spatial_width = np.clip(.08 + .65 * stats[:, 3], .08, .5)
    likelihood = np.exp((near_local - 1) / spatial_width[:, None]**2)
    evidence = 1 + (1 + 3 * agreement * concentration)[:, None] * likelihood
    out["density_retrieval_likelihood"] = _mean(atoms, weights * evidence, pooled)

    classifier_var = np.mean(confidence[:, :, 9].clip(0, np.pi)**2, axis=1) + .018**2
    # The saved joint compactness is sqrt(2*(1-resultant)), a chord spread.
    # A penalty for disagreement represents extra uncertainty in the measurement.
    retrieval_var = (.055 + stats[:, 3].clip(0, 2))**2 + .5 * (1 - agreement)
    gain = np.clip(classifier_var / (classifier_var + retrieval_var), .04, .58)
    out["density_precision_fusion"] = _toward(base, hybrid, gain)

    tangents = _logmap(pooled[:, None, :], atoms)
    tangents -= np.sum(tangents * weights[..., None], axis=1, keepdims=True)
    covariance = tangents.swapaxes(1, 2) @ (tangents * weights[..., None])
    eye = np.eye(3, dtype=np.float32)
    noise = (.04**2 + .65 * retrieval_var)[:, None, None] * eye
    innovation = _logmap(pooled, hybrid)
    update = covariance @ np.linalg.solve(covariance + noise, innovation[..., None])
    out["density_anisotropic_update"] = _expmap(pooled, .65 * update[..., 0], max_step=.65)

    top = points[:, :, 0, :]
    modal_dots = np.clip(top @ top.swapaxes(1, 2), -1, 1)
    concordance = (np.exp((modal_dots - 1) / .16**2).sum(axis=(1, 2)) - 3) / 6
    relative_prior = confidence[:, :, 12] / np.maximum(confidence[:, :, 11], 1e-6)
    popularity = np.mean(np.log(np.clip(relative_prior, .2, 5)), axis=1)
    temp_direction = np.tanh(1.6 * (concordance - .5) - .6 * popularity)
    temperature = np.where((temp_direction > 0)[:, None], experts[:, 2], experts[:, 3])
    out["density_prior_temperature"] = _toward(base, temperature, 1.5 * abs(temp_direction))

    top_retrieval = np.sum(top * hybrid[:, None, :], axis=2).clip(-1, 1)
    matches = np.exp((top_retrieval - 1) / .25**2)
    peer_at_top = peer.reshape(n, 3, 11)[:, :, 0]
    # Positive evidence for a leading region decreases the mass attributed
    # to unreported cells. Weak or contradictory evidence restores that mass.
    tail_odds = np.exp(.85 * (.68 - matches - peer_at_top))
    calibrated = probs.copy()
    calibrated[:, :, -1] *= tail_odds
    out["density_tail_calibration"] = _mean(atoms, _prob(calibrated).reshape(n, 33), pooled)

    transported = points.copy()
    fine = points[:, 1, :10]
    for family in (0, 2):
        coarse = points[:, family, :10]
        cross = np.clip(coarse @ fine.swapaxes(1, 2), -1, 1)
        affinity = np.exp((cross - 1) / local_width[:, None, None]**2) * probs[:, 1, None, :10]
        matched = affinity.sum(axis=2)
        destination = _unit(affinity @ fine, coarse)
        transported[:, family, :10] = _toward(coarse, destination, matched / (.06 + matched))
    out["density_coarse_fine_transport"] = _mean(transported.reshape(n, 33, 3), weights, pooled)

    joint, direct = _recover_modalities(experts, ref)
    probes = np.stack((joint, direct), axis=2)
    probe_dot = np.clip(atoms @ probes, -1, 1)
    probe_mass = (weights[..., None] * np.exp((probe_dot - 1) / .22**2)).sum(axis=1)
    joint_weight = (.08 + agreement * probe_mass[:, 0]) / (.16 + agreement * probe_mass[:, 0] + probe_mass[:, 1])
    endorsed_modality = _toward(direct, joint, joint_weight)
    out["density_modality_recovery"] = _toward(base, endorsed_modality, .12 + .45 * gain)

    # Compare distributions with the kernel maximum mean discrepancy. Pairwise
    # support is averaged symmetrically and the within-grid terms correct for
    # different levels of concentration across 512- and 1024-cell grids.
    regional_kernel = np.exp((dots - 1) / .28**2)
    regional_support = (regional_kernel.reshape(n, 33, 3, 11) * probs[:, None]).sum(axis=3)
    overlap = (regional_support.reshape(n, 3, 11, 3) * probs[..., None]).sum(axis=2)
    self_overlap = np.diagonal(overlap, axis1=1, axis2=2)
    discrepancy = np.maximum(0, self_overlap[:, :, None] + self_overlap[:, None, :] - 2 * overlap)
    reliability = _prob(np.exp(-3 * discrepancy.sum(axis=2)))
    out["density_classifier_reliability"] = _mean(atoms, (probs * reliability[..., None]).reshape(n, 33), pooled)
    return out


def predict(bundle):
    """Predict from V3 features, seven experts and the 512-cell ensemble midpoint."""
    features = np.asarray(bundle["features"])
    experts = np.asarray(bundle["experts"])
    midpoint = np.asarray(bundle["ensemble_512"])
    n = len(features)
    if features.ndim != 2 or features.shape[1] < 225 or experts.shape != (n, 7, 3) or midpoint.shape != (n, 3):
        raise ValueError("Expected the V3 features, seven XYZ experts and the 512-cell midpoint")
    result = {name: np.empty((n, 3), dtype=np.float32) for name in DESCRIPTIONS}
    for start in range(0, n, 512):
        rows = slice(start, start + 512)
        data = _batch(features[rows].astype(np.float32, copy=False),
                      experts[rows].astype(np.float32, copy=False),
                      midpoint[rows].astype(np.float32, copy=False))
        for name, xyz in data.items():
            result[name][rows] = xyz
    return result
