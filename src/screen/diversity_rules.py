"""Corrections from complementary experts around a fixed WGS84 consensus."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from strong_rules import _GEOD, _RADIUS, _direct, _inverse, _lonlat, _to_ecef, _unit


RULES = {
    "diverse:signed": "Остаточные направления дополнительных экспертов после учёта четырёх основных, с усиленной регуляризацией.",
    "diverse:context": "Поправка дополнительных экспертов зависит от согласия с соседями и величины отклонения от основной смеси.",
    "diverse:convex": "Выпуклая смесь с ограничением суммарного веса дополнительных экспертов и уточнением по геодезическому риску WGS84.",
    "diverse:tailguard": "Согласованная поправка дополнительных экспертов, усиление которой зависит от прогнозируемого риска крупного промаха.",
}


def _geometry(experts):
    e = _unit(experts, 3)
    if not 4 <= e.shape[1] <= 7:
        raise ValueError("Expected V4, meta, paper, retrieval and up to three complementary experts")
    v4 = _lonlat(e[:, 0])
    anchor_step = .30 * _inverse(v4, _lonlat(e[:, 1])) + .15 * _inverse(v4, _lonlat(e[:, 2]))
    origin = _direct(v4, anchor_step)
    d = np.stack([_inverse(origin, _lonlat(e[:, j])) for j in range(e.shape[1])], axis=-1)
    return origin, d


def _residualize(d, state, fit):
    core, weak = d[:, :, :4], d[:, :, 4:]
    if fit:
        x = core.reshape(-1, 4)
        gram = x.T @ x
        penalty = .03 * max(np.mean(np.diag(gram)), len(d) * 1e-6)
        state["projection"] = np.linalg.solve(gram + penalty * np.eye(4), x.T @ weak.reshape(len(x), -1))
    return core, weak - core @ state["projection"]


def _ridge(x, y, strengths):
    n = len(x)
    flat = x.reshape(-1, x.shape[-1])
    gram = flat.T @ flat
    # Penalize high-amplitude, noisy directions at their observed scale.
    scale = np.maximum(np.diag(gram), max(n * 1e-5, .05 * np.diag(gram).mean()))
    return np.linalg.solve(gram + np.diag(scale * strengths), flat.T @ y.ravel())


def _signals(d):
    sizes = np.linalg.norm(d, axis=1)
    weak = d[:, :, 4:]
    unit_weak = weak / np.maximum(sizes[:, None, 4:], 1e-8)
    guide = d[:, :, 3] / np.maximum(sizes[:, 3:4], 1e-8)
    other = unit_weak.sum(axis=2)[:, :, None] - unit_weak + guide[:, :, None]
    alignment = (unit_weak * other).sum(axis=1) / np.maximum(np.linalg.norm(other, axis=1), 1e-8)
    core_spread = np.sqrt(np.mean(sizes[:, :4]**2, axis=1))
    excess = np.log1p(sizes[:, 4:] / (.025 + core_spread[:, None]))
    return alignment, excess, core_spread, sizes


def _linear_design(d, name, state, fit):
    core, residual = _residualize(d, state, fit)
    parts = [core, residual]
    if name == "diverse:context":
        agreement, excess, _, _ = _signals(d)
        cov = np.stack([agreement, excess], axis=1)
        if fit:
            state["center"] = cov.mean(axis=0)
            state["scale"] = np.maximum(cov.std(axis=0), .15)
        z = np.clip((cov - state["center"]) / state["scale"], -2.5, 2.5)
        parts.extend([residual * z[:, j, None, :] for j in range(2)])
    return np.concatenate(parts, axis=2)


def _fit_convex(d, y, origin, target, seed):
    n, _, p = d.shape
    x = d.reshape(-1, p)
    gram = x.T @ x / n
    cross = x.T @ y.ravel() / n
    prior = np.r_[.04, .05, .04, .02, np.full(p - 4, .02 / max(1, p - 4))]
    penalty = np.maximum(np.diag(gram), 1e-5) * np.r_[np.full(4, .035), np.full(p - 4, .35)]
    scale = max(np.mean(np.diag(gram)), 1e-5)
    sample = np.random.default_rng(seed).choice(n, min(9000, n), replace=False)
    dx, dy = d[sample], y[sample]
    here = tuple(v[sample] for v in origin)
    there = tuple(v[sample] for v in target)

    def objective(w, exact):
        step = dx @ w
        cost = w @ gram @ w - 2 * cross @ w + penalty @ (w - prior)**2
        if exact:
            dist = _GEOD.inv(*_direct(here, step), *there)[2] / _RADIUS
            cost += np.mean(dist**2 - np.sum((dy - step)**2, axis=1))
        return cost / scale

    # Remaining weight belongs to the fixed strong anchor. All component
    # weights are nonnegative and the complementary group receives at most 25%.
    constraints = [{"type": "ineq", "fun": lambda w: 1 - w.sum()},
                   {"type": "ineq", "fun": lambda w: .25 - w[4:].sum()}]
    best = prior
    for exact in (False, True):
        opt = minimize(objective, best, args=(exact,), method="SLSQP", bounds=[(0, 1)] * p,
                       constraints=constraints, options={"maxiter": 30, "ftol": 1e-8, "eps": 1e-5})
        if opt.success and opt.fun <= objective(best, exact) + 1e-9:
            best = opt.x
    return best


def _guard(d):
    size = np.linalg.norm(d, axis=1)
    spread = np.sqrt(np.mean(size[:, :4]**2, axis=1))
    # Limit each weak displacement before pooling, so one distant location
    # cannot create a strong directional vote. The retrieval champion also
    # votes, so the identical rule reduces naturally to its vote at ablation.
    weak = d[:, :, 3:]
    length = size[:, 3:]
    units = weak / np.maximum(length[:, None, :], 1e-8)
    specialists = d[:, :, 1:3] / np.maximum(size[:, None, 1:3], 1e-8)
    peers = units.sum(axis=2)[:, :, None] - units + specialists.sum(axis=2)[:, :, None]
    agreement = (units * peers).sum(axis=1) / np.maximum(np.linalg.norm(peers, axis=1), 1e-8)
    excess = np.log1p(length / (.025 + spread[:, None]))
    limit = np.clip(.04 + 1.7 * spread, .04, .65)[:, None]
    bounded = weak * np.minimum(1, limit / np.maximum(length, 1e-8))[:, None, :]
    support = .12 + .88 * ((agreement.clip(-1, 1) + 1) / 2)**2
    proposal = np.mean(bounded * support[:, None, :], axis=2)
    features = np.column_stack([np.log1p(20 * spread), excess.mean(axis=1),
                                agreement.mean(axis=1), np.log1p(20 * np.linalg.norm(proposal, axis=1))])
    return proposal, features


def _tail_fit(d, y, state):
    proposal, features = _guard(d)
    state["center"] = features.mean(0)
    state["scale"] = np.maximum(features.std(0), .1)
    x = np.column_stack([np.ones(len(d)), np.clip((features - state["center"]) / state["scale"], -4, 4)])
    error = np.linalg.norm(y, axis=1)
    threshold = max(np.quantile(error, .80), .035)
    label = (error > threshold).astype(float)
    prevalence = np.clip(label.mean(), .005, .995)
    prior = np.r_[np.log(prevalence / (1 - prevalence)), np.zeros(x.shape[1] - 1)]
    regularization = np.r_[.012, np.full(x.shape[1] - 1, .055)]

    def loss(w):
        z = x @ w
        delta = w - prior
        value = np.mean(np.logaddexp(0, z) - label*z) + .5 * np.sum(regularization * delta**2)
        grad = x.T @ (expit(z) - label) / len(x) + regularization * delta
        return value, grad

    fit = minimize(loss, prior, jac=True, method="L-BFGS-B", options={"maxiter": 35})
    state["tail_coef"] = fit.x if fit.success else prior
    state["tail_threshold_rad"] = np.asarray(threshold)
    p = expit(x @ state["tail_coef"])
    # Interpolate typical/tail gains. A modest prior preserves an explicit
    # exploratory vote even where the uncertain tail supplies little support.
    design = np.stack([proposal * (1-p)[:, None], proposal * p[:, None]], axis=2).reshape(-1, 2)
    gram = design.T @ design
    penalty = .18 * max(gram.trace() / 2, len(d)*1e-6)
    gains = np.linalg.solve(gram + np.eye(2)*penalty,
                            design.T @ y.ravel() + penalty*np.array([.12, .32]))
    state["gain"] = np.clip(gains, .025, .85)


def fit_rule(name, train, y, seed=0):
    """Fit strictly on training predictions and aligned ECEF truth."""
    if name not in RULES:
        raise KeyError(name)
    origin, d = _geometry(train)
    truth = _unit(y, 2)
    if not len(d) or len(d) != len(truth):
        raise ValueError("Missing or misaligned training truth")
    target = _lonlat(truth)
    offset = _inverse(origin, target)
    p = d.shape[2]
    state = {"expert_count": np.asarray(p)}
    if name == "diverse:convex":
        state["coef"] = _fit_convex(d, offset, origin, target, seed)
    elif name == "diverse:tailguard":
        _tail_fit(d, offset, state)
    else:
        design = _linear_design(d, name, state, True)
        strength = np.r_[np.full(4, .08), np.full(p - 4, .32),
                         np.full(design.shape[2] - p, .65)]
        state["coef"] = _ridge(design, offset, strength)
    return state


def apply_rule(name, state, query):
    """Predict unit ECEF directions without labels."""
    if name not in RULES:
        raise KeyError(name)
    origin, d = _geometry(query)
    if int(state["expert_count"]) != d.shape[2]:
        raise ValueError("The checkpoint was fitted with a different expert count")
    if not len(d):
        return np.empty((0, 3), np.float32)
    if name == "diverse:convex":
        step = d @ state["coef"]
    elif name == "diverse:tailguard":
        proposal, features = _guard(d)
        x = np.column_stack([np.ones(len(d)), np.clip((features-state["center"]) / state["scale"], -4, 4)])
        p = expit(x @ state["tail_coef"])
        step = proposal * ((1 - p) * state["gain"][0] + p * state["gain"][1])[:, None]
    else:
        design = _linear_design(d, name, state, False)
        step = design @ state["coef"]
        limit = np.maximum(.12, 1.2 * np.max(np.linalg.norm(d[:, :, :4], axis=1), axis=1))
        step *= np.minimum(1., np.minimum(limit, np.pi) / np.maximum(np.linalg.norm(step, axis=1), 1e-10))[:, None]
    return _to_ecef(_direct(origin, step))
