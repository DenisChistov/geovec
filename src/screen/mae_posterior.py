"""Absolute-distance decisions from the three V3 location posteriors and retrieval."""

import numpy as np
from pyproj import Geod

from posterior_ideas import _posterior_features, _frechet, _weights, _unit


R_KM = 6371.0088
GEOD = Geod(ellps="WGS84")
ONE_MINUS_E2 = (1 - 1 / 298.257223563) ** 2
BASELINE = "maepost:v4_rank_spherical_median"


def grid():
    """Return serializable, predeclared configurations; no labels enter these rules."""
    out = {BASELINE: {"kind": "baseline"}}
    def add(kind, label, **values):
        out[f"maepost:{kind}/{label}"] = dict(kind=kind, **values)
    for scheme in ("equal", "fine", "peer"):
        for power in (.70, 1., 1.40, 1.90):
            add("median", f"{scheme}/p{power:g}", scheme=scheme, power=power)
    for km in (80, 225, 500):
        for fraction in (.45, .8, 1.):
            add("basin", f"h{km}/w{fraction:g}", km=km, fraction=fraction)
    for candidates in ("centers", "nearby"):
        for power in (.75, 1., 1.5):
            add("medoid", f"{candidates}/p{power:g}", candidates=candidates, power=power)
    for scheme in ("ordinary", "sharp", "trust"):
        for fraction in (.08, .18, .32, .45):
            add("localatom", f"{scheme}/w{fraction:g}", scheme=scheme, fraction=fraction)
    for km in (75, 200, 500):
        for fraction in (.30, .60, .85):
            add("localgate", f"h{km}/w{fraction:g}", km=km, fraction=fraction)
    for share in (.35, .75, 1.):
        add("tail", f"w{share:g}", share=share)
    for km in (25, 75, 150):
        add("smooth", f"h{km}", km=km)
    for km in (80, 250, 600):
        add("likelihood", f"h{km}", km=km)
    add("adaptive", "entropy_and_agreement")
    add("ellipsoid", "weiszfeld")
    return out


GATE_EXPERTS = (
    BASELINE,
    "maepost:median/peer/p1.4",
    "maepost:basin/h225/w0.8",
    "maepost:localatom/sharp/w0.32",
    "maepost:localgate/h200/w0.6",
)


def _normalize(v, fallback):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-12, v / np.maximum(n, 1e-12), fallback)


def _prob(w):
    total = w.sum(axis=-1, keepdims=True)
    return np.where(total > 1e-14, w / np.maximum(total, 1e-14), 1 / w.shape[-1])


def _toward(a, b, weight):
    cosine = np.sum(a * b, axis=1, keepdims=True).clip(-1, 1)
    tangent = b - cosine * a
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    perpendicular = np.cross(a, np.eye(3)[np.argmin(abs(a), axis=1)])
    axis = _normalize(tangent, _normalize(perpendicular, a))
    step = np.arctan2(length, cosine) * np.asarray(weight).reshape(-1, 1)
    return _normalize(np.cos(step) * a + np.sin(step) * axis, a)


def _median(points, weights, anchor, smooth_km=4., initial=None):
    """Smoothed angular L1 Weiszfeld iterations, initialized by the resultant."""
    weights = _prob(weights)
    mean = initial if initial is not None else _normalize(np.einsum("nk,nki->ni", weights, points), anchor)
    floor = smooth_km / R_KM
    for _ in range(8):
        dot = np.einsum("nkc,nc->nk", points, mean).clip(-1, 1)
        sine = np.sqrt(np.maximum(0., (1 - dot) * (1 + dot)))
        angle = np.arctan2(sine, dot)
        inv = weights / np.sqrt(angle * angle + floor * floor)
        # Subtracting the radial component in float64 is important for points
        # that differ by less than a kilometre.
        scale = inv * angle / np.maximum(sine, 1e-10)
        step = np.einsum("nk,nkc->nc", scale, points)
        step -= np.sum(step * mean, axis=1, keepdims=True) * mean
        step /= np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
        size = np.linalg.norm(step, axis=1, keepdims=True)
        move = np.minimum(size, .70)
        mean = _normalize(np.cos(move) * mean + np.sin(move) * step / np.maximum(size, 1e-15), mean)
    return mean


def _lonlat(x):
    return (np.rad2deg(np.arctan2(x[..., 1], x[..., 0])),
            np.rad2deg(np.arctan2(x[..., 2], ONE_MINUS_E2 * np.hypot(x[..., 0], x[..., 1]))))


def _ellipsoidal_median(points, weights, initial):
    """Two local L1 descent steps with exact WGS84 inverse and direct geodesics."""
    lon, lat = _lonlat(initial)
    dest_lon, dest_lat = _lonlat(points)
    k = weights.shape[1]
    for _ in range(2):
        az, _, distance = GEOD.inv(np.repeat(lon, k), np.repeat(lat, k), dest_lon.ravel(), dest_lat.ravel())
        az = np.radians(az).reshape(weights.shape)
        distance = distance.reshape(weights.shape)
        inv = weights / np.hypot(distance, 1500.)
        slope = inv * distance
        east = (slope * np.sin(az)).sum(axis=1)
        north = (slope * np.cos(az)).sum(axis=1)
        step = np.hypot(east, north) / inv.sum(axis=1)
        lon, lat, _ = GEOD.fwd(lon, lat, np.rad2deg(np.arctan2(east, north)), step.clip(0, 600000.))
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    coords = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), ONE_MINUS_E2*np.sin(lat)))
    return _normalize(coords, initial)


def _batch(bundle, selected):
    prob_raw, xyz_raw, entropy, _, _, _ = _posterior_features(bundle)
    n = len(prob_raw)
    # Preserve the original dtype and reduction order for the historical reference.
    v4_points = xyz_raw.reshape(n, 30, 3)
    original = bundle["ensemble_multiscale"]
    anchor = _normalize(original.astype(np.float64), np.array([1., 0., 0.]))
    points = _normalize(v4_points.astype(np.float64), anchor[:, None])
    rank = prob_raw.astype(np.float64)
    groups = np.repeat(np.arange(3), 10)
    row = np.arange(n)
    ref = bundle["features"][:, 180:198]
    local = _normalize(bundle["experts"][:, 5].astype(np.float64), anchor)
    local_mean = _normalize(bundle["experts"][:, 4].astype(np.float64), anchor)
    # Explicit memoization is local to the slice. It avoids retaining old rows.
    cache = {}
    def cached(key, compute):
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    def angles():
        return cached("angles", lambda: np.arccos((points @ points.swapaxes(1, 2)).clip(-1, 1)))

    def cross_support():
        def compute():
            # The adaptive bandwidth allows for independently fitted cell centers.
            width = np.clip(2.2 * ref[:, 3], 65 / R_KM, 330 / R_KM)
            kernel = np.exp(-.5 * (angles() / width[:, None, None])**2)
            by_grid = np.einsum("nijk,njk->nij", kernel.reshape(n, 30, 3, 10), _prob(rank))
            return .5 * (by_grid.sum(axis=2) - by_grid[:, np.arange(30), groups]).clip(0)
        return cached("cross_support", compute)

    def weights(scheme="equal", power=1.):
        def compute():
            gr = _prob(rank ** power)
            if scheme == "fine":
                gr = gr * np.array([.25, .50, .25])[None, :, None]
            w = gr.reshape(n, 30)
            if scheme == "peer":
                w = w * np.sqrt(.055 + cross_support())
            return _prob(w)
        return cached(("weights", scheme, power), compute)

    def med(scheme="equal", power=1.):
        return cached(("med",scheme,power), lambda: _median(points, weights(scheme,power), anchor))

    def local_info():
        def compute():
            local_angle = np.arccos(np.einsum("nkc,nc->nk", points, local).clip(-1, 1))
            width = np.clip(2 * ref[:, 3], 100/R_KM, 350/R_KM)
            support = (np.exp(-.5*(local_angle/width[:,None])**2) * weights()).sum(axis=1)
            concentration = ref[:, 7].clip(0, 1)
            trust = np.sqrt(np.clip(support, 0, 1) * concentration)
            return local_angle, trust
        return cached("local_info", compute)

    def basin(km):
        def compute():
            kernel = np.exp(-.5 * (angles() / (km/R_KM))**2)
            w = weights()
            density = np.einsum("nij,nj->ni", kernel, w)
            seed = np.argmax(np.where(w > 0, density, -np.inf), axis=1)
            within = w * kernel[row, seed]
            # Use a local centroid inside a resolved basin to avoid pinning
            # the estimate to one quantized classifier center.
            center = _normalize(np.einsum("nk,nki->ni", within, points), points[row, seed])
            return center, density[row, seed]
        return cached(("basin",km), compute)

    def tail_atoms():
        from density_ideas import _atoms
        return cached("tail_atoms", lambda: _atoms(bundle["features"], bundle["experts"], bundle["ensemble_512"])[:2])

    result = {}
    for name, spec in selected.items():
        kind = spec["kind"]
        if kind == "baseline":
            w = _weights(prob_raw.reshape(n, 30))
            ans = _unit(_frechet(v4_points, w, original, median=True), original)
        elif kind == "median":
            ans = med(spec["scheme"], spec["power"])
        elif kind == "basin":
            center, mass = basin(spec["km"])
            confidence = np.clip(1.4 * mass, 0, 1)
            ans = _toward(med(), center, spec["fraction"] * confidence)
        elif kind == "medoid":
            w = weights("equal", spec["power"])
            risk = np.einsum("nij,nj->ni", angles(), w)
            risk = np.where(w > 0, risk, np.inf)
            actions = points
            if spec["candidates"] == "nearby":
                extra = np.stack((local, local_mean, basin(225)[0]), axis=1)
                gap = np.arccos((extra @ points.swapaxes(1,2)).clip(-1,1))
                risk = np.column_stack((risk, np.einsum("nij,nj->ni", gap, w)))
                actions = np.concatenate((actions, extra), axis=1)
            ans = actions[row, np.argmin(risk, axis=1)]
        elif kind == "localatom":
            power = 1.5 if spec["scheme"] == "sharp" else 1.
            fraction = np.full(n, spec["fraction"], dtype=np.float64)
            if spec["scheme"] == "trust":
                fraction *= local_info()[1]
            p = cached("local_atom_points", lambda: np.concatenate((points,local[:,None]), axis=1))
            w = np.column_stack((weights("equal", power) * (1 - fraction[:, None]), fraction))
            ans = _median(p, w, anchor, initial=med("equal", power))
        elif kind == "localgate":
            m = med()
            d = np.arccos(np.sum(m*local, axis=1).clip(-1,1))
            gate = np.exp(-.5*(d/(spec["km"]/R_KM))**2) * np.sqrt(local_info()[1])
            ans = _toward(m, local, spec["fraction"] * gate)
        elif kind == "tail":
            full_pts, mass = tail_atoms()
            mass = mass.copy()
            mass[:, :, -1] *= spec["share"]
            p = _normalize(full_pts.reshape(n,33,3).astype(np.float64), anchor[:,None])
            ans = _median(p, _prob(mass.astype(np.float64)).reshape(n,33), anchor, initial=med())
        elif kind == "smooth":
            ans = _median(points, weights(), anchor, smooth_km=spec["km"])
        elif kind == "likelihood":
            local_angle, trust = local_info()
            likelihood = np.exp(-.5*(local_angle/(spec["km"]/R_KM))**2)
            w = weights() * (1 + (1 + 4 * trust[:,None]) * likelihood)
            ans = _median(points, w, anchor, initial=med())
        elif kind == "adaptive":
            peer = cross_support()[:, ::10].mean(axis=1)
            power = np.clip(1 + 1.5 * peer - .8 * entropy.mean(axis=1), .65, 2.)
            w = _prob(rank ** power[:,None,None]).reshape(n,30)
            ans = _median(points, w, anchor)
        elif kind == "ellipsoid":
            ans = _ellipsoidal_median(points, weights(), med())
        else:
            raise ValueError(f"Unknown posterior rule {kind}")
        result[name] = np.asarray(ans, dtype=np.float32)
    return result


def predictions(bundle, only=None):
    """Return only requested deterministic rules, preserving order and row isolation."""
    spec = grid()
    names = list(spec) if only is None else [only] if isinstance(only, str) else list(dict.fromkeys(only))
    unknown = set(names).difference(spec)
    if unknown:
        raise KeyError(f"Unknown MAE posterior rules: {sorted(unknown)}")
    n = len(bundle["experts"])
    required = ["experts", "features", "ensemble_multiscale"]
    if any(spec[k]["kind"] == "tail" for k in names):
        required.append("ensemble_512")
    src = {k: np.asarray(bundle[k], dtype=np.float32) for k in required}
    if (src["experts"].shape != (n,7,3) or src["ensemble_multiscale"].shape != (n,3) or
            src["features"].ndim != 2 or src["features"].shape[0] != n or src["features"].shape[1] < 198):
        raise ValueError("Expected V3 expert and ranked-posterior features")
    selected = {name:spec[name] for name in names}
    result = {name: np.empty((n,3),dtype=np.float32) for name in names}
    if not names:
        return result
    for start in range(0, n, 768):
        rows = slice(start,start+768)
        q = _batch({k:v[rows] for k,v in src.items()}, selected)
        for name in names:
            result[name][rows] = q[name]
    return result
