"""Training-fold ECEF geometry and WGS84 targets shared by screen models."""

import numpy as np
from pyproj import Geod
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree


_ELLIPSOID = Geod(ellps="WGS84")
_ONE_MINUS_E2 = (1.0 - 1.0 / 298.257223563) ** 2


def _center(points, weights):
    mean = np.average(points, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-10 else points[np.argmax(weights)].copy()


def _balanced(points, weights, size):
    """Build a weighted principal-axis tree with indivisible identical labels."""
    labels = np.empty(len(points), dtype=np.int64)
    centers = []
    coarse_centers = []
    coarse_depth = min(3, size.bit_length() - 1)

    def split(rows, leaves, depth):
        values, mass = points[rows], weights[rows]
        center = _center(values, mass)
        if depth == coarse_depth:
            coarse_centers.append(center)
        if leaves == 1:
            labels[rows] = len(centers)
            centers.append(center)
            return
        # Cartesian PCA remains defined for broad regions crossing the antipode.
        residual = values - np.average(values, axis=0, weights=mass)
        covariance = (residual.T * mass) @ residual
        _, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        order = np.argsort(values @ axis, kind="stable")
        counts = mass[order].cumsum()
        half = leaves // 2
        cuts = np.arange(half, len(rows) - half + 1)
        cut = cuts[np.argmin(np.abs(2 * counts[cuts - 1] - counts[-1]))]
        split(rows[order[:cut]], half, depth + 1)
        split(rows[order[cut:]], half, depth + 1)

    split(np.arange(len(points)), size, 0)
    return np.array(centers), labels, np.array(coarse_centers)


def _spherical_lloyd(points, counts, seeds, seed):
    # A fixed perturbation from an observed member allows the Lloyd solution
    # to differ from balanced partitions while keeping all centres on support.
    centers = seeds.copy()
    rng = np.random.default_rng(seed)
    tree = cKDTree(points)
    _, nearby = tree.query(centers, k=min(4, len(points)))
    nearby = np.asarray(nearby).reshape(len(centers), -1)
    chosen = nearby[np.arange(len(centers)), rng.integers(nearby.shape[1], size=len(centers))]
    centers += .08 * points[chosen]
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    previous = None
    for _ in range(16):
        labels = cKDTree(centers).query(points)[1]
        occupied, compact = np.unique(labels, return_inverse=True)
        compact = compact.reshape(-1)
        if previous is not None and np.array_equal(previous, compact):
            break
        previous = compact
        sums = np.column_stack([np.bincount(compact, weights=counts * points[:, d],
                                            minlength=len(occupied)) for d in range(3)])
        norm = np.linalg.norm(sums, axis=1, keepdims=True)
        centers = np.where(norm > 1e-10, sums / np.maximum(norm, 1e-10), centers[occupied])
    labels = cKDTree(centers).query(points)[1]
    occupied, labels = np.unique(labels, return_inverse=True)
    return centers[occupied], labels.reshape(-1).astype(np.int64)


def _coords(points):
    x, y, z = points.T
    return np.column_stack((np.degrees(np.arctan2(y, x)),
                            np.degrees(np.arctan2(z, _ONE_MINUS_E2 * np.hypot(x, y)))))


def _distances(first, second):
    lon1, lon2 = np.broadcast_arrays(first[..., 0], second[..., 0])
    lat1, lat2 = np.broadcast_arrays(first[..., 1], second[..., 1])
    _, _, metres = _ELLIPSOID.inv(lon1.ravel(), lat1.ravel(), lon2.ravel(), lat2.ravel())
    return np.asarray(metres).reshape(lon1.shape) / 1000.0


def _nearby_targets(points, ll, centers, center_ll):
    width = min(8, len(centers))
    ids = np.empty((len(points), width), dtype=np.int64)
    weights = np.empty((len(points), width), dtype=np.float32)
    tree = cKDTree(centers)
    candidates = min(len(centers), 2 * width)
    for start in range(0, len(points), 4096):
        rows = slice(start, start + 4096)
        _, nearest = tree.query(points[rows], k=np.arange(1, candidates + 1))
        distances = _distances(ll[rows, None], center_ll[nearest])
        order = np.argsort(distances, axis=1)[:, :width]
        ids[rows] = np.take_along_axis(nearest, order, axis=1)
        km = np.take_along_axis(distances, order, axis=1)
        logit = -.5 * (km / 110.) ** 2
        values = np.exp(logit - logit.max(axis=1, keepdims=True))
        weights[rows] = values / values.sum(axis=1, keepdims=True)
    return ids, weights


def _spectral(distances):
    n = len(distances)
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    k = min(8, n - 1)
    near = np.argsort(distances, axis=1)[:, :k + 1]
    scale = np.maximum(10., np.take_along_axis(distances, near[:, -1:], axis=1)[:, 0])
    rows = np.arange(n)[:, None]
    graph = np.zeros_like(distances)
    graph[rows, near] = 1
    mst_row, mst_col = minimum_spanning_tree(distances).nonzero()
    graph[mst_row, mst_col] = 1
    graph = np.maximum(graph, graph.T)
    graph *= np.exp(-np.minimum(20., distances ** 2 / (scale[:, None] * scale[None, :])))
    np.fill_diagonal(graph, 0)
    root_degree = np.maximum(graph.sum(axis=1), 1e-12) ** .5
    normalized = graph / (root_degree[:, None] * root_degree[None, :])
    _, eigenvectors = np.linalg.eigh(np.eye(n) - normalized)
    features = eigenvectors[:, 1:min(n, 25)] / root_degree[:, None]
    features -= features.mean(axis=0, keepdims=True)
    features /= np.maximum(features.std(axis=0, keepdims=True), 1e-12)
    return features.astype(np.float32)


def _triangles(points, centers, nearest):
    # A locally inverse-chord fallback is defined also for collinear or tiny
    # reference sets and for points outside a hemisphere's training coverage.
    closest = nearest[:, :3]
    delta = np.linalg.norm(points[:, None, :] - centers[closest], axis=2)
    exact = delta[:, 0] < 1e-11
    values = 1.0 / np.maximum(delta, 1e-11)
    values /= values.sum(axis=1, keepdims=True)
    values[exact] = 0
    values[exact, 0] = 1
    ids = np.pad(closest, ((0, 0), (0, 3 - closest.shape[1])), mode="edge")
    weights = np.pad(values, ((0, 0), (0, 3 - closest.shape[1])))
    if len(centers) < 3:
        return ids, weights.astype(np.float32)

    if len(centers) >= 4:
        try:
            hull = ConvexHull(centers)
        except QhullError:
            return ids, weights.astype(np.float32)
        # Each candidate face bounds the ray exit by t <= -b / (n dot x).
        outer = hull.equations[:, 3] < -1e-10
        faces = hull.simplices[outer]
        plane = hull.equations[outer, :3] / -hull.equations[outer, 3:4]
    else:
        faces = np.array([[0, 1, 2]])
        plane = np.zeros((1, 3))
    matrix = centers[faces].transpose(0, 2, 1)
    good = np.abs(np.linalg.det(matrix)) > 1e-12
    if not good.any():
        return ids, weights.astype(np.float32)
    faces, plane = faces[good], plane[good]
    inverse = np.linalg.inv(matrix[good])
    for start in range(0, len(points), 1024):
        stop = min(len(points), start + 1024)
        p = points[start:stop]
        face = np.argmax(p @ plane.T, axis=1)
        raw = (inverse[face] @ p[..., None])[..., 0]
        valid = np.all(raw >= -1e-7, axis=1) & (raw.sum(axis=1) > 1e-8)
        row_ids = np.arange(start, stop)[valid]
        bary = np.maximum(raw[valid], 0)
        weights[row_ids] = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-15)
        ids[row_ids] = faces[face[valid]]
    return ids, weights.astype(np.float32)


def prepare(Y_train, seed):
    """Prepare per-training-row targets, sharing identical geographic rows."""
    y = np.asarray(Y_train, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 3 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Expected a nonempty finite [N,3] array of ECEF directions")
    norm = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("ECEF directions must be nonzero")
    y = (y / norm).astype(np.float32)
    unique, inverse, counts = np.unique(y, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    points = unique.astype(np.float64)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    size = 2 ** int(round(np.log2(np.clip(len(points) / 16, 1, 256))))
    balanced, blabels, mixture = _balanced(points, counts, size)
    centers, labels = _spherical_lloyd(points, counts, balanced, seed)
    prior = np.bincount(labels, weights=counts, minlength=len(centers))
    dot = np.clip(np.sum(points * centers[labels], axis=1), -1, 1)
    angle = np.arctan2(np.linalg.norm(np.cross(points, centers[labels]), axis=1), dot)
    var = np.bincount(labels, weights=counts * angle ** 2, minlength=len(centers)) / prior
    true_ll, center_ll = _coords(points), _coords(centers)
    km = _distances(center_ll[:, None], center_ll[None, :])
    distance_ids, distance_weights = _nearby_targets(points, true_ll, centers, center_ll)
    tri_ids, tri_weights = _triangles(points, centers, distance_ids)

    landmarks = [int(np.argmax(prior))]
    while len(landmarks) < min(8, len(centers)):
        nearest = km[:, landmarks].min(axis=1)
        nearest[landmarks] = -1
        landmarks.append(int(np.argmax(nearest)))
    thresholds = np.array([250., 1000., 3500., 9000.])
    ring_codes = (km[:, landmarks, None] <= thresholds).reshape(len(centers), -1)
    ring_targets = np.empty((len(points), ring_codes.shape[1]), dtype=np.float32)
    for start in range(0, len(points), 4096):
        rows = slice(start, start + 4096)
        distance = _distances(true_ll[rows, None], center_ll[None, landmarks])
        ring_targets[rows] = (distance[..., None] <= thresholds).reshape(len(distance), -1)
    return {
        "y": y, "balanced_centers": balanced.astype(np.float32),
        "balanced_labels": blabels[inverse], "centers": centers.astype(np.float32),
        "labels": labels[inverse], "prior": (prior / prior.sum()).astype(np.float32),
        "distance_ids": distance_ids[inverse], "distance_weights": distance_weights[inverse],
        "risk_table": ((km / 4000.) ** 2).astype(np.float32),
        "ring_codes": ring_codes.astype(np.float32), "ring_targets": ring_targets[inverse],
        "spectral_codes": _spectral(km), "radius": np.maximum(.01, np.sqrt(var)).astype(np.float32),
        "mixture_centers": mixture.astype(np.float32),
        "tri_ids": tri_ids[inverse], "tri_weights": tri_weights[inverse],
    }
