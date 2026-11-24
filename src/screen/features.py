"""Ten fold-local feature pipelines for paired 512-D GeoCLIP and 768-D StreetCLIP.

prepare yields (name, (train_features, query_features)) in FEATURES order.
Each matrix is a contiguous float32 NumPy array. Original-width pipelines
split at 512; all other pipelines have two equal-width output blocks.
"""
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.nn import functional as F


FEATURES = {
    "raw_blocks": "Раздельная L2-нормировка исходных GeoCLIP и StreetCLIP.",
    "signed_sqrt": "Квадратный корень из модуля каждой координаты с сохранением знака и раздельной нормировкой.",
    "balanced_pca": "Отдельное центрирование и одинаковое число главных компонент для двух энкодеров.",
    "shrink_whiten": "Раздельные главные компоненты с регуляризованным отбеливанием и нормировкой.",
    "geo_projection": "Географически направленные проекции по связи с мягкими геоячейками плюс обычные компоненты каждого энкодера.",
    "fourier_rbf": "Случайные признаки Фурье радиального ядра с шириной по обучающим расстояниям для каждого энкодера.",
    "cross_bilinear": "Низкоранговые произведения двух независимых проекций модальностей вместе с их основными компонентами.",
    "canonical_pair": "Согласованные пары координат двух энкодеров из регуляризованного канонического корреляционного анализа.",
    "prototype_metric": "Стандартизованные угловые расстояния до прототипов двух отдельно обученных сферических кластеризаций.",
    "sparse_jl": "Две независимые разреженные знаковые проекции Джонсона — Линденштраусса.",
}

_VIEWS = (slice(0, 512), slice(512, 1280))
_SAMPLE = 12288
_BATCH = 4096


def _check_vectors(a, name):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 1280 or not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"{name} must be a floating-point [rows, 1280] matrix")
    for start in range(0, len(a), _BATCH):
        block = a[start:start + _BATCH]
        if not np.isfinite(block).all():
            raise ValueError(f"{name} contains non-finite values")
        for view in _VIEWS:
            if np.any(np.max(np.abs(block[:, view]), axis=1) <= 0):
                raise ValueError(f"{name} contains a zero view")
    return a


def _orient(p):
    columns = torch.arange(p.shape[1], device=p.device)
    signs = torch.sign(p[p.abs().argmax(dim=0), columns])
    return torch.where(signs[None, :] < 0, -p, p)


def _pca(x, cap=192):
    n, d = x.shape
    k = min(n - 1, d, cap)
    # A thin decomposition avoids a large covariance solve for small smoke tests.
    if n < d:
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        return _orient(vh[:k].T), s[:k].square() / max(n - 1, 1)
    cov = (x.T @ x) / (n - 1)
    values, axes = torch.linalg.eigh((cov + cov.T) * .5)
    return _orient(axes[:, -k:].flip(1)), values[-k:].flip(0).clamp_min(0)


def _encode_targets(y):
    # Equal-area fixed landmarks put the same geographic scale on each fold.
    j = torch.arange(96, dtype=y.dtype, device=y.device)
    z = (2 * j + 1) / len(j) - 1
    angle = j * (np.pi * (3 - np.sqrt(5)))
    horiz = (1 - z.square()).sqrt()
    centers = torch.stack((horiz * angle.cos(), horiz * angle.sin(), z), dim=1)
    return torch.softmax(14 * (F.normalize(y, dim=1) @ centers.T), dim=1)


class _Fold:
    def __init__(self, train, labels, seed):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.train = train
        self.labels = labels

    def _rng(self, salt):
        return np.random.default_rng(np.random.SeedSequence([self.seed % (2**32), salt]))

    def tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @cached_property
    def ids(self):
        return self._rng(1).choice(len(self.train), min(_SAMPLE, len(self.train)), replace=False)

    @cached_property
    def sample(self):
        selected = self.tensor(self.train[self.ids])
        return [F.normalize(selected[:, view], dim=1) for view in _VIEWS]

    @cached_property
    def means(self):
        return [x.mean(dim=0) for x in self.sample]

    @cached_property
    def components(self):
        return [_pca(x - m) for x, m in zip(self.sample, self.means)]

    @cached_property
    def whitening(self):
        # A ridge proportional to leading variance keeps small eigenvalues stable.
        return [p * (v + (.035 * v[0]).clamp_min(1e-8)).rsqrt() for p, v in self.components]

    @torch.inference_mode()
    def transformation(self, name):
        norm = lambda a: F.normalize(a, dim=1)
        join = lambda blocks: torch.cat([norm(a) for a in blocks], dim=1)
        if name == "raw_blocks":
            return lambda xs: torch.cat(xs, dim=1)
        if name == "signed_sqrt":
            return lambda xs: join([x.sign() * x.abs().sqrt() for x in xs])
        if name == "sparse_jl":
            maps = []
            rng = self._rng(10)
            for width in (512, 768):
                p = np.zeros((width, 192), np.float32)
                coordinate = np.arange(width)
                for band in range(3):
                    ids = rng.integers(64, size=width) + 64 * band
                    p[coordinate, ids] = rng.choice([-1, 1], size=width) / np.sqrt(3)
                maps.append(self.tensor(p))
            return lambda xs: join([x @ p for x, p in zip(xs, maps)])
        if name == "fourier_rbf":
            maps, phases = [], []
            rng = self._rng(6)
            for x in self.sample:
                indices = rng.integers(len(x), size=(4096, 2))
                distance = (x[indices[:, 0]] - x[indices[:, 1]]).norm(dim=1)
                sigma = distance.median().clamp_min(.03)
                maps.append(self.tensor(rng.standard_normal((x.shape[1], 256)).astype("f")) / sigma)
                phases.append(self.tensor(rng.uniform(0, 2 * np.pi, 256).astype("f")))
            return lambda xs: join([(x @ p + b).cos() for x, p, b in zip(xs, maps, phases)])
        if name == "prototype_metric":
            prototypes = []
            for x in self.sample:
                k = min(96, len(x))
                c = x[:k].clone()
                for _ in range(5):
                    idx = (x @ c.T).argmax(dim=1)
                    sums = torch.zeros_like(c).index_add_(0, idx, x)
                    c = torch.where(sums.norm(dim=1, keepdim=True) > 1e-7, norm(sums), c)
                distances = (x @ c.T).clamp(-1, 1).acos()
                prototypes.append((c, distances.mean(0), distances.std(0).clamp_min(.025)))
            return lambda xs: join([((x @ c.T).clamp(-1, 1).acos() - mu) / scale
                                    for x, (c, mu, scale) in zip(xs, prototypes)])

        means = self.means
        axes = [pair[0] for pair in self.components]
        if name == "balanced_pca":
            return lambda xs: join([(x - m) @ p for x, m, p in zip(xs, means, axes)])
        if name == "shrink_whiten":
            white = self.whitening
            return lambda xs: join([(x - m) @ p for x, m, p in zip(xs, means, white)])
        if name == "canonical_pair":
            white = self.whitening
            a, b = [(x - m) @ p for x, m, p in zip(self.sample, means, white)]
            u, _, vt = torch.linalg.svd((a.T @ b) / len(a), full_matrices=False)
            k = min(128, u.shape[1])
            maps = [white[0] @ u[:, :k], white[1] @ vt[:k].T]
            return lambda xs: join([(x - m) @ p for x, m, p in zip(xs, means, maps)])
        if name == "geo_projection":
            y = _encode_targets(self.tensor(self.labels[self.ids]))
            y = y - y.mean(dim=0)
            maps = []
            for x, m, p in zip(self.sample, means, axes):
                cross = (x - m).T @ y / len(x)
                u, _, _ = torch.linalg.svd(cross, full_matrices=False)
                k = min(48, len(x) - 1)
                maps.append(torch.cat((p[:, :96], _orient(u[:, :k])), dim=1))
            return lambda xs: join([(x - m) @ p for x, m, p in zip(xs, means, maps)])
        if name == "cross_bilinear":
            rng = self._rng(7)
            maps = [self.tensor(rng.standard_normal((x.shape[1], 192)).astype("f")) for x in self.sample]

            def interact(xs):
                projected = [(x - m) @ p for x, m, p in zip(xs, means, maps)]
                halves = (projected[0] * projected[1]).chunk(2, dim=1)
                blocks = [torch.cat((norm((x - m) @ p[:, :64]), norm(h)), dim=1)
                          for x, m, p, h in zip(xs, means, axes, halves)]
                return join(blocks)

            return interact
        raise KeyError(name)


class _ViewTransform(torch.nn.Module):
    """Include the original block normalization in the deployed mapping."""

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, batch):
        return self.transform([F.normalize(batch[:, view], dim=1) for view in _VIEWS])


@torch.inference_mode()
def _save_view(transform, name, device, save_dir):
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    example = torch.ones((2, 1280), dtype=torch.float32, device=device)
    mapping = torch.jit.trace(transform.eval(), (example,), check_trace=False)
    with NamedTemporaryFile(dir=root, prefix=f".feature-{name}-", suffix=".pt", delete=False) as f:
        pending = Path(f.name)
    try:
        torch.jit.save(mapping, str(pending))
        pending.replace(root / f"feature-{name}.pt")
    finally:
        pending.unlink(missing_ok=True)


@torch.inference_mode()
def _matrix(values, device, transform, width=None):
    if not len(values):
        if width is None:
            probe = torch.empty((0, 1280), dtype=torch.float32, device=device)
            width = transform(probe).shape[1]
        return np.empty((0, width), dtype=np.float32)
    out = None
    for begin in range(0, len(values), _BATCH):
        rows = slice(begin, min(begin + _BATCH, len(values)))
        block = np.ascontiguousarray(values[rows], dtype=np.float32)
        batch = torch.as_tensor(block, dtype=torch.float32, device=device)
        result = transform(batch).cpu().numpy()
        if out is None:
            out = np.empty((len(values), result.shape[1]), dtype=np.float32)
        out[rows] = result
    return out


def prepare(X_train, Y_train, X_query, seed, only=None, save_dir=None):
    """Yield fold-fitted features and optionally save each selected mapping."""
    train = _check_vectors(X_train, "X_train")
    query = _check_vectors(X_query, "X_query")
    labels = np.asarray(Y_train)
    if (len(train) < 2 or labels.shape != (len(train), 3) or not np.isfinite(labels).all()
            or np.any(np.linalg.norm(labels, axis=1) <= 1e-8)):
        raise ValueError("Y_train must contain one nonzero ECEF XYZ direction per training row")
    names = set(FEATURES) if only is None else {only} if isinstance(only, str) else set(only)
    if not names or not names.issubset(FEATURES):
        raise ValueError(f"Unknown or empty feature selection: {names}")
    fold = _Fold(train, labels, seed)
    for name in FEATURES:
        if name in names:
            fn = _ViewTransform(fold.transformation(name))
            if save_dir is not None:
                _save_view(fn, name, fold.device, save_dir)
            train_features = _matrix(train, fold.device, fn)
            query_features = _matrix(query, fold.device, fn, train_features.shape[1])
            yield name, (train_features, query_features)
            del train_features, query_features, fn


def transform_saved(X_query, name, save_dir):
    """Transform new embeddings using only the saved training-fold mapping."""
    if name not in FEATURES:
        raise ValueError(f"Unknown feature view: {name}")
    query = _check_vectors(X_query, "X_query")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = torch.jit.load(str(Path(save_dir) / f"feature-{name}.pt"), map_location=device)
    return _matrix(query, device, mapping.eval())
