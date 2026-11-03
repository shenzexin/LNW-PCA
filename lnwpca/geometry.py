"""Weighted rigid alignment and correspondence refinement."""
import numpy as np


def points(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 3:
        raise ValueError('Expected at least three XYZ points.')
    if not np.isfinite(value).all():
        raise ValueError('Point coordinates must be finite.')
    return value


def transform(source, matrix):
    return np.asarray(source) @ matrix[:3, :3].T + matrix[:3, 3]


def nearest(source, target, chunk=256):
    # Chunking bounds memory for the dependency-free implementation.
    indices, distances = [], []
    for start in range(0, len(source), chunk):
        squared = ((source[start:start + chunk, None] - target[None]) ** 2).sum(-1)
        index = squared.argmin(1)
        indices.append(index)
        distances.append(np.sqrt(squared[np.arange(len(index)), index]))
    return np.concatenate(distances), np.concatenate(indices)


def rigid_fit(source, target, weights=None):
    source, target = points(source), points(target)
    if source.shape != target.shape:
        raise ValueError('Correspondence arrays must have identical shapes.')
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(source),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Weights must be finite, nonnegative, and point-wise.')
    if weights.sum() <= 1e-8:
        raise ValueError('Weights have insufficient total mass.')
    weights = weights / weights.sum()
    p, q = weights @ source, weights @ target
    cross = ((source - p) * weights[:, None]).T @ (target - q)
    if np.linalg.matrix_rank(cross) < 2:
        raise ValueError('Correspondences do not constrain a rigid rotation.')
    u, _, vt = np.linalg.svd(cross)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, q - rotation @ p
    return result


def features(source, k=20):
    source = points(source)
    k = min(k, len(source) - 1)
    rows, graph = [], []
    for i, point in enumerate(source):
        distance = np.linalg.norm(source - point, axis=1)
        distance[i] = np.inf
        index = np.argsort(distance)[:k]
        local = source[index] - source[index].mean(0)
        values, vectors = np.linalg.eigh(local.T @ local / k)
        normal = vectors[:, 0]
        if normal @ (point - source.mean(0)) < 0:
            normal = -normal
        rows.append(np.r_[point, normal, values[0] / max(values.sum(), 1e-12),
                          1 / max(distance[index].mean(), 1e-12)])
        graph.append(index)
    return np.asarray(rows, dtype=np.float32), np.asarray(graph)


def refine(source, target, initial=None, weights=None, policy=None,
           threshold=0.03, steps=10, tolerance=1e-4):
    source, target = points(source), points(target)
    matrix = np.eye(4) if initial is None else np.asarray(initial, dtype=float).copy()
    base = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    density = features(source)[0][:, -1]
    diagonal = np.linalg.norm(np.ptp(target, axis=0))
    history, stable, previous = [], 0, np.inf
    for _ in range(steps):
        moved = transform(source, matrix)
        residual, index = nearest(moved, target)
        valid = residual < threshold
        if valid.sum() < 3:
            break
        reverse, _ = nearest(target, moved)
        state = np.array([residual.mean(), residual.max(), valid.mean(),
                          0.5 * (valid.mean() + (reverse < threshold).mean()),
                          density.var(), 1 - valid.mean()], dtype=np.float32)
        alpha, beta = (1., 0.) if policy is None else policy(state)
        adaptive = np.clip(alpha * base + beta, 0, 1)
        if adaptive[valid].sum() <= 1e-8:
            adaptive = base.copy()
        delta = rigid_fit(moved[valid], target[index[valid]], adaptive[valid])
        matrix = delta @ matrix
        error, _ = nearest(transform(source, matrix), target)
        mean = error.mean()
        history.append(float(mean))
        stable = stable + 1 if abs(previous - mean) < tolerance * diagonal else 0
        previous = mean
        if stable >= 2:
            break
    return matrix, history
