"""Deterministic, Blender-independent diagnostics for sculptable surfaces.

The metrics in this module are deliberately descriptive rather than a
universal beauty score.  They give an agent enough signal to distinguish a
healthy dense base from an untouched ellipsoid, and to decide when another
inspection pass is warranted.  Inputs are plain coordinate/edge/normal
sequences so the same implementation can be used in tests and in Blender.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

Point = Sequence[float]
Edge = Sequence[int]


def _point(value: Point) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("points must have three coordinates")
    return (float(value[0]), float(value[1]), float(value[2]))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Iterable[float], fraction: float) -> float:
    """Linear percentile with deterministic behavior for sparse meshes."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = _clamp(fraction) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median(values: Iterable[float]) -> float:
    return _percentile(values, 0.5)


def _trimmed(values: Iterable[float], proportion: float = 0.1) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if len(ordered) < 3:
        return ordered
    cut = min(int(len(ordered) * max(0.0, proportion)), (len(ordered) - 1) // 2)
    return ordered[cut : len(ordered) - cut] or ordered


def _winsorized(values: Iterable[float], lower: float = 0.05, upper: float = 0.95) -> list[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    low = _percentile(values, lower)
    high = _percentile(values, upper)
    return [max(low, min(high, value)) for value in values]


def _mad(values: Iterable[float], center: Optional[float] = None) -> float:
    values = list(float(value) for value in values)
    if not values:
        return 0.0
    midpoint = _median(values) if center is None else float(center)
    return _median(abs(value - midpoint) for value in values)


def _std(values: Iterable[float], mean: Optional[float] = None) -> float:
    values = list(values)
    if not values:
        return 0.0
    center = _mean(values) if mean is None else float(mean)
    return math.sqrt(_mean((value - center) ** 2 for value in values))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_cv(values: Iterable[float]) -> float:
    values = list(values)
    mean = _mean(values)
    return _std(values, mean) / max(abs(mean), 1e-12)


def _bounds(points: Sequence[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    if not points:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return (
        [min(point[index] for point in points) for index in range(3)],
        [max(point[index] for point in points) for index in range(3)],
    )


def _distance(left: Point, right: Point) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _normal_variation(edges: Sequence[Edge], normals: Sequence[tuple[float, float, float]]) -> float:
    values = []
    for edge in edges:
        if len(edge) != 2:
            continue
        left, right = int(edge[0]), int(edge[1])
        if not (0 <= left < len(normals) and 0 <= right < len(normals)):
            continue
        first, second = normals[left], normals[right]
        first_len = math.sqrt(sum(value * value for value in first))
        second_len = math.sqrt(sum(value * value for value in second))
        if first_len < 1e-12 or second_len < 1e-12:
            continue
        cosine = sum(a * b for a, b in zip(first, second)) / (first_len * second_len)
        values.append(_clamp(1.0 - cosine, 0.0, 2.0))
    # A few discontinuous edges are useful evidence, but should not let one
    # malformed face dominate the detail signal for an otherwise smooth mesh.
    return _mean(_winsorized(values))


def _raw_normal_variation(edges: Sequence[Edge], normals: Sequence[tuple[float, float, float]]) -> float:
    values = []
    for edge in edges:
        if len(edge) != 2:
            continue
        left, right = int(edge[0]), int(edge[1])
        if not (0 <= left < len(normals) and 0 <= right < len(normals)):
            continue
        first, second = normals[left], normals[right]
        first_len = math.sqrt(sum(value * value for value in first))
        second_len = math.sqrt(sum(value * value for value in second))
        if first_len >= 1e-12 and second_len >= 1e-12:
            cosine = sum(a * b for a, b in zip(first, second)) / (first_len * second_len)
            values.append(_clamp(1.0 - cosine, 0.0, 2.0))
    return _mean(values)


def _symmetry_error(
    points: Sequence[tuple[float, float, float]],
    center: tuple[float, float, float],
    axes: Sequence[str],
    sample_limit: int,
) -> Mapping[str, float]:
    """Nearest-point mirror error, normalized by the bounding-box diagonal.

    A capped deterministic prefix keeps inspection bounded for production
    meshes.  The full point set remains the search set, so a high-resolution
    mesh still gives a useful answer even when only a sample is measured.
    """
    if not points or not axes:
        return {}
    low, high = _bounds(points)
    diagonal = max(_distance(low, high), 1e-12)
    count = max(1, min(int(sample_limit), len(points)))
    stride = max(1, len(points) // count)
    sample = [points[(index * stride) % len(points)] for index in range(count)]
    # Spatial hashing keeps nearest-mirror queries bounded for production
    # meshes.  A deterministic full-search fallback handles sparse samples.
    cell_size = diagonal / 64.0
    grid: Dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for candidate in points:
        key = tuple(int(math.floor((candidate[index] - low[index]) / cell_size)) for index in range(3))
        grid.setdefault(key, []).append(candidate)

    def nearest(reflected: list[float]) -> float:
        key = tuple(int(math.floor((reflected[index] - low[index]) / cell_size)) for index in range(3))
        candidates: list[tuple[float, float, float]] = []
        for radius in range(3):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        candidates.extend(grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
            if candidates:
                break
        if not candidates:
            candidates = points
        return min(_distance(reflected, candidate) for candidate in candidates)

    result: dict[str, float] = {}
    for axis_name in axes:
        axis = {"x": 0, "y": 1, "z": 2}.get(str(axis_name).lower())
        if axis is None:
            continue
        total = 0.0
        for point in sample:
            reflected = list(point)
            reflected[axis] = 2.0 * center[axis] - reflected[axis]
            total += nearest(reflected)
        error = total / max(len(sample), 1) / diagonal
        result[str(axis_name).lower()] = _clamp(error, 0.0, 1.0)
    return result


def sculpt_quality_metrics(
    vertices: Sequence[Point],
    *,
    edges: Sequence[Edge] = (),
    normals: Sequence[Point] = (),
    symmetry_axes: Sequence[str] = (),
    sample_limit: int = 512,
) -> dict:
    """Return bounded geometric diagnostics for a mesh surface.

    ``ellipsoid_likeness`` is intentionally a warning signal: a high value
    means the sampled radial profile has little relief and should prompt a
    blockout pass, not that the model is invalid.  ``detail_signal`` combines
    radial relief, normal variation and edge-length regularity into one
    diagnostic signal; it must be interpreted with topology and visual review,
    not used as an isolated reward or rejection threshold.
    """
    points = [_point(value) for value in (vertices or ())]
    parsed_edges = list(edges or ())
    parsed_normals = [_point(value) for value in (normals or ())]
    low, high = _bounds(points)
    dimensions = [max(high[index] - low[index], 1e-12) for index in range(3)]
    diagonal = math.sqrt(sum(value * value for value in dimensions))
    center = tuple((low[index] + high[index]) * 0.5 for index in range(3))
    half = tuple(max(dimensions[index] * 0.5, 1e-12) for index in range(3))
    radii = [
        math.sqrt(sum(((point[index] - center[index]) / half[index]) ** 2 for index in range(3)))
        for point in points
    ]
    radius_mean = _mean(radii)
    raw_radius_std = _std(radii, radius_mean)
    raw_radial_residual = _mean(abs(value - 1.0) for value in radii)

    # Fit the profile from central coordinate quantiles once the mesh is dense
    # enough for a percentile to be meaningful.  A single spike then cannot
    # move the center/scale of the entire object.  Tiny meshes retain the old
    # bounding-box fit: their sparse, often synthetic samples cannot support
    # a defensible robust estimate.
    if len(points) >= 20:
        robust_low = [_percentile((point[index] for point in points), 0.02) for index in range(3)]
        robust_high = [_percentile((point[index] for point in points), 0.98) for index in range(3)]
        robust_center = tuple((robust_low[index] + robust_high[index]) * 0.5 for index in range(3))
        robust_half = [max((robust_high[index] - robust_low[index]) * 0.5, 1e-12) for index in range(3)]
        robust_fit = "central_quantile_02_98"
    else:
        robust_center = center
        robust_half = list(half)
        robust_fit = "bounds_sparse_fallback"
    robust_radii = [
        math.sqrt(sum(((point[index] - robust_center[index]) / robust_half[index]) ** 2 for index in range(3)))
        for point in points
    ]
    robust_radius_mean = _mean(robust_radii)
    trimmed_radii = _trimmed(robust_radii)
    winsor_radii = _winsorized(robust_radii)
    radius_mad = _mad(robust_radii)
    trimmed_radius_std = _std(trimmed_radii, _mean(trimmed_radii))
    robust_radial_residual = _mean(_winsorized(
        [abs(value - 1.0) for value in robust_radii]
    ))
    radial_p10 = _percentile(robust_radii, 0.10)
    radial_p90 = _percentile(robust_radii, 0.90)

    edge_lengths = []
    for edge in parsed_edges:
        if len(edge) != 2:
            continue
        left, right = int(edge[0]), int(edge[1])
        if 0 <= left < len(points) and 0 <= right < len(points):
            edge_lengths.append(_distance(points[left], points[right]))
    normal_variation = _normal_variation(parsed_edges, parsed_normals) if parsed_normals else 0.0
    raw_normal_variation = _raw_normal_variation(parsed_edges, parsed_normals) if parsed_normals else 0.0
    edge_cv = _safe_cv(_winsorized(edge_lengths))
    radial_relief = _clamp(
        0.60 * trimmed_radius_std * 4.0
        + 0.40 * max(0.0, radial_p90 - radial_p10) * 1.5
    )
    normal_relief = _clamp(normal_variation * 3.0)
    # A regular base is desirable; a wildly varying edge length is a signal
    # of stretched topology, not detail.  Keep this term diagnostic only.
    topology_irregularity = _clamp(edge_cv / 2.0)
    detail_signal = _clamp(0.55 * radial_relief + 0.35 * normal_relief - 0.10 * topology_irregularity)
    ellipsoid_likeness = _clamp(
        1.0 - (robust_radial_residual * 4.0 + radial_relief * 0.35 + normal_relief * 0.15)
    )
    symmetry = _symmetry_error(points, center, list(symmetry_axes or ()), sample_limit)

    return {
        "vertices": len(points),
        "edges": len(parsed_edges),
        "bounds": {"min": [round(value, 8) for value in low], "max": [round(value, 8) for value in high]},
        "dimensions": [round(value, 8) for value in dimensions],
        "diagonal": round(diagonal, 8),
        "radial_profile": {
            "mean": round(radius_mean, 8),
            "std": round(raw_radius_std, 8),
            "residual_mean": round(raw_radial_residual, 8),
            "ellipsoid_likeness": round(ellipsoid_likeness, 8),
            "robust": {
                "center": [round(value, 8) for value in robust_center],
                "half_extent": [round(value, 8) for value in robust_half],
                "fit": robust_fit,
                "mean": round(robust_radius_mean, 8),
                "median": round(_median(robust_radii), 8),
                "mad": round(radius_mad, 8),
                "p10": round(radial_p10, 8),
                "p90": round(radial_p90, 8),
                "trimmed_std": round(trimmed_radius_std, 8),
                "winsorized_std": round(_std(winsor_radii, _mean(winsor_radii)), 8),
                "residual_mean": round(robust_radial_residual, 8),
            },
            "raw": {
                "mean": round(radius_mean, 8),
                "std": round(raw_radius_std, 8),
                "residual_mean": round(raw_radial_residual, 8),
            },
            "diagnostic_only": True,
        },
        "surface": {
            "normal_variation": round(normal_variation, 8),
            "raw_normal_variation": round(raw_normal_variation, 8),
            "edge_length_cv": round(edge_cv, 8),
            "topology_irregularity": round(topology_irregularity, 8),
            "detail_signal": round(detail_signal, 8),
            "diagnostic_only": True,
        },
        "symmetry": {
            "axes": sorted(symmetry),
            "mean_error": round(_mean(symmetry.values()), 8),
            "score": round(1.0 - _mean(symmetry.values()), 8) if symmetry else None,
            "sample_limit": max(1, min(int(sample_limit), len(points))) if points else 0,
        },
    }


__all__ = ["sculpt_quality_metrics"]
