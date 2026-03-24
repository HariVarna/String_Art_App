import numpy as np


def _circular_distance(a, b, count):
    forward = abs(a - b)
    return min(forward, count - forward)


def _line_pixel_indices(start, end, image_size):
    steps = int(max(abs(end[0] - start[0]), abs(end[1] - start[1]))) + 1
    xs = np.rint(np.linspace(start[0], end[0], steps)).astype(np.int16)
    ys = np.rint(np.linspace(start[1], end[1], steps)).astype(np.int16)
    return np.unique(ys.astype(np.int32) * image_size + xs.astype(np.int32))


def build_line_cache(nails, image_size):
    line_cache = [[None] * len(nails) for _ in range(len(nails))]

    for start_index, start in enumerate(nails):
        for end_index in range(start_index + 1, len(nails)):
            indices = _line_pixel_indices(start, nails[end_index], image_size)
            line_cache[start_index][end_index] = indices
            line_cache[end_index][start_index] = indices

    return line_cache


def _line_improvement_score(residual_values, line_weight):
    remaining = np.maximum(residual_values - line_weight, 0.0)
    improvement = residual_values * residual_values - remaining * remaining
    return float(improvement.mean() / max(line_weight, 1.0))


def pick_best_nail(
    residual,
    line_cache,
    current,
    line_weight,
    min_gap=10,
    recent_nails=(),
):
    nail_count = len(line_cache)
    best_score = -1.0
    best_index = current

    for candidate in range(nail_count):
        if candidate == current:
            continue

        if _circular_distance(candidate, current, nail_count) <= min_gap:
            continue

        line_pixels = line_cache[current][candidate]
        if line_pixels is None or line_pixels.size == 0:
            continue

        score = _line_improvement_score(residual[line_pixels], line_weight)

        if candidate in recent_nails:
            score *= 0.88

        if score > best_score:
            best_score = score
            best_index = candidate

    return best_index, best_score


def apply_line_to_residual(residual, line_pixels, line_weight):
    residual[line_pixels] = np.maximum(residual[line_pixels] - line_weight, 0.0)
