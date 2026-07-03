"""Largest-Triangle-Three-Buckets downsampling (X5) — shrink a learning curve for display/export.

A finished run can hold thousands of metric frames; an XLSX sheet with a native Excel chart wants
~500–1 000 points per series, not the full history (the CSV export stays full-resolution — this is
only for the workbook's curve tables + charts). LTTB is the standard visually-lossless line
downsampler: it keeps the first and last point and, for each of ``threshold`` buckets in between,
picks the point that forms the **largest-area triangle** with the previously-kept point and the mean
of the next bucket — so peaks, dips and the overall shape survive where a naive stride would clip them.

Pure: no I/O, no global state. ``x`` values are the canonical axis (env_steps or wall_clock); points
whose ``y`` is ``None`` are dropped first (a gap in the reward series), so the caller can pass a raw
score list straight in.
"""

from collections.abc import Sequence


def downsample(
    xs: Sequence[float], ys: Sequence[float | None], threshold: int
) -> list[tuple[float, float]]:
    """Downsample ``(x, y)`` to at most ``threshold`` points, preserving the curve's shape (LTTB).

    Drops points with a ``None`` ``y`` first. Returns the surviving points unchanged when there are
    already ``≤ threshold`` of them (or ``threshold < 3`` — LTTB needs the two endpoints + ≥1 bucket).
    ``xs`` and ``ys`` must be the same length.
    """
    pts = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True) if y is not None]
    n = len(pts)
    if threshold < 3 or n <= threshold:
        return pts

    sampled: list[tuple[float, float]] = [pts[0]]  # always keep the first point
    # Bucket size over the interior points (endpoints excluded), as a float for even spacing.
    every = (n - 2) / (threshold - 2)
    a = 0  # index of the last point we kept — the triangle's fixed vertex

    for i in range(threshold - 2):
        # The next bucket's average point (the triangle's third, forward-looking vertex).
        start = int((i + 1) * every) + 1
        end = int((i + 2) * every) + 1
        end = min(end, n)
        avg_x = sum(p[0] for p in pts[start:end]) / (end - start)
        avg_y = sum(p[1] for p in pts[start:end]) / (end - start)

        # This bucket's candidate points — pick the one making the largest-area triangle with a→avg.
        cur_start = int(i * every) + 1
        cur_end = int((i + 1) * every) + 1
        ax, ay = pts[a]
        best_area = -1.0
        best = cur_start
        for j in range(cur_start, cur_end):
            px, py = pts[j]
            area = abs((ax - avg_x) * (py - ay) - (ax - px) * (avg_y - ay)) * 0.5
            if area > best_area:
                best_area = area
                best = j
        sampled.append(pts[best])
        a = best

    sampled.append(pts[-1])  # always keep the last point
    return sampled
