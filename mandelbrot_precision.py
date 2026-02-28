import numpy as np


def _sample_escape(x0, y0, max_iter, dtype):
    x0 = dtype(x0)
    y0 = dtype(y0)
    x = dtype(0.0)
    y = dtype(0.0)

    for iteration in range(max_iter):
        xx = x * x
        yy = y * y
        if xx + yy > dtype(4.0):
            return iteration, float(x), float(y)
        xtemp = xx - yy + x0
        y = dtype(2.0) * x * y + y0
        x = xtemp

    return max_iter, float(x), float(y)


def choose_precision(WIDTH, HEIGHT, max_iter, x_min, x_max, y_min, y_max):
    x_span = abs(float(x_max) - float(x_min))
    y_span = abs(float(y_max) - float(y_min))
    dx = x_span / max(WIDTH - 1, 1)
    dy = y_span / max(HEIGHT - 1, 1)
    span = max(x_span, y_span)
    coord_scale = max(abs(float(x_min)), abs(float(x_max)), abs(float(y_min)), abs(float(y_max)), 1.0)

    # First gate: if the pixel spacing approaches float32 ULP scale,
    # neighboring pixels will alias onto the same coordinates.
    if min(dx, dy) <= 32.0 * np.finfo(np.float32).eps * coord_scale:
        return "float64"

    # Second gate: compare a few representative sample orbits in float32 and
    # float64. Small windows and/or high iteration counts near the boundary tend
    # to diverge here before a full render diverges badly.
    sample_budget = min(int(max_iter), 128)
    xs = [float(x_min), 0.5 * (float(x_min) + float(x_max)), float(x_max)]
    ys = [float(y_min), 0.5 * (float(y_min) + float(y_max)), float(y_max)]

    for x0 in xs:
        for y0 in ys:
            iter32, zx32, zy32 = _sample_escape(x0, y0, sample_budget, np.float32)
            iter64, zx64, zy64 = _sample_escape(x0, y0, sample_budget, np.float64)
            if iter32 != iter64:
                return "float64"

            orbit_scale = max(abs(zx64), abs(zy64), span, 1.0)
            orbit_tol = 128.0 * np.finfo(np.float32).eps * orbit_scale
            if abs(zx32 - zx64) > orbit_tol or abs(zy32 - zy64) > orbit_tol:
                return "float64"

    return "float32"
