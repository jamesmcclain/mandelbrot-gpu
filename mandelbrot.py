#!/usr/bin/env python3

import argparse
import os

import numpy as np
from PIL import Image

BACKENDS = {
    "cuda": ("mandelbrot_cuda", "mandelbrot.ptx"),
    "opencl": ("mandelbrot_opencl", "mandelbrot.cl"),
    "amdhsa": ("mandelbrot_amdhsa", "mandelbrot.hsaco"),
}

# python3 mandelbrot.py
# python3 mandelbrot.py --views seahorse_tail:-0.7613:-0.7257:0.1214:0.1414:2048:ice

# ─────────────────────────────────────────────────────────────────────────────
# Famous Mandelbrot views
#
# Each entry is a tuple:
#   (name, slug, x_min, x_max, y_min, y_max, recommended_max_iter, theme)
#
# Coordinates chosen to look great at float32 precision (not excessively deep).
# ─────────────────────────────────────────────────────────────────────────────
FAMOUS_VIEWS = [
    {
        "name": "Overview",
        "slug": "overview",
        "x_min": -2.9722,
        "x_max": 1.4722,
        "y_min": -1.25,
        "y_max": 1.25,
        "max_iter": 512,
        "theme": "classic",
        "desc": "The full Mandelbrot set — the iconic cardioid and primary bulb",
    },
    {
        "name": "Seahorse Valley",
        "slug": "seahorse_valley",
        "x_min": -0.7828,
        "x_max": -0.6832,
        "y_min": 0.092,
        "y_max": 0.148,
        "max_iter": 1024,
        "theme": "ice",
        "desc": "The classic 'Seahorse Valley' between the main cardioid and period-2 bulb",
    },
    {
        "name": "Elephant Valley",
        "slug": "elephant_valley",
        "x_min": 0.1994,
        "x_max": 0.4306,
        "y_min": -0.065,
        "y_max": 0.065,
        "max_iter": 1024,
        "theme": "fire",
        "desc": "The 'Elephant Valley' on the right side — elephants marching in a row",
    },
    {
        "name": "Double Spiral",
        "slug": "double_spiral",
        "x_min": -0.7801,
        "x_max": -0.7249,
        "y_min": 0.1000,
        "y_max": 0.1310,
        "max_iter": 2048,
        "theme": "emacs",
        "desc": "A stunning double-spiral inside Seahorse Valley",
    },
    {
        "name": "Starfish",
        "slug": "starfish",
        "x_min": -0.5122,
        "x_max": -0.3878,
        "y_min": 0.540,
        "y_max": 0.610,
        "max_iter": 1024,
        "theme": "rainbow",
        "desc": "The 'Starfish' — a five-armed spiral off the main body",
    },
    {
        "name": "Triple Spiral",
        "slug": "triple_spiral",
        "x_min": -0.0984,
        "x_max": -0.0576,
        "y_min": 0.6490,
        "y_max": 0.6720,
        "max_iter": 2048,
        "theme": "classic",
        "desc": "A triple spiral near the upper filaments",
    },
    {
        "name": "Mini Mandelbrot",
        "slug": "mini_mandelbrot",
        "x_min": -1.8082,
        "x_max": -1.7548,
        "y_min": -0.0150,
        "y_max": 0.0150,
        "max_iter": 2048,
        "theme": "emacs",
        "desc": "A self-similar 'mini-brot' on the real axis to the left of the main set",
    },
    {
        "name": "Feather",
        "slug": "feather",
        "x_min": -0.7538,
        "x_max": -0.7422,
        "y_min": 0.0920,
        "y_max": 0.0985,
        "max_iter": 2048,
        "theme": "ice",
        "desc": "A delicate feather-shaped region inside Seahorse Valley",
    },
    {
        "name": "Spiral Tendril",
        "slug": "spiral_tendril",
        "x_min": -1.6450,
        "x_max": -1.5810,
        "y_min": -0.0180,
        "y_max": 0.0180,
        "max_iter": 1024,
        "theme": "fire",
        "desc": "Spiralling tendrils on the tip of the left antenna",
    },
    {
        "name": "Quad Spiral",
        "slug": "quad_spiral",
        "x_min": -0.1658,
        "x_max": -0.0982,
        "y_min": 0.8430,
        "y_max": 0.8810,
        "max_iter": 2048,
        "theme": "rainbow",
        "desc": "A four-armed spiral galaxy in the upper filaments",
    },
]

# Build a lookup dict by slug for CLI use
_VIEWS_BY_SLUG = {v["slug"]: v for v in FAMOUS_VIEWS}

# ─────────────────────────────────────────────────────────────────────────────


def parse_arguments():
    slug_list = ", ".join(v["slug"] for v in FAMOUS_VIEWS)

    # yapf: disable
    parser = argparse.ArgumentParser(
        description='Generate famous Mandelbrot set images using a GPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available view slugs:\n  {slug_list}\n\n"
               "Custom view format:  name:x_min:x_max:y_min:y_max[:max_iter[:theme]]\n"
               "  Example: myview:-0.76:-0.70:0.09:0.15:1024:ice\n"
    )
    parser.add_argument('--views', '-v', nargs='+', metavar='SLUG_OR_CUSTOM', help='Views to render: slug names from the built-in list, or custom specs (see below). Defaults to all 10 built-in views.')
    parser.add_argument('--list-views', action='store_true',help='Print all built-in views and exit')
    parser.add_argument('--width',    type=int, default=3840, help='Image width in pixels (default: 3840)')
    parser.add_argument('--height',   type=int, default=2160, help='Image height in pixels (default: 2160)')
    parser.add_argument('--max-iter', type=int, default=None, help='Override max iterations for all views')
    parser.add_argument('--theme',    type=str, default=None, choices=['grayscale', 'emacs', 'fire', 'ice', 'rainbow', 'classic'], help='Override color theme for all views')
    parser.add_argument('--output-dir', '-d', type=str, default='.', help='Directory to write output PNGs (default: current dir)')
    parser.add_argument('--backend', '-b', type=str, default='cuda', choices=['cuda', 'opencl', 'amdhsa'], help='GPU backend to use (default: cuda)')
    parser.add_argument('--precision', type=str, default='auto', choices=['single', 'double', 'auto'], help='Floating-point precision to use: single, double, or auto (default: auto)')
    # yapf: enable
    return parser.parse_args()


def parse_custom_view(spec):
    """Parse a custom view spec: name:x_min:x_max:y_min:y_max[:max_iter[:theme]]"""
    parts = spec.split(':')
    if len(parts) < 5:
        raise argparse.ArgumentTypeError(f"Custom view '{spec}' needs at least name:x_min:x_max:y_min:y_max")
    name = parts[0]
    x_min = float(parts[1])
    x_max = float(parts[2])
    y_min = float(parts[3])
    y_max = float(parts[4])
    max_iter = int(parts[5]) if len(parts) > 5 else 1024
    theme = parts[6] if len(parts) > 6 else 'classic'
    slug = name.lower().replace(' ', '_')
    return {
        "name": name,
        "slug": slug,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "max_iter": max_iter,
        "theme": theme,
        "desc": "Custom view",
    }


def resolve_views(view_specs):
    """Resolve a list of slug names or custom specs to view dicts."""
    result = []
    for spec in view_specs:
        if spec in _VIEWS_BY_SLUG:
            result.append(_VIEWS_BY_SLUG[spec])
        else:
            result.append(parse_custom_view(spec))
    return result


def apply_color_theme(data, theme):
    """
    Apply a color theme to grayscale Mandelbrot data.

    Args:
        data: 2D numpy array with values 0-255
        theme: Color theme name

    Returns:
        RGB image as numpy array (or grayscale for grayscale theme)
    """
    if theme == 'grayscale':
        return data

    height, width = data.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    normalized = data.astype(np.float32) / 255.0

    if theme == 'emacs':
        stops = [
            (0.0, (0, 0, 0)),
            (0.0156, (0, 0, 51)),
            (0.0312, (0, 0, 102)),
            (0.0625, (0, 0, 204)),
            (0.125, (0, 102, 255)),
            (0.25, (0, 204, 255)),
            (0.375, (0, 255, 204)),
            (0.5, (204, 255, 0)),
            (0.625, (255, 204, 0)),
            (0.75, (255, 102, 0)),
            (0.875, (255, 51, 0)),
            (1.0, (204, 0, 0)),
        ]
        for i in range(len(stops) - 1):
            pos1, color1 = stops[i]
            pos2, color2 = stops[i + 1]
            mask = (normalized >= pos1) & (normalized < pos2)
            if np.any(mask):
                t = (normalized[mask] - pos1) / (pos2 - pos1)
                rgb[mask, 0] = (color1[0] + t * (color2[0] - color1[0])).astype(np.uint8)
                rgb[mask, 1] = (color1[1] + t * (color2[1] - color1[1])).astype(np.uint8)
                rgb[mask, 2] = (color1[2] + t * (color2[2] - color1[2])).astype(np.uint8)
        mask = normalized >= 1.0
        if np.any(mask):
            rgb[mask] = stops[-1][1]

    elif theme == 'fire':
        rgb[:, :, 0] = np.minimum(255, normalized * 512).astype(np.uint8)
        rgb[:, :, 1] = np.maximum(0, (normalized - 0.5) * 512).astype(np.uint8)
        rgb[:, :, 2] = np.maximum(0, (normalized - 0.75) * 1024).astype(np.uint8)

    elif theme == 'ice':
        rgb[:, :, 2] = np.minimum(255, normalized * 512).astype(np.uint8)
        rgb[:, :, 1] = np.maximum(0, (normalized - 0.5) * 512).astype(np.uint8)
        rgb[:, :, 0] = np.maximum(0, (normalized - 0.75) * 1024).astype(np.uint8)

    elif theme == 'rainbow':
        hue = normalized * 6.0
        h_i = hue.astype(np.int32) % 6
        f = hue - h_i
        p = np.zeros_like(normalized)
        q = normalized * (1 - f)
        t = normalized * f
        v = normalized
        rgb[:, :, 0] = np.where(h_i == 0, v * 255, np.where(h_i == 1, q * 255, np.where(h_i == 2, p * 255, np.where(h_i == 3, p * 255, np.where(h_i == 4, t * 255,
                                                                                                                                                v * 255))))).astype(np.uint8)
        rgb[:, :, 1] = np.where(h_i == 0, t * 255, np.where(h_i == 1, v * 255, np.where(h_i == 2, v * 255, np.where(h_i == 3, q * 255, np.where(h_i == 4, p * 255,
                                                                                                                                                p * 255))))).astype(np.uint8)
        rgb[:, :, 2] = np.where(h_i == 0, p * 255, np.where(h_i == 1, p * 255, np.where(h_i == 2, t * 255, np.where(h_i == 3, v * 255, np.where(h_i == 4, v * 255,
                                                                                                                                                q * 255))))).astype(np.uint8)

    elif theme == 'classic':
        rgb[:, :, 0] = np.minimum(255, normalized * 400).astype(np.uint8)
        rgb[:, :, 1] = np.maximum(0, (normalized - 0.6) * 640).astype(np.uint8)
        rgb[:, :, 2] = np.minimum(255, normalized * 600).astype(np.uint8)

    return rgb


def render_view(view, kernel, WIDTH, HEIGHT, max_iter_override, theme_override, output_dir, backend, precision):
    max_iter = max_iter_override if max_iter_override is not None else view["max_iter"]
    theme = theme_override if theme_override is not None else view["theme"]

    precision_map = {
        'single': 'float32',
        'double': 'float64',
        'auto': 'auto',
    }
    runtime_precision = precision_map[precision]

    result = backend.run_kernel(kernel, WIDTH, HEIGHT, max_iter, view["x_min"], view["x_max"], view["y_min"], view["y_max"], precision=runtime_precision)

    # Logarithmic normalisation to 0-255
    normalized = np.zeros_like(result, dtype=np.uint8)
    in_set = result >= max_iter
    escaped = ~in_set
    normalized[in_set] = 0
    if np.any(escaped):
        log_iter = np.log(result[escaped] + 1)
        log_max = np.log(max_iter + 1)
        normalized[escaped] = (255 * log_iter / log_max).astype(np.uint8)

    colored = apply_color_theme(normalized, theme)

    filename = f"mandelbrot_{view['slug']}.png"
    filepath = os.path.join(output_dir, filename)

    if theme == 'grayscale':
        img = Image.fromarray(colored, mode='L')
    else:
        img = Image.fromarray(colored, mode='RGB')

    img.save(filepath)

    print(f"  [{view['name']}]  →  {filepath}")
    print(f"    {view['desc']}")
    print(f"    x=[{view['x_min']}, {view['x_max']}]  y=[{view['y_min']}, {view['y_max']}]"
          f"  max_iter={max_iter}  theme={theme}")
    print(f"    iter range: {result.min()}–{result.max()}")
    print()

    return filepath


def main():
    args = parse_arguments()

    if args.list_views:
        print(f"{'Slug':<22} {'Name':<20} {'Theme':<10} Description")
        print("─" * 90)
        for v in FAMOUS_VIEWS:
            print(f"{v['slug']:<22} {v['name']:<20} {v['theme']:<10} {v['desc']}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which views to render
    if args.views:
        views = resolve_views(args.views)
    else:
        views = FAMOUS_VIEWS  # all 10 by default

    WIDTH = args.width
    HEIGHT = args.height

    import importlib
    module_name, compiled_file = BACKENDS[args.backend]
    backend = importlib.import_module(module_name)

    print(f"Rendering {len(views)} view(s) at {WIDTH}×{HEIGHT} px\n")

    kernel = backend.load_kernel(compiled_file)

    for view in views:
        render_view(view, kernel, WIDTH, HEIGHT, args.max_iter, args.theme, args.output_dir, backend, args.precision)

    print(f"Done. {len(views)} image(s) written to '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
