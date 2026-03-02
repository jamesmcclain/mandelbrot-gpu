#!/usr/bin/env python3
"""
animation.py — Render a smooth zoom animation between two Mandelbrot views.

Uses the same GPU backends as mandelbrot.py and streams each rendered frame
directly into an imageio/FFmpeg video pipeline — no per-frame disk I/O.

Examples
--------
# Zoom from the full overview into Seahorse Valley, 120 frames at 240p:
  python3 animation.py --start overview --end seahorse_valley \
          --frames 120 --resolution 240p --theme ice

# Custom start → end, lossless FFV1 codec:
  python3 animation.py \
      --start "custom:-2.5:1.0:-1.25:1.25:512:classic" \
      --end   "custom:-0.7828:-0.6832:0.092:0.148:1024:ice" \
      --frames 300 --fps 30 --resolution 1080p \
      --codec ffv1 --output zoom.mkv
"""

import argparse
import importlib
import math
import sys

import imageio
import numpy as np
from tqdm import tqdm

# ── re-use the view catalogue and helpers from mandelbrot.py ─────────────────
from mandelbrot import (_VIEWS_BY_SLUG, BACKENDS, FAMOUS_VIEWS, apply_color_theme, parse_custom_view)

# ── resolution presets ────────────────────────────────────────────────────────
RESOLUTION_PRESETS = {
    "240p": (426, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}


# ── view interpolation ────────────────────────────────────────────────────────
def interpolate_view(start, end, t):
    """
    Linearly interpolate the four corners and max_iter at normalised time
    t ∈ [0, 1].  Each frame's viewport is exactly lerp(start_corner, end_corner, t).
    """

    def lerp(a, b):
        return a + t * (b - a)

    return dict(
        name="interp",
        slug="interp",
        x_min=lerp(start["x_min"], end["x_min"]),
        x_max=lerp(start["x_max"], end["x_max"]),
        y_min=lerp(start["y_min"], end["y_min"]),
        y_max=lerp(start["y_max"], end["y_max"]),
        max_iter=round(lerp(start["max_iter"], end["max_iter"])),
        theme=start["theme"],
        desc="Interpolated frame",
    )


# ── single-frame rendering ────────────────────────────────────────────────────
def render_frame_rgb(view, kernel, width, height, theme, backend, precision):
    """Return an (H, W, 3) uint8 RGB array for *view*."""
    precision_map = {"single": "float32", "double": "float64", "auto": "auto"}
    result = backend.run_kernel(
        kernel,
        width,
        height,
        view["max_iter"],
        view["x_min"],
        view["x_max"],
        view["y_min"],
        view["y_max"],
        precision=precision_map[precision],
    )

    max_iter = view["max_iter"]
    normalized = np.zeros_like(result, dtype=np.uint8)
    escaped = result < max_iter
    if np.any(escaped):
        log_iter = np.log(result[escaped].astype(np.float32) + 1.0)
        log_max = math.log(max_iter + 1.0)
        normalized[escaped] = (255.0 * log_iter / log_max).astype(np.uint8)

    colored = apply_color_theme(normalized, theme)

    # Guarantee (H, W, 3) — even for the grayscale theme
    if colored.ndim == 2:
        colored = np.stack([colored] * 3, axis=-1)

    return colored


# ── resolution helper ─────────────────────────────────────────────────────────
def _parse_resolution(s):
    if s in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[s]
    if "x" in s.lower():
        w, h = s.lower().split("x", 1)
        return int(w), int(h)
    raise argparse.ArgumentTypeError(f"Unknown resolution '{s}'.  Use a preset "
                                     f"({', '.join(RESOLUTION_PRESETS)}) or WxH (e.g. 1280x720).")


def _resolve_view(spec):
    if spec in _VIEWS_BY_SLUG:
        return _VIEWS_BY_SLUG[spec]
    return parse_custom_view(spec)


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_arguments():
    slug_list = ", ".join(v["slug"] for v in FAMOUS_VIEWS)
    res_list = ", ".join(RESOLUTION_PRESETS.keys())

    # yapf: disable
    parser = argparse.ArgumentParser(
        description="Render a Mandelbrot zoom animation between two views.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(f"Built-in view slugs:\n  {slug_list}\n\n"
                "Custom view format:  name:x_min:x_max:y_min:y_max[:max_iter[:theme]]\n"
                "  Example: myview:-0.76:-0.70:0.09:0.15:1024:ice\n\n"
                f"Resolution presets: {res_list}\n"
                "  Or specify WxH directly, e.g. 1280x720\n"),
    )
    parser.add_argument("--start", "-s", required=True, metavar="SLUG_OR_CUSTOM", help="Starting view (slug or custom spec)")
    parser.add_argument("--end", "-e", required=True, metavar="SLUG_OR_CUSTOM", help="Ending view (slug or custom spec)")
    parser.add_argument("--frames", "-n", type=int, default=60, help="Total number of frames (default: 60)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--resolution", "-r", default="1440p", metavar="PRESET_OR_WxH", help=f"Output resolution (default: 1440p).  Presets: {res_list}")
    parser.add_argument("--theme", "-t", default=None, choices=["grayscale", "emacs", "fire", "ice", "rainbow", "classic"], help="Colour theme (overrides the start-view theme)")
    parser.add_argument("--backend", "-b", default="cuda", choices=list(BACKENDS.keys()), help="GPU backend (default: cuda)")
    parser.add_argument("--precision", default="auto", choices=["single", "double", "auto"], help="Floating-point precision (default: auto)")
    parser.add_argument("--codec", "-c", default="h264", choices=["h264", "ffv1"], help="Video codec: h264 (default) or ffv1 (always lossless, requires .mkv)")
    parser.add_argument("--lossless", action="store_true", help="Force lossless h264 (CRF=0); ffv1 is always lossless")
    parser.add_argument("--output", "-o", default="animation.mp4", help="Output video file (default: animation.mp4)")
    return parser.parse_args()
    # yapf: enable


# ── video writer ──────────────────────────────────────────────────────────────
def _open_writer(path, fps, codec, lossless):
    """
    Return an (imageio writer, resolved path) pair.

    imageio streams raw RGB frames into an FFmpeg subprocess, so only the
    current frame lives in memory uncompressed at any given time.
    """
    if codec == "ffv1" and not path.endswith(".mkv"):
        print("[warning] FFV1 needs a Matroska container — "
              "renaming output to .mkv", file=sys.stderr)
        path = path.rsplit(".", 1)[0] + ".mkv"

    if codec == "h264":
        crf = "0" if lossless else "18"
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=None,
            ffmpeg_params=["-crf", crf, "-preset", "fast"],
            macro_block_size=2,
        )
    else:  # ffv1
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="ffv1",
            quality=None,
        )

    return writer, path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()

    start_view = _resolve_view(args.start)
    end_view = _resolve_view(args.end)
    theme = args.theme or start_view["theme"]

    width, height = _parse_resolution(args.resolution)
    # H.264 requires even dimensions
    width += width % 2
    height += height % 2

    module_name, compiled_file = BACKENDS[args.backend]
    backend = importlib.import_module(module_name)
    kernel = backend.load_kernel(compiled_file)

    writer, output_path = _open_writer(args.output, args.fps, args.codec, args.lossless)

    n_frames = args.frames

    print(f"Rendering {n_frames} frame(s) at {width}×{height}  "
          f"({args.fps} fps · {args.codec} · theme={theme})")
    print(f"  Start : {start_view['name']}  "
          f"x=[{start_view['x_min']:.6f}, {start_view['x_max']:.6f}]  "
          f"y=[{start_view['y_min']:.6f}, {start_view['y_max']:.6f}]")
    print(f"  End   : {end_view['name']}  "
          f"x=[{end_view['x_min']:.6f}, {end_view['x_max']:.6f}]  "
          f"y=[{end_view['y_min']:.6f}, {end_view['y_max']:.6f}]")
    print(f"  Output: {output_path}\n")

    with writer, tqdm(total=n_frames, unit="frame") as bar:
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            view = interpolate_view(start_view, end_view, t)
            rgb = render_frame_rgb(view, kernel, width, height, theme, backend, args.precision)
            writer.append_data(rgb)
            bar.set_postfix(iter=view["max_iter"])
            bar.update()

    duration = n_frames / args.fps
    print(f"\nDone.  {output_path}  ({duration:.1f} s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
