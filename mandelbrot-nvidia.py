#!/usr/bin/env python3

import argparse

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image


def parse_arguments():
    # yapf: disable
    parser = argparse.ArgumentParser(description='Generate Mandelbrot set image using CUDA')
    parser.add_argument('--width', type=int, default=3840, help='Image width in pixels (default: 3840)')
    parser.add_argument('--height', type=int, default=2160, help='Image height in pixels (default: 2160)')
    parser.add_argument('--max-iter', type=int, default=1024, help='Maximum iterations (default: 1024)')
    parser.add_argument('--output', '-o', type=str, default='mandelbrot.png', help='Output PNG file (default: mandelbrot.png)')
    parser.add_argument('--theme', type=str, default='classic', choices=['grayscale', 'emacs', 'fire', 'ice', 'rainbow', 'classic'], help='Color theme (default: classic)')
    args = parser.parse_args()
    # yapf: enable
    return args


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

    # Create RGB image
    height, width = data.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Normalize to 0-1 range
    normalized = data.astype(np.float32) / 255.0

    if theme == 'emacs':
        # Emacs color scheme: Black → Blue → Cyan → Green-Yellow → Yellow → Orange → Red
        # Based on the mandelbrot-text.el color scheme

        # Define color stops (normalized positions and RGB values)
        # Thresholds from emacs: 0, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256
        # As fractions: 0, 0.0156, 0.0312, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
        stops = [
            (0.0, (0, 0, 0)),  # Black (in set)
            (0.0156, (0, 0, 51)),  # Very dark blue
            (0.0312, (0, 0, 102)),  # Dark blue
            (0.0625, (0, 0, 204)),  # Blue
            (0.125, (0, 102, 255)),  # Light blue
            (0.25, (0, 204, 255)),  # Cyan
            (0.375, (0, 255, 204)),  # Cyan-green
            (0.5, (204, 255, 0)),  # Yellow-green
            (0.625, (255, 204, 0)),  # Yellow
            (0.75, (255, 102, 0)),  # Orange
            (0.875, (255, 51, 0)),  # Red-orange
            (1.0, (204, 0, 0))  # Red
        ]

        # Interpolate colors
        for i in range(len(stops) - 1):
            pos1, color1 = stops[i]
            pos2, color2 = stops[i + 1]

            # Find pixels in this range
            mask = (normalized >= pos1) & (normalized < pos2)

            # Linear interpolation
            if np.any(mask):
                t = (normalized[mask] - pos1) / (pos2 - pos1)
                rgb[mask, 0] = (color1[0] + t *
                                (color2[0] - color1[0])).astype(np.uint8)
                rgb[mask, 1] = (color1[1] + t *
                                (color2[1] - color1[1])).astype(np.uint8)
                rgb[mask, 2] = (color1[2] + t *
                                (color2[2] - color1[2])).astype(np.uint8)

        # Handle the maximum value (exactly 1.0)
        mask = normalized >= 1.0
        if np.any(mask):
            rgb[mask] = stops[-1][1]

    elif theme == 'fire':
        # Black -> Red -> Yellow -> White
        rgb[:, :, 0] = np.minimum(255,
                                  normalized * 512).astype(np.uint8)  # Red
        rgb[:, :,
            1] = np.maximum(0,
                            (normalized - 0.5) * 512).astype(np.uint8)  # Green
        rgb[:, :, 2] = np.maximum(0, (normalized - 0.75) * 1024).astype(
            np.uint8)  # Blue

    elif theme == 'ice':
        # Black -> Blue -> Cyan -> White
        rgb[:, :, 2] = np.minimum(255,
                                  normalized * 512).astype(np.uint8)  # Blue
        rgb[:, :,
            1] = np.maximum(0,
                            (normalized - 0.5) * 512).astype(np.uint8)  # Green
        rgb[:, :,
            0] = np.maximum(0,
                            (normalized - 0.75) * 1024).astype(np.uint8)  # Red

    elif theme == 'rainbow':
        # Full spectrum cycling
        # Use HSV-like mapping: vary hue based on value
        hue = normalized * 6.0  # 0-6 range for color cycling

        # Convert HSV to RGB (simplified, S=1, V=normalized)
        h_i = hue.astype(np.int32) % 6
        f = hue - h_i

        p = np.zeros_like(normalized)
        q = normalized * (1 - f)
        t = normalized * f
        v = normalized

        # Assign RGB based on hue sector
        rgb[:, :, 0] = np.where(
            h_i == 0, v * 255,
            np.where(
                h_i == 1, q * 255,
                np.where(
                    h_i == 2, p * 255,
                    np.where(h_i == 3, p * 255,
                             np.where(h_i == 4, t * 255,
                                      v * 255))))).astype(np.uint8)

        rgb[:, :, 1] = np.where(
            h_i == 0, t * 255,
            np.where(
                h_i == 1, v * 255,
                np.where(
                    h_i == 2, v * 255,
                    np.where(h_i == 3, q * 255,
                             np.where(h_i == 4, p * 255,
                                      p * 255))))).astype(np.uint8)

        rgb[:, :, 2] = np.where(
            h_i == 0, p * 255,
            np.where(
                h_i == 1, p * 255,
                np.where(
                    h_i == 2, t * 255,
                    np.where(h_i == 3, v * 255,
                             np.where(h_i == 4, v * 255,
                                      q * 255))))).astype(np.uint8)

    elif theme == 'classic':
        # Classic blue-purple Mandelbrot coloring
        # Black -> Dark Blue -> Purple -> Pink -> White
        rgb[:, :,
            0] = np.minimum(255,
                            normalized * 400).astype(np.uint8)  # Red (slower)
        rgb[:, :, 1] = np.maximum(0, (normalized - 0.6) * 640).astype(
            np.uint8)  # Green (latest)
        rgb[:, :,
            2] = np.minimum(255,
                            normalized * 600).astype(np.uint8)  # Blue (faster)

    return rgb


def main():
    args = parse_arguments()

    WIDTH = args.width
    HEIGHT = args.height
    MAX_ITER = args.max_iter
    OUTPUT_FILE = args.output
    THEME = args.theme

    X_MIN, X_MAX = np.float32(-2.5), np.float32(1.0)
    Y_MIN, Y_MAX = np.float32(-1.0), np.float32(1.0)

    mod = cuda.module_from_file("mandelbrot.ptx")
    mandelbrot_kernel = mod.get_function("mandelbrot")

    # Allocate output buffer on GPU (int32 for raw iteration counts)
    output_host = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    output_gpu = cuda.mem_alloc(output_host.nbytes)

    # Configure grid and block dimensions
    threads_per_block = (16, 16, 1)
    blocks_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
    grid = (blocks_x, blocks_y, 1)

    # Launch kernel
    mandelbrot_kernel(output_gpu,
                      np.int32(WIDTH),
                      np.int32(HEIGHT),
                      np.int32(MAX_ITER),
                      X_MIN,
                      X_MAX,
                      Y_MIN,
                      Y_MAX,
                      block=threads_per_block,
                      grid=grid)

    # Copy result back to CPU
    cuda.memcpy_dtoh(output_host, output_gpu)
    result = output_host

    # Normalize iteration counts to 0-255 range for color mapping
    # Use a fixed scale for consistent colors regardless of max_iter
    # Points in the set (iteration == max_iter) map to 0 (black)
    # Use logarithmic scaling for better visual distribution
    normalized = np.zeros_like(result, dtype=np.uint8)

    # Points in the set (didn't escape)
    in_set = result >= MAX_ITER
    normalized[in_set] = 0

    # Points that escaped - use log scaling for better color distribution
    escaped = ~in_set
    if np.any(escaped):
        # Log scale: log(iteration + 1) / log(max_iter + 1)
        log_iter = np.log(result[escaped] + 1)
        log_max = np.log(MAX_ITER + 1)
        normalized[escaped] = (255 * log_iter / log_max).astype(np.uint8)

    # Apply color theme
    colored_result = apply_color_theme(normalized, THEME)

    # Save as PNG
    if THEME == 'grayscale':
        img = Image.fromarray(colored_result, mode='L')
    else:
        img = Image.fromarray(colored_result, mode='RGB')

    img.save(OUTPUT_FILE)

    print(f"Saved {OUTPUT_FILE}")
    print(f"Image size: {img.size}")
    print(f"Iteration range: {result.min()} to {result.max()}")
    print(f"Normalized range: {normalized.min()} to {normalized.max()}")


if __name__ == "__main__":
    main()
