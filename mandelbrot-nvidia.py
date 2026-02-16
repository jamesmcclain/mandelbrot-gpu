import argparse

import cupy as cp
import numpy as np
from PIL import Image


def parse_arguments():
    # yapf: disable
    parser = argparse.ArgumentParser(description='Generate Mandelbrot set image using CUDA')
    parser.add_argument('--width', type=int, default=3840, help='Image width in pixels (default: 3840)')
    parser.add_argument('--height', type=int, default=2160, help='Image height in pixels (default: 2160)')
    parser.add_argument('--max-iter', type=int, default=1024, help='Maximum iterations (default: 1024)')
    parser.add_argument('--output', '-o', type=str, default='mandelbrot.png', help='Output PNG file (default: mandelbrot.png)')
    args = parser.parse_args()
    # yapf: enable
    return args


def main():
    args = parse_arguments()

    WIDTH = args.width
    HEIGHT = args.height
    MAX_ITER = args.max_iter
    OUTPUT_FILE = args.output

    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.0, 1.0

    mod = cp.RawModule(path="mandelbrot.ptx")
    mandelbrot_kernel = mod.get_function("mandelbrot")

    # Allocate output buffer on GPU
    output = cp.zeros((HEIGHT, WIDTH), dtype=cp.uint8)

    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
    blocks = (blocks_x, blocks_y)

    print(
        f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max_iter={MAX_ITER})...")
    print(f"Grid: {blocks}, Block: {threads_per_block}")

    # Launch kernel
    mandelbrot_kernel(
        blocks, threads_per_block,
        (output, WIDTH, HEIGHT, MAX_ITER, cp.float32(X_MIN), cp.float32(X_MAX),
         cp.float32(Y_MIN), cp.float32(Y_MAX)))

    # Copy result back to CPU and save as PNG
    result = cp.asnumpy(output)
    img = Image.fromarray(result, mode='L')
    img.save(OUTPUT_FILE)

    print(f"Saved {OUTPUT_FILE}")
    print(f"Image size: {img.size}")
    print(f"Pixel range: {result.min()} to {result.max()}")


if __name__ == "__main__":
    main()
