import cupy as cp
import numpy as np
from PIL import Image

# Configuration
WIDTH = 3840
HEIGHT = 2160
MAX_ITER = 2**10

# Mandelbrot set bounds (classic view centered on origin)
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

# Load the kernel
# To use OpenCL version later, just change the path to "mandelbrot_cl.ptx"
mod = cp.RawModule(path="mandelbrot.ptx")
mandelbrot_kernel = mod.get_function("mandelbrot")

# Allocate output buffer on GPU
output = cp.zeros((HEIGHT, WIDTH), dtype=cp.uint8)

# Configure grid and block dimensions
threads_per_block = (16, 16)
blocks_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
blocks = (blocks_x, blocks_y)

print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max_iter={MAX_ITER})...")
print(f"Grid: {blocks}, Block: {threads_per_block}")

# Launch kernel
mandelbrot_kernel(
    blocks,
    threads_per_block,
    (output, WIDTH, HEIGHT, MAX_ITER,
     cp.float32(X_MIN), cp.float32(X_MAX),
     cp.float32(Y_MIN), cp.float32(Y_MAX))
)

# Copy result back to CPU and save as PNG
result = cp.asnumpy(output)
img = Image.fromarray(result, mode='L')
img.save('mandelbrot.png')

print("Saved mandelbrot.png")
print(f"Image size: {img.size}")
print(f"Pixel range: {result.min()} to {result.max()}")
