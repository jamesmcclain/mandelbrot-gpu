import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda


def load_kernel(compiled_file="mandelbrot.ptx"):
    mod = cuda.module_from_file(compiled_file)
    return mod.get_function("mandelbrot")


def run_kernel(kernel, WIDTH, HEIGHT, max_iter, x_min, x_max, y_min, y_max):
    output_host = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    output_gpu = cuda.mem_alloc(output_host.nbytes)

    threads_per_block = (16, 16, 1)
    blocks_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]

    kernel(output_gpu,
           np.int32(WIDTH),
           np.int32(HEIGHT),
           np.int32(max_iter),
           np.float32(x_min),
           np.float32(x_max),
           np.float32(y_min),
           np.float32(y_max),
           block=threads_per_block,
           grid=(blocks_x, blocks_y, 1))

    cuda.memcpy_dtoh(output_host, output_gpu)
    output_gpu.free()

    return output_host
