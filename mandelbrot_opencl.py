import numpy as np
import pyopencl as cl

from mandelbrot_precision import choose_precision


def load_kernel(source_file="mandelbrot.cl"):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    with open(source_file, "r") as f:
        source = f.read()

    program = cl.Program(ctx, source).build(options="-cl-std=CL2.0")
    kernels = {
        "float32": program.mandelbrot_f32,
        "float64": program.mandelbrot_f64,
    }

    return (ctx, queue, kernels)


def run_kernel(handle, WIDTH, HEIGHT, max_iter, x_min, x_max, y_min, y_max, precision="auto"):
    ctx, queue, kernels = handle

    if precision == "auto":
        precision = choose_precision(WIDTH, HEIGHT, max_iter, x_min, x_max, y_min, y_max)

    output_host = np.zeros((HEIGHT, WIDTH), dtype=np.int32)

    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=output_host.nbytes)

    local_size = (16, 16)
    global_size = (
        ((WIDTH + local_size[0] - 1) // local_size[0]) * local_size[0],
        ((HEIGHT + local_size[1] - 1) // local_size[1]) * local_size[1],
    )

    kernel = kernels[precision]
    real_dtype = np.float32 if precision == "float32" else np.float64

    kernel(queue, global_size, local_size, output_buf, np.int32(WIDTH), np.int32(HEIGHT), np.int32(max_iter), real_dtype(x_min), real_dtype(x_max), real_dtype(y_min), real_dtype(y_max))

    cl.enqueue_copy(queue, output_host, output_buf)
    queue.finish()

    return output_host
