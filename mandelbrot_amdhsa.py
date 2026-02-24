import numpy as np
import pyopencl as cl


def load_kernel(compiled_file="mandelbrot.hsaco"):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    with open(compiled_file, "rb") as f:
        binary = f.read()

    device = ctx.devices[0]
    program = cl.Program(ctx, [device], [binary])
    program.build()
    kernel = program.mandelbrot

    return (ctx, queue, kernel)


def run_kernel(handle, WIDTH, HEIGHT, max_iter, x_min, x_max, y_min, y_max):
    ctx, queue, kernel = handle

    output_host = np.zeros((HEIGHT, WIDTH), dtype=np.int32)

    output_buf = cl.Buffer(ctx,
                           cl.mem_flags.WRITE_ONLY,
                           size=output_host.nbytes)

    local_size = (16, 16)
    global_size = (
        ((WIDTH + local_size[0] - 1) // local_size[0]) * local_size[0],
        ((HEIGHT + local_size[1] - 1) // local_size[1]) * local_size[1],
    )

    kernel(queue, global_size, local_size, output_buf, np.int32(WIDTH),
           np.int32(HEIGHT), np.int32(max_iter), np.float32(x_min),
           np.float32(x_max), np.float32(y_min), np.float32(y_max))

    cl.enqueue_copy(queue, output_host, output_buf)
    queue.finish()

    return output_host
