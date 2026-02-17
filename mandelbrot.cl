// Compile with: $HOME/local/llvm-22.1.0-rc3/bin/clang -target nvptx64 -x cl -S mandelbrot.cl -o mandelbrot.ptx

// Helper function to get global work-item ID using NVVM intrinsics
inline size_t get_global_id(uint dim) {
    if (dim == 0) return __nvvm_read_ptx_sreg_tid_x() + __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x();
    if (dim == 1) return __nvvm_read_ptx_sreg_tid_y() + __nvvm_read_ptx_sreg_ctaid_y() * __nvvm_read_ptx_sreg_ntid_y();
    return 0;
}

__kernel void mandelbrot(__global int* output,
                         const int width,
                         const int height,
                         const int max_iter,
                         const float x_min,
                         const float x_max,
                         const float y_min,
                         const float y_max)
{
    int px = (int)get_global_id(0);
    int py = (int)get_global_id(1);

    if (px >= width || py >= height) return;

    // Map pixel to complex plane
    float x0 = x_min + (x_max - x_min) * px / (width - 1.0f);
    float y0 = y_min + (y_max - y_min) * py / (height - 1.0f);

    // Mandelbrot iteration: z = z^2 + c
    float x = 0.0f;
    float y = 0.0f;
    int iteration = 0;

    while (x*x + y*y <= 4.0f && iteration < max_iter) {
        float xtemp = x*x - y*y + x0;
        y = 2.0f*x*y + y0;
        x = xtemp;
        iteration++;
    }

    // Output raw iteration count
    int idx = py * width + px;
    output[idx] = iteration;
}
