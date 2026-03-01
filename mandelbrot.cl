// $HOME/local/llvm-22.1.0-rc3/bin/clang \
//   -target nvptx64-nvidia-cuda -march=sm_86 \
//   -x cl -O3 -cl-std=CL2.0 \
//   -Xclang -finclude-default-header \
//   -Xclang -target-feature -Xclang +ptx71 \
//   -Xclang -mlink-builtin-bitcode \
//   -Xclang $HOME/local/llvm-22.1.0-rc3/share/clc/nvptx64-nvidia-cuda.bc \
//   -S mandelbrot.cl -o mandelbrot.ptx

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void mandelbrot_f32(__global int* output,
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

    float width_denom = (width > 1) ? (float)(width - 1) : 1.0f;
    float height_denom = (height > 1) ? (float)(height - 1) : 1.0f;

    // Map pixel to complex plane
    float x0 = x_min + (x_max - x_min) * (float)px / width_denom;
    float y0 = y_min + (y_max - y_min) * (float)py / height_denom;

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

    int idx = py * width + px;
    output[idx] = iteration;
}

__kernel void mandelbrot_f64(__global int* output,
                             const int width,
                             const int height,
                             const int max_iter,
                             const double x_min,
                             const double x_max,
                             const double y_min,
                             const double y_max)
{
    int px = (int)get_global_id(0);
    int py = (int)get_global_id(1);

    if (px >= width || py >= height) return;

    double width_denom = (width > 1) ? (double)(width - 1) : 1.0;
    double height_denom = (height > 1) ? (double)(height - 1) : 1.0;

    // Map pixel to complex plane
    double x0 = x_min + (x_max - x_min) * (double)px / width_denom;
    double y0 = y_min + (y_max - y_min) * (double)py / height_denom;

    // Mandelbrot iteration: z = z^2 + c
    double x = 0.0;
    double y = 0.0;
    int iteration = 0;

    while (x*x + y*y <= 4.0 && iteration < max_iter) {
        double xtemp = x*x - y*y + x0;
        y = 2.0*x*y + y0;
        x = xtemp;
        iteration++;
    }

    int idx = py * width + px;
    output[idx] = iteration;
}
