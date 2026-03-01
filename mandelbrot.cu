// nvcc -arch=sm_86 -ptx mandelbrot.cu -o mandelbrot.ptx

extern "C" __global__
void mandelbrot_f32(int* output, int width, int height, int max_iter,
                    float x_min, float x_max, float y_min, float y_max) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

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

extern "C" __global__
void mandelbrot_f64(int* output, int width, int height, int max_iter,
                    double x_min, double x_max, double y_min, double y_max) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

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
