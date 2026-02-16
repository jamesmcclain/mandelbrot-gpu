// Compile with: nvcc -ptx mandelbrot.cu -o mandelbrot.ptx -arch=sm_86

extern "C" __global__
void mandelbrot(unsigned char* output, int width, int height, int max_iter,
                float x_min, float x_max, float y_min, float y_max) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
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
    
    // Map iteration count to grayscale value
    int idx = py * width + px;
    if (iteration == max_iter) {
        output[idx] = 0;  // Inside set = black
    } else {
        // Smooth coloring: normalize to 0-255
        output[idx] = (unsigned char)(255.0f * iteration / max_iter);
    }
}
