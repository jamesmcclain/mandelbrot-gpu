#!/usr/bin/env bash

nvcc -arch=sm_86 -ptx mandelbrot.cu -o mandelbrot.ptx
