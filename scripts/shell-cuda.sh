#!/usr/bin/env bash

docker run -it --rm --gpus all -v $(pwd):/workdir -v $HOME/Desktop:/desktop -w /workdir mandelbrot-cuda bash
