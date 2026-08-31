#!/usr/bin/env python3
"""End-to-end CPU regression test for the Pascal Mandelbrot renderer."""

import argparse
import hashlib
import math
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

WIDTH = 320
HEIGHT = 180
MAX_ITER = 512
ARGS = ("1", "d", "5", str(WIDTH), str(HEIGHT))
EXPECTED_SHA256 = Path(__file__).with_name("cpu_grayscale_320x180.sha256")


def read_png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    position = 8
    chunks = []
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        chunks.append((kind, payload))
        position += length + 12
    ihdr = next(payload for kind, payload in chunks if kind == b"IHDR")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr)
    assert (width, height, depth, color, compression, filtering, interlace) == (
        WIDTH, HEIGHT, 8, 0, 0, 0, 0), "unexpected PNG IHDR"
    raw = zlib.decompress(b"".join(payload for kind, payload in chunks if kind == b"IDAT"))
    return data, unfilter(raw, width, height)


def unfilter(raw, width, height):
    """Decode the PNG filters used by libpng for an 8-bit grayscale image."""
    stride = width
    rows = []
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        encoded = raw[offset:offset + stride]
        offset += stride
        previous = rows[-1] if rows else bytes(stride)
        row = bytearray(stride)
        for i, value in enumerate(encoded):
            left = row[i - 1] if i else 0
            above = previous[i]
            upper_left = previous[i - 1] if i else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                p = left + above - upper_left
                pa, pb, pc = abs(p - left), abs(p - above), abs(p - upper_left)
                predictor = left if pa <= pb and pa <= pc else above if pb <= pc else upper_left
            else:
                raise AssertionError(f"unsupported PNG filter {filter_type}")
            row[i] = (value + predictor) & 0xff
        rows.append(bytes(row))
    assert offset == len(raw), "unexpected trailing PNG image data"
    return b"".join(rows)


def reference_pixels():
    """Scalar CPU oracle for the committed CUDA/Pascal f64 kernel, view 1."""
    output = bytearray()
    x_min, x_max, y_min, y_max = -2.9722, 1.4722, -1.25, 1.25
    log_max = math.log(MAX_ITER + 1.0)
    for py in range(HEIGHT):
        y0 = y_min + (y_max - y_min) * py / (HEIGHT - 1)
        for px in range(WIDTH):
            x0 = x_min + (x_max - x_min) * px / (WIDTH - 1)
            x = y = 0.0
            iteration = 0
            while x * x + y * y <= 4.0 and iteration < MAX_ITER:
                x, y = x * x - y * y + x0, 2.0 * x * y + y0
                iteration += 1
            output.append(0 if iteration == MAX_ITER else int(255.0 * math.log(iteration + 1.0) / log_max))
    return bytes(output)


def render(binary, output):
    completed = subprocess.run([str(binary), str(output), *ARGS], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file() and output.stat().st_size > 0, "renderer did not create PNG"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    assert binary.is_file(), f"missing renderer: {binary}"

    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.png"
        second = Path(directory) / "second.png"
        render(binary, first)
        render(binary, second)
        first_data, first_pixels = read_png(first)
        second_data, second_pixels = read_png(second)
        first_hash = hashlib.sha256(first_data).hexdigest()
        second_hash = hashlib.sha256(second_data).hexdigest()
        assert first_hash == second_hash, "repeated renders are not byte-identical"
        expected_hash = EXPECTED_SHA256.read_text().split()[0]
        assert first_hash == expected_hash, f"checksum drift: {first_hash} != {expected_hash}"
        assert first_pixels == second_pixels, "repeated renders have different pixels"
        assert first_pixels == reference_pixels(), "Pascal pixels differ from scalar CPU oracle"
        print(f"CPU render verified: {WIDTH}x{HEIGHT}, sha256 {first_hash}")


if __name__ == "__main__":
    main()
