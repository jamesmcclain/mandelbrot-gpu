(*$INCLUDE:'mandelbrot.inc'*)
PROGRAM mandelbrot_host(output);

USES MANDELBROT (mandelbrot_f32, mandelbrot_f64);

{ Host-side end-to-end renderer for this repository's Pascal DEVICE UNIT.

  The DEVICE code is unchanged. This PROGRAM does the rest:
    - allocate a device buffer
    - launch the kernel through the CPU-device shim or CUDA shim
    - copy the iteration counts back
    - hand the buffer to a tiny C/libpng helper that writes the PNG

  The build defaults to DEVICE=cpu, so the whole pipeline runs on a machine with
  no GPU and no NVIDIA toolchain. See pascal/Makefile. }

FUNCTION malloc(nbytes: CSIZE_T): ADRMEM [C]; EXTERN;
PROCEDURE free(p: ADRMEM) [C]; EXTERN;
FUNCTION write_mandelbrot_png(file_name: ADRMEM;
                              width: CINT;
                              height: CINT;
                              max_iter: CINT;
                              theme: CINT;
                              iters: ADRMEM): CINT [C]; EXTERN;

CONST
  width    = 640;
  height   = 360;
  max_iter = 512;

  block_x = 16;
  block_y = 16;

  theme_classic   = 0;
  theme_fire      = 1;
  theme_ice       = 2;
  theme_rainbow   = 3;
  theme_emacs     = 4;
  theme_grayscale = 5;

  use_f32 = FALSE;
  theme   = theme_classic;

VAR
  dev, host_buf: ADRMEM;
  x_min, x_max, y_min, y_max: REAL;
  x_min32, x_max32, y_min32, y_max32: REAL32;
  bytes: INTEGER32;
  blocks_x, blocks_y: INTEGER32;
  ok: CINT;
  filename: ARRAY [0..31] OF CHAR;

PROCEDURE build_filename;
BEGIN
  filename[0] := 'm'; filename[1] := 'a'; filename[2] := 'n'; filename[3] := 'd';
  filename[4] := 'e'; filename[5] := 'l'; filename[6] := 'b'; filename[7] := 'r';
  filename[8] := 'o'; filename[9] := 't'; filename[10] := '_'; filename[11] := 'p';
  filename[12] := 'a'; filename[13] := 's'; filename[14] := 'c'; filename[15] := 'a';
  filename[16] := 'l';
  IF use_f32 THEN
  BEGIN
    filename[17] := '_'; filename[18] := 'f'; filename[19] := '3'; filename[20] := '2';
    filename[21] := '.'; filename[22] := 'p'; filename[23] := 'n'; filename[24] := 'g';
    filename[25] := CHR(0)
  END
  ELSE
  BEGIN
    filename[17] := '_'; filename[18] := 'f'; filename[19] := '6'; filename[20] := '4';
    filename[21] := '.'; filename[22] := 'p'; filename[23] := 'n'; filename[24] := 'g';
    filename[25] := CHR(0)
  END
END;

BEGIN
  { Full-set overview, matching the Python project's built-in overview view. }
  x_min := -2.9722;
  x_max :=  1.4722;
  y_min := -1.25;
  y_max :=  1.25;

  build_filename;
  bytes := width * height * 4;
  blocks_x := (width + block_x - 1) DIV block_x;
  blocks_y := (height + block_y - 1) DIV block_y;

  dev := DEVALLOC(bytes);
  host_buf := malloc(bytes);

  IF use_f32 THEN
  BEGIN
    x_min32 := x_min; x_max32 := x_max; y_min32 := y_min; y_max32 := y_max;
    LAUNCH(mandelbrot_f32, blocks_x, blocks_y, 1, block_x, block_y, 1,
           dev, width, height, max_iter, x_min32, x_max32, y_min32, y_max32)
  END
  ELSE
    LAUNCH(mandelbrot_f64, blocks_x, blocks_y, 1, block_x, block_y, 1,
           dev, width, height, max_iter, x_min, x_max, y_min, y_max);

  DEVCOPYFROM(host_buf, dev, bytes);
  DEVFREE(dev);

  ok := write_mandelbrot_png(ADR filename, width, height, max_iter, theme, host_buf);
  free(host_buf);

  IF ok = 1 THEN
  BEGIN
    WRITELN('wrote ', filename);
    IF use_f32 THEN
      WRITELN('precision: f32')
    ELSE
      WRITELN('precision: f64')
  END
  ELSE
    WRITELN('png write failed')
END.
