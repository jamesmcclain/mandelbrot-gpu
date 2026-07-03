(*$INCLUDE:'mandelbrot.inc'*)
PROGRAM mandelbrot_host(output, outfile, view, prec, theme_sel);

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
FUNCTION write_mandelbrot_png_lstring(file_name: ADRMEM;
                                      width: CINT;
                                      height: CINT;
                                      max_iter: CINT;
                                      theme: CINT;
                                      iters: ADRMEM): CINT [C]; EXTERN;

CONST
  width    = 3840;
  height   = 2160;
  max_iter = 512;

  block_x = 16;
  block_y = 16;

  theme_classic   = 0;
  theme_fire      = 1;
  theme_ice       = 2;
  theme_rainbow   = 3;
  theme_emacs     = 4;
  theme_grayscale = 5;

VAR
  outfile: LSTRING(96);
  view: INTEGER;
  prec: CHAR;
  theme_sel: INTEGER;

  dev, host_buf: ADRMEM;
  x_min, x_max, y_min, y_max: REAL;
  x_min32, x_max32, y_min32, y_max32: REAL32;
  bytes: INTEGER32;
  malloc_bytes: CSIZE_T;
  blocks_x, blocks_y: INTEGER32;
  ok: CINT;
  use_f32: BOOLEAN;
  theme: INTEGER32;

PROCEDURE apply_view(which: INTEGER);
BEGIN
  CASE which OF
    1:
      BEGIN
        x_min := -2.9722; x_max := 1.4722; y_min := -1.25; y_max := 1.25
      END;
    2:
      BEGIN
        x_min := -0.7828; x_max := -0.6832; y_min := 0.092; y_max := 0.148
      END;
    3:
      BEGIN
        x_min := 0.1994; x_max := 0.4306; y_min := -0.065; y_max := 0.065
      END;
    4:
      BEGIN
        x_min := -0.7801; x_max := -0.7249; y_min := 0.1000; y_max := 0.1310
      END;
    OTHERWISE
      BEGIN
        WRITELN('unknown view ', which, '; using overview');
        x_min := -2.9722; x_max := 1.4722; y_min := -1.25; y_max := 1.25
      END
  END
END;

BEGIN
  use_f32 := FALSE;
  IF (prec = 's') OR (prec = 'S') THEN
    use_f32 := TRUE;

  theme := theme_sel;
  IF (theme < theme_classic) OR (theme > theme_grayscale) THEN
  BEGIN
    WRITELN('unknown theme ', theme, '; using classic');
    theme := theme_classic
  END;

  apply_view(view);
  bytes := width * height * SIZEOF(INTEGER32);
  malloc_bytes := bytes;
  blocks_x := (width + block_x - 1) DIV block_x;
  blocks_y := (height + block_y - 1) DIV block_y;

  dev := DEVALLOC(bytes);
  IF dev = NIL THEN
    ABORT('device allocation failed', WRD(1), WRD(1));

  host_buf := malloc(malloc_bytes);
  IF host_buf = NIL THEN
  BEGIN
    DEVFREE(dev);
    ABORT('host allocation failed', WRD(1), WRD(2))
  END;

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

  ok := write_mandelbrot_png_lstring(ADR outfile, width, height, max_iter, theme, host_buf);
  free(host_buf);

  IF ok = 1 THEN
  BEGIN
    WRITELN('wrote ', outfile);
    IF use_f32 THEN
      WRITELN('precision: f32')
    ELSE
      WRITELN('precision: f64')
  END
  ELSE
    WRITELN('png write failed')
END.
