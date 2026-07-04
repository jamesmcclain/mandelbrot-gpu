(*$INCLUDE:'mandelbrot.inc'*)
PROGRAM mandelbrot_host(output, outfile, view, prec, theme_sel);

USES MANDELBROT (mandelbrot_f32, mandelbrot_f64);

{ Host-side end-to-end renderer for this repository's Pascal DEVICE UNIT.

  The DEVICE code is unchanged.  This PROGRAM does everything else -- in
  Pascal, with no C shim:
    - allocate the iteration buffer as a heap SUPER ARRAY (long-form NEW)
    - launch the kernel through the CPU-device shim or CUDA shim
    - copy the iteration counts back
    - colorize into a WORD8 pixel buffer (classic/fire/ice/rainbow/emacs/
      grayscale themes, log-normalized escape counts)
    - write the PNG through libpng's simplified write API, called directly
      via the [C] foreign-function interface

  The libpng boundary uses two toolchain guarantees documented in
  pascal-1981's docs/c-abi-foreign-functions.md: the C record-layout
  guarantee (PNGIMAGE below is a field-for-field transcription of libpng
  1.6's png_image, passed by VAR = by pointer), and the heap super-array
  host-buffer pattern (the buffer pointers coerce to the void* parameters).

  The build defaults to DEVICE=cpu, so the whole pipeline runs on a machine
  with no GPU and no NVIDIA toolchain.  See pascal/Makefile. }

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

  { libpng 1.6 simplified-API constants (png.h): PNG_IMAGE_VERSION, and the
    PNG_FORMAT_* composition where GRAY = 0 and RGB = PNG_FORMAT_FLAG_COLOR. }
  png_image_version = 1;
  png_format_gray   = 0;
  png_format_rgb    = 2;

  emacs_stops = 11;   { high index of the emacs gradient stop tables }

TYPE
  ITERBUF = SUPER ARRAY [0..*] OF INTEGER32;
  PITER   = ^ITERBUF;
  PIXBUF  = SUPER ARRAY [0..*] OF WORD8;
  PPIX    = ^PIXBUF;

  { Field-for-field transcription of libpng 1.6's png_image (the simplified
    API control structure).  Passed by VAR (= by pointer) to
    png_image_write_to_file; the record-layout guarantee makes the offsets
    and SIZEOF match the C struct exactly. }
  PNGIMAGE = RECORD
    opaque: ADRMEM;
    version: WORD32;
    width: WORD32;
    height: WORD32;
    format: WORD32;
    flags: WORD32;
    colormap_entries: WORD32;
    warning_or_error: WORD32;
    message: ARRAY[0..63] OF CHAR
  END;

FUNCTION png_image_write_to_file(VAR image: PNGIMAGE;
                                 file_name: ADRMEM;
                                 convert_to_8bit: CINT;
                                 buffer: ADRMEM;
                                 row_stride: CINT;
                                 colormap: ADRMEM): CINT [C]; EXTERN;

VAR
  outfile: LSTRING(96);
  view: INTEGER;
  prec: CHAR;
  theme_sel: INTEGER;

  dev: ADRMEM;
  iters: PITER;
  pixels: PPIX;
  x_min, x_max, y_min, y_max: REAL;
  x_min32, x_max32, y_min32, y_max32: REAL32;
  bytes, npixels, channels: INTEGER32;
  blocks_x, blocks_y: INTEGER32;
  ok: CINT;
  use_f32, grayscale: BOOLEAN;
  theme: INTEGER32;

  { Emacs theme gradient stops (position 0..1 and RGB at each stop). }
  emacs_pos: ARRAY[0..emacs_stops] OF REAL;
  emacs_r, emacs_g, emacs_b: ARRAY[0..emacs_stops] OF INTEGER;

  { Per-pixel colorization results. }
  out_r, out_g, out_b: WORD8;

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

PROCEDURE init_emacs_palette;
  PROCEDURE stop(i: INTEGER; pos: REAL; r, g, b: INTEGER);
  BEGIN
    emacs_pos[i] := pos; emacs_r[i] := r; emacs_g[i] := g; emacs_b[i] := b
  END;
BEGIN
  stop(0,  0.0,    0,   0,   0);
  stop(1,  0.0156, 0,   0,   51);
  stop(2,  0.0312, 0,   0,   102);
  stop(3,  0.0625, 0,   0,   204);
  stop(4,  0.125,  0,   102, 255);
  stop(5,  0.25,   0,   204, 255);
  stop(6,  0.375,  0,   255, 204);
  stop(7,  0.5,    204, 255, 0);
  stop(8,  0.625,  255, 204, 0);
  stop(9,  0.75,   255, 102, 0);
  stop(10, 0.875,  255, 51,  0);
  stop(11, 1.0,    204, 0,   0)
END;

FUNCTION clamp_u8(x: REAL): WORD8;
BEGIN
  IF x < 0.0 THEN
    clamp_u8 := WRD8(0)
  ELSE IF x > 255.0 THEN
    clamp_u8 := WRD8(255)
  ELSE
    clamp_u8 := WRD8(TRUNC(x))
END;

PROCEDURE color_emacs(normalized: REAL);
VAR i: INTEGER; span, t: REAL;
BEGIN
  IF normalized >= 1.0 THEN
  BEGIN
    out_r := WRD8(emacs_r[emacs_stops]);
    out_g := WRD8(emacs_g[emacs_stops]);
    out_b := WRD8(emacs_b[emacs_stops]);
  END
  ELSE
  BEGIN
    out_r := WRD8(0); out_g := WRD8(0); out_b := WRD8(0);
    FOR i := 0 TO emacs_stops - 1 DO
      IF (normalized >= emacs_pos[i]) AND (normalized < emacs_pos[i + 1]) THEN
      BEGIN
        span := emacs_pos[i + 1] - emacs_pos[i];
        IF span > 0.0 THEN
          t := (normalized - emacs_pos[i]) / span
        ELSE
          t := 0.0;
        out_r := clamp_u8(emacs_r[i] + t * (emacs_r[i + 1] - emacs_r[i]));
        out_g := clamp_u8(emacs_g[i] + t * (emacs_g[i + 1] - emacs_g[i]));
        out_b := clamp_u8(emacs_b[i] + t * (emacs_b[i + 1] - emacs_b[i]))
      END
  END
END;

PROCEDURE color_rainbow(normalized: REAL);
VAR hue, f, p, q, t, v, rf, gf, bf: REAL; h_i: INTEGER;
BEGIN
  hue := normalized * 6.0;
  h_i := TRUNC(hue) MOD 6;
  f := hue - TRUNC(hue);
  p := 0.0;
  q := normalized * (1.0 - f);
  t := normalized * f;
  v := normalized;
  CASE h_i OF
    0: BEGIN rf := v; gf := t; bf := p END;
    1: BEGIN rf := q; gf := v; bf := p END;
    2: BEGIN rf := p; gf := v; bf := t END;
    3: BEGIN rf := p; gf := q; bf := v END;
    4: BEGIN rf := t; gf := p; bf := v END;
    OTHERWISE BEGIN rf := v; gf := p; bf := q END
  END;
  out_r := clamp_u8(rf * 255.0);
  out_g := clamp_u8(gf * 255.0);
  out_b := clamp_u8(bf * 255.0)
END;

PROCEDURE color_pixel(level: INTEGER);
VAR normalized: REAL;
BEGIN
  normalized := level / 255.0;
  CASE theme OF
    theme_fire:
      BEGIN
        out_r := clamp_u8(normalized * 512.0);
        out_g := clamp_u8((normalized - 0.5) * 512.0);
        out_b := clamp_u8((normalized - 0.75) * 1024.0)
      END;
    theme_ice:
      BEGIN
        out_b := clamp_u8(normalized * 512.0);
        out_g := clamp_u8((normalized - 0.5) * 512.0);
        out_r := clamp_u8((normalized - 0.75) * 1024.0)
      END;
    theme_rainbow:
      color_rainbow(normalized);
    theme_emacs:
      color_emacs(normalized);
    OTHERWISE
      BEGIN
        out_r := clamp_u8(normalized * 400.0);
        out_g := clamp_u8((normalized - 0.6) * 640.0);
        out_b := clamp_u8(normalized * 600.0)
      END
  END
END;

PROCEDURE colorize;
VAR
  i, it, level: INTEGER32;
  itr, log_iter, log_max: REAL;
BEGIN
  log_max := LN(max_iter + 1.0);
  FOR i := 0 TO npixels - 1 DO
  BEGIN
    it := iters^[i];
    level := 0;
    IF it < max_iter THEN
    BEGIN
      itr := it;
      log_iter := LN(itr + 1.0);
      level := TRUNC(255.0 * log_iter / log_max)
    END;
    IF grayscale THEN
      pixels^[i] := WRD8(level)
    ELSE
    BEGIN
      color_pixel(WRD8(level));   { WRD8 narrows the 0..255 INTEGER32 }
      pixels^[3 * i] := out_r;
      pixels^[3 * i + 1] := out_g;
      pixels^[3 * i + 2] := out_b
    END
  END
END;

PROCEDURE write_png;
VAR
  image: PNGIMAGE;
  fname: ARRAY[0..96] OF CHAR;
  i, len: INTEGER;
BEGIN
  { NUL-terminate the LSTRING filename for the C API. }
  len := ORD(outfile.LEN);
  FOR i := 1 TO len DO
    fname[i - 1] := outfile[i];
  fname[len] := CHR(0);

  { Zero the control record (png_image requires it), then fill it in. }
  FILLC(ADR image, SIZEOF(image), CHR(0));
  image.version := png_image_version;
  image.width := width;
  image.height := height;
  IF grayscale THEN
    image.format := png_format_gray
  ELSE
    image.format := png_format_rgb;

  ok := png_image_write_to_file(image, ADR fname, 0, pixels, 0, NIL)
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
  grayscale := theme = theme_grayscale;
  IF theme = theme_emacs THEN
    init_emacs_palette;

  apply_view(view);
  npixels := width * height;
  bytes := npixels * SIZEOF(INTEGER32);
  IF grayscale THEN
    channels := 1
  ELSE
    channels := 3;
  blocks_x := (width + block_x - 1) DIV block_x;
  blocks_y := (height + block_y - 1) DIV block_y;

  dev := DEVALLOC(bytes);
  IF dev = NIL THEN
    ABORT('device allocation failed', WRD(1), WRD(1));

  NEW(iters, npixels - 1);
  NEW(pixels, npixels * channels - 1);

  IF use_f32 THEN
  BEGIN
    x_min32 := x_min; x_max32 := x_max; y_min32 := y_min; y_max32 := y_max;
    LAUNCH(mandelbrot_f32, blocks_x, blocks_y, 1, block_x, block_y, 1,
           dev, width, height, max_iter, x_min32, x_max32, y_min32, y_max32)
  END
  ELSE
    LAUNCH(mandelbrot_f64, blocks_x, blocks_y, 1, block_x, block_y, 1,
           dev, width, height, max_iter, x_min, x_max, y_min, y_max);

  DEVCOPYFROM(iters, dev, bytes);
  DEVFREE(dev);

  colorize;
  write_png;

  DISPOSE(pixels);
  DISPOSE(iters);

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
