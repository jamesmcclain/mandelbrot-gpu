#include <math.h>
#include <png.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  THEME_CLASSIC = 0,
  THEME_FIRE = 1,
  THEME_ICE = 2,
  THEME_RAINBOW = 3,
  THEME_EMACS = 4,
  THEME_GRAYSCALE = 5,
};

static uint8_t clamp_u8(float x) {
  if (x < 0.0f) return 0;
  if (x > 255.0f) return 255;
  return (uint8_t)x;
}

static void color_emacs(float normalized, uint8_t *r, uint8_t *g, uint8_t *b) {
  static const struct {
    float pos;
    uint8_t rgb[3];
  } stops[] = {
      {0.0f, {0, 0, 0}},       {0.0156f, {0, 0, 51}},   {0.0312f, {0, 0, 102}},
      {0.0625f, {0, 0, 204}},  {0.125f, {0, 102, 255}}, {0.25f, {0, 204, 255}},
      {0.375f, {0, 255, 204}}, {0.5f, {204, 255, 0}},   {0.625f, {255, 204, 0}},
      {0.75f, {255, 102, 0}},  {0.875f, {255, 51, 0}},  {1.0f, {204, 0, 0}},
  };

  if (normalized >= 1.0f) {
    *r = stops[11].rgb[0];
    *g = stops[11].rgb[1];
    *b = stops[11].rgb[2];
    return;
  }

  for (size_t i = 0; i + 1 < sizeof(stops) / sizeof(stops[0]); ++i) {
    if (normalized >= stops[i].pos && normalized < stops[i + 1].pos) {
      float span = stops[i + 1].pos - stops[i].pos;
      float t = (span > 0.0f) ? (normalized - stops[i].pos) / span : 0.0f;
      *r = clamp_u8(stops[i].rgb[0] + t * (stops[i + 1].rgb[0] - stops[i].rgb[0]));
      *g = clamp_u8(stops[i].rgb[1] + t * (stops[i + 1].rgb[1] - stops[i].rgb[1]));
      *b = clamp_u8(stops[i].rgb[2] + t * (stops[i + 1].rgb[2] - stops[i].rgb[2]));
      return;
    }
  }

  *r = *g = *b = 0;
}

static void color_rainbow(float normalized, uint8_t *r, uint8_t *g, uint8_t *b) {
  float hue = normalized * 6.0f;
  int h_i = ((int)hue) % 6;
  float f = hue - floorf(hue);
  float p = 0.0f;
  float q = normalized * (1.0f - f);
  float t = normalized * f;
  float v = normalized;
  float rf, gf, bf;

  switch (h_i) {
    case 0: rf = v; gf = t; bf = p; break;
    case 1: rf = q; gf = v; bf = p; break;
    case 2: rf = p; gf = v; bf = t; break;
    case 3: rf = p; gf = q; bf = v; break;
    case 4: rf = t; gf = p; bf = v; break;
    default: rf = v; gf = p; bf = q; break;
  }

  *r = clamp_u8(rf * 255.0f);
  *g = clamp_u8(gf * 255.0f);
  *b = clamp_u8(bf * 255.0f);
}

static void color_pixel(uint8_t value, int32_t theme, uint8_t *r, uint8_t *g, uint8_t *b) {
  float normalized = (float)value / 255.0f;

  switch (theme) {
    case THEME_FIRE:
      *r = clamp_u8(normalized * 512.0f);
      *g = clamp_u8((normalized - 0.5f) * 512.0f);
      *b = clamp_u8((normalized - 0.75f) * 1024.0f);
      break;
    case THEME_ICE:
      *b = clamp_u8(normalized * 512.0f);
      *g = clamp_u8((normalized - 0.5f) * 512.0f);
      *r = clamp_u8((normalized - 0.75f) * 1024.0f);
      break;
    case THEME_RAINBOW:
      color_rainbow(normalized, r, g, b);
      break;
    case THEME_EMACS:
      color_emacs(normalized, r, g, b);
      break;
    case THEME_CLASSIC:
    default:
      *r = clamp_u8(normalized * 400.0f);
      *g = clamp_u8((normalized - 0.6f) * 640.0f);
      *b = clamp_u8(normalized * 600.0f);
      break;
  }
}

int32_t write_mandelbrot_png(const char *file_name,
                             int32_t width,
                             int32_t height,
                             int32_t max_iter,
                             int32_t theme,
                             const int32_t *iters) {
  size_t pixels = (size_t)width * (size_t)height;
  int grayscale = (theme == THEME_GRAYSCALE);
  size_t channels = grayscale ? 1u : 3u;
  size_t image_bytes = pixels * channels;
  uint8_t *buffer = (uint8_t *)malloc(image_bytes);
  png_image image;
  int ok;

  if (buffer == NULL) return 0;

  for (size_t i = 0; i < pixels; ++i) {
    int32_t it = iters[i];
    uint8_t value = 0;

    if (it < max_iter) {
      double log_iter = log((double)it + 1.0);
      double log_max = log((double)max_iter + 1.0);
      value = (uint8_t)(255.0 * log_iter / log_max);
    }

    if (grayscale) {
      buffer[i] = value;
    } else {
      uint8_t r = 0, g = 0, b = 0;
      color_pixel(value, theme, &r, &g, &b);
      buffer[3 * i] = r;
      buffer[3 * i + 1] = g;
      buffer[3 * i + 2] = b;
    }
  }

  memset(&image, 0, sizeof(image));
  image.version = PNG_IMAGE_VERSION;
  image.width = (png_uint_32)width;
  image.height = (png_uint_32)height;
  image.format = grayscale ? PNG_FORMAT_GRAY : PNG_FORMAT_RGB;

  ok = png_image_write_to_file(&image, file_name, 0, buffer, 0, NULL);
  free(buffer);
  return ok;
}
