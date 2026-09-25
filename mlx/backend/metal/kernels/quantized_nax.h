// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 1) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 2.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 8.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 32.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 128.0f;
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 1) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 2.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 8.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 32.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 128.0f;
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 1) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      accum +=
          (x_thread[8 * i] * (w[i] & 0x01) +
           x_thread[8 * i + 1] * (w[i] & 0x02) +
           x_thread[8 * i + 2] * (w[i] & 0x04) +
           x_thread[8 * i + 3] * (w[i] & 0x08) +
           x_thread[8 * i + 4] * (w[i] & 0x10) +
           x_thread[8 * i + 5] * (w[i] & 0x20) +
           x_thread[8 * i + 6] * (w[i] & 0x40) +
           x_thread[8 * i + 7] * (w[i] & 0x80));
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 1) {
    for (int i = 0; i < (N / 8); i++) {
      accum +=
          (x_thread[8 * i] * (w[i] & 0x01) +
           x_thread[8 * i + 1] * (w[i] & 0x02) +
           x_thread[8 * i + 2] * (w[i] & 0x04) +
           x_thread[8 * i + 3] * (w[i] & 0x08) +
           x_thread[8 * i + 4] * (w[i] & 0x10) +
           x_thread[8 * i + 5] * (w[i] & 0x20) +
           x_thread[8 * i + 6] * (w[i] & 0x40) +
           x_thread[8 * i + 7] * (w[i] & 0x80));
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  if (bits == 1) {
    U s[8] = {
        scale,
        scale / 2.0f,
        scale / 4.0f,
        scale / 8.0f,
        scale / 16.0f,
        scale / 32.0f,
        scale / 64.0f,
        scale / 128.0f};
    for (int i = 0; i < (values_per_thread / 8); i++) {
      result[8 * i] += x * (s[0] * (w[i] & 0x01) + bias);
      result[8 * i + 1] += x * (s[1] * (w[i] & 0x02) + bias);
      result[8 * i + 2] += x * (s[2] * (w[i] & 0x04) + bias);
      result[8 * i + 3] += x * (s[3] * (w[i] & 0x08) + bias);
      result[8 * i + 4] += x * (s[4] * (w[i] & 0x10) + bias);
      result[8 * i + 5] += x * (s[5] * (w[i] & 0x20) + bias);
      result[8 * i + 6] += x * (s[6] * (w[i] & 0x40) + bias);
      result[8 * i + 7] += x * (s[7] * (w[i] & 0x80) + bias);
    }
  }

  else if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <typename U, int N, int bits>
inline void
dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  const float s = float(scale);
  const float b = float(bias);

  if (bits == 1) {
    for (int i = 0; i < (N / 8); i++) {
      for (int j = 0; j < 8; j++) {
        w_local[8 * i + j] = static_cast<U>(s * ((w[i] >> j) & 1) + b);
      }
    }
  } else if (bits == 2) {
    float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = static_cast<U>(sc[0] * (w[i] & 0x03) + b);
      w_local[4 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0x0c) + b);
      w_local[4 * i + 2] = static_cast<U>(sc[2] * (w[i] & 0x30) + b);
      w_local[4 * i + 3] = static_cast<U>(sc[3] * (w[i] & 0xc0) + b);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = static_cast<U>((w[0] & 0x7) * s + b);
      w_local[1] = static_cast<U>(((w[0] & 0x38) >> 3) * s + b);
      w_local[2] =
          static_cast<U>((((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * s + b);
      w_local[3] = static_cast<U>(((w[1] & 0xe) >> 1) * s + b);
      w_local[4] = static_cast<U>(((w[1] & 0x70) >> 4) * s + b);
      w_local[5] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * s + b);
      w_local[6] = static_cast<U>(((w[2] & 0x1c) >> 2) * s + b);
      w_local[7] = static_cast<U>(((w[2] & 0xe0) >> 5) * s + b);
    }
  }

  else if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = static_cast<U>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<U>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<U>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<U>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<U>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<U>(((w[4] & 0xf8) >> 3) * s + b);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<U>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<U>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<U>(((w[2] >> 2) & 0x3f) * s + b);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<U>(s * w[i] + b);
    }
  }
}

// Same values as four dequantize<U, 4, 2> calls on the bytes of `word`.
template <typename U>
inline void
dequantize_2bit_word(uint32_t word, U scale, U bias, threadgroup U* w_local) {
  const float s = float(scale);
  const float b = float(bias);
  float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
  for (int i = 0; i < 4; i++) {
    const uint8_t wb = static_cast<uint8_t>((word >> (8 * i)) & 0xff);
    w_local[4 * i] = static_cast<U>(sc[0] * (wb & 0x03) + b);
    w_local[4 * i + 1] = static_cast<U>(sc[1] * (wb & 0x0c) + b);
    w_local[4 * i + 2] = static_cast<U>(sc[2] * (wb & 0x30) + b);
    w_local[4 * i + 3] = static_cast<U>(sc[3] * (wb & 0xc0) + b);
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short bits>
struct QuantizedBlockLoader<
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size,
    32,
    bits> {
  MLX_MTL_CONST short group_size = 32;

  static_assert(
      BCOLS % group_size == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  static_assert(
      (BCOLS_PACKED / n_reads) == n_groups,
      "Other configurations are not yet supported");

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id),
        biases(biases_ + bi * src_ld / group_size + group_id) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      // if (group_steps > 1) {
      //   group_step_cnt++;
      //   if (group_step_cnt == group_steps) {
      //     group_step_cnt = 0;
      //     scales++;
      //     biases++;
      //   }
      // } else {
      scales += n_groups;
      biases += n_groups;
      // }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_t_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  // Visit adjacent M tiles of each N tile so prompt-width calls reuse the
  // packed weight tile while it remains in cache. For a fixed NAX grid this
  // remaps each (M,N) tile exactly once; tid.z remains the batch index.
  // 16-bit activations only: with FP32 activations the remap measured 5-23%
  // slower at M=512 on an M5 Pro (2-bit gs128, N 5120..248320), and the
  // linear order 8-10% faster than the pre-port kernel.
  const int m_tiles = (M + BM - 1) / BM;
  const int n_tiles = (N + BN - 1) / BN;
  int y_row;
  int y_col;
  if (m_tiles == 1 || sizeof(T) != 2) {
    y_row = int(tid.y) * BM;
    y_col = int(tid.x) * BN;
  } else {
    const int tile_id = int(tid.y) * n_tiles + int(tid.x);
    y_row = (tile_id % m_tiles) * BM;
    y_col = (tile_id / m_tiles) * BN;
  }

#ifdef MLX_QMM_M16_NAX
  // Few-row tiles (M - y_row <= 16 with the host's 32-row tile): the shared
  // few-row core in quantized_utils.h. Simdgroups (0, 1) take columns
  // [0, 32) and (2, 3) columns [32, 64), each pair splitting K; partials
  // are summed through Ws, which this path does not otherwise use.
  if constexpr (
      bits == 2 && group_size == 128 && BM == 32 && BN == 64 && WM * WN == 4) {
    if (M - y_row <= 16 && N < 65536) {
      const uint cb = simd_gid >> 1;
      const uint ks = simd_gid & 1;
      threadgroup float* red = (threadgroup float*)Ws + cb * (16 * 32);
      qmm_m16_block<T, 2>(
          w,
          scales,
          biases,
          x + y_row * static_cast<int64_t>(K),
          y + y_row * static_cast<int64_t>(N),
          K,
          N,
          M - y_row,
          y_col + 32 * int(cb),
          0,
          K,
          ks,
          simd_lid,
          red,
          red);
      return;
    }
  }
#endif

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  // Verify-width inputs (M <= SM rows left in this tile, e.g. M=13..16 with
  // BM=32): the second row of simdgroups would multiply only zero rows.
  //   1: those simdgroups skip their MMAs (bit-identical output).
  //   2: all simdgroups take rows [0, SM); simdgroup row r does the K
  //      sub-steps kk1 = r*SK (mod BK) and the halves are summed through
  //      threadgroup memory at the end.
  constexpr int kNaxRowFit = 2;
  constexpr bool kRowFitShape = (BM == 32) && (WM == 2) && (BK == 2 * SK);
  const bool row_fit =
      kNaxRowFit != 0 && kRowFitShape && (M - y_row) <= int(SM);
  const bool k_half = kNaxRowFit == 2 && row_fit;

  const short tm = k_half ? short(0) : short(SM * (simd_gid / WN));
  const short tn = SN * (simd_gid % WN);
  const short k_row = short(simd_gid / WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  // Rows of MMA work this simdgroup does per BK step: [kk_first, BK) by
  // kk_step.
  const bool mma_idle = kNaxRowFit == 1 && row_fit && sgp_sm <= 0;
  const short kk_first =
      mma_idle ? short(BK) : (k_half ? short(k_row * SK) : short(0));
  const short kk_step = k_half ? short(BK) : short(SK);
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  // 2-bit: load the next K block of weights into registers before the MMAs.
  // Same thread mapping and same Ws values as loader_w.
  constexpr bool kPrefetchW2 = (bits == 2) && (BN == 64) && (BK == 64) &&
      (WM * WN == 4) && (group_size >= BK) && (group_size % BK == 0);
  const int pf_thr = int(simd_gid) * 32 + int(simd_lid);
  const int pf_row = pf_thr >> 1;
  const device uint32_t* pf_w =
      w + (y_col + pf_row) * (K / 16) + (pf_thr & 1) * 2;
  const device T* pf_s = scales + pf_row * K_g;
  const device T* pf_b = biases + pf_row * K_g;
  threadgroup T* pf_dst = Ws + pf_row * BK_padded + (pf_thr & 1) * 32;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      uint32_t pf_w0 = 0;
      uint32_t pf_w1 = 0;
      T pf_scale = T(0);
      T pf_bias = T(0);
      if constexpr (kPrefetchW2 && kAlignedN.value) {
        pf_w0 = pf_w[0];
        pf_w1 = pf_w[1];
        pf_scale = pf_s[0];
        pf_bias = pf_b[0];
      }

      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kPrefetchW2 && kAlignedN.value) {
          dequantize_2bit_word<T>(pf_w0, pf_scale, pf_bias, pf_dst);
          dequantize_2bit_word<T>(pf_w1, pf_scale, pf_bias, pf_dst + 16);
          if (k + BK < K) {
            const int pf_k = k + BK;
            const int pf_g = pf_k / group_size;
            pf_w0 = pf_w[pf_k / 16];
            pf_w1 = pf_w[pf_k / 16 + 1];
            pf_scale = pf_s[pf_g];
            pf_bias = pf_b[pf_g];
          }
        } else if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = kk_first; kk1 < BK; kk1 += kk_step) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        if constexpr (!(kPrefetchW2 && kAlignedN.value)) {
          loader_w.next();
        }
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kRowFitShape) {
        if (k_half) {
          // Simdgroup row 1 hands its partial sums to row 0 (same tn, same
          // lane layout); Ws is free after the barrier above.
          constexpr short kE = NAXTile<AccumType, TM, TN>::kElemsPerTile;
          static_assert(
              WN * 32 * kE * sizeof(AccumType) <= BN * BK_padded * sizeof(T),
              "k-half reduction must fit in Ws");
          threadgroup AccumType* red = (threadgroup AccumType*)Ws +
              (simd_gid % WN) * (32 * kE) + simd_lid;
          if (k_row != 0) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kE; i++) {
              red[i * 32] = Dtile.elems()[i];
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (k_row != 0) {
            return;
          }
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kE; i++) {
            Dtile.elems()[i] += red[i * 32];
          }
        } else if (mma_idle) {
          return;
        }
      }

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_n_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  // Here w is [K, N]: packed and group-quantized along N, with row stride N.
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, N, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short sgp_sm = min(int(SM), M - (y_row + tm));

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TK, TN> Btile;

      volatile int compiler_barrier;

      if (sgp_sm == SM) {
        Atile.load(x + kk1, K);
      } else {
        Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
      }

      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm == SM) {
    Dtile.store(y + tm * N + tn, N);
  } else {
    Dtile.store_safe(y + tm * N + tn, N, short2(SN, sgp_sm));
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 32,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false>
[[kernel]] void affine_qmm_t_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_qmm_n_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false>
[[kernel]] void affine_gather_qmm_t_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_gather_qmm_n_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void affine_gather_qmm_rhs_nax(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = align_M ? SM : min(int(SM), max(0, M - (y_row + tm)));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = m_hi_lim > m_lo_lim;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }

          xn += BK;
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              const short psk = min(int(SK), max(0, (k_remain - kk1)));
              Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store results to device memory
        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
}
