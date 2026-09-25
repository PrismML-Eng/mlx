// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_aligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    loader_a.load_unsafe();
    loader_b.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <
    bool rows_aligned,
    bool cols_aligned,
    bool transpose,
    typename T,
    typename mma_t,
    typename loader_a_t,
    typename loader_b_t>
METAL_FUNC void gemm_loop_unaligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations,
    const short tgp_bm,
    const short tgp_bn,
    const short tgp_bk) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    if (rows_aligned) {
      loader_a.load_unsafe();
    } else {
      loader_a.load_safe(short2(tgp_bk, tgp_bm));
    }
    if (cols_aligned) {
      loader_b.load_unsafe();
    } else {
      loader_b.load_safe(
          transpose ? short2(tgp_bk, tgp_bn) : short2(tgp_bn, tgp_bk));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_finalize(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const short2 tile_a,
    const short2 tile_b) {
  loader_a.load_safe(tile_a);
  loader_b.load_safe(tile_b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.mma(As, Bs);
}

///////////////////////////////////////////////////////////////////////////////
// Few-row packed matmul core (M <= 16 rows), 2-bit affine, group size 128.
//
// One call computes y[v][col0 + c] for v < rows (<= 16) and c < 32 over the
// K partition [k0, k0 + Kp): a 16 x 32 output block. The KS simdgroups that
// share a column block split the partition's 128-wide quantization groups
// round-robin (no threadgroup traffic inside the K loop, no per-step
// barriers) and their partial sums are added through threadgroup memory at
// the end. Weights are read as 16-byte words (64 k per weight row), one
// whole group ahead of the block being consumed, and every weight is
// dequantized straight into the tensor unit's right-operand fragment, so
// each weight is decoded exactly once per threadgroup. Fragment layout,
// descriptor and cooperative-tensor copies follow steel/gemm/nax.h
// (BaseNAXFrag::get_coord) as used by qmm_t_nax and the split-K NAX body.
//
// Numerics: products run at the tensor unit's precision for the input type
// (half x half -> fp32 accumulate for a half activation; fp32 inputs are
// TF32-class), the same classes qmm_t_nax and the split-K NAX body use.
///////////////////////////////////////////////////////////////////////////////
#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 400) && \
    defined(__has_include)
#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#define MLX_QMM_M16_NAX 1
#endif
#endif
#ifdef MLX_QMM_M16_NAX
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#ifndef QMM_M16_DBG_NODQ
#define QMM_M16_DBG_NODQ 0
#endif
#ifndef QMM_M16_DBG_NOA
#define QMM_M16_DBG_NOA 0
#endif
#ifndef QMM_M16_DBG_NOMMA
#define QMM_M16_DBG_NOMMA 0
#endif
#ifndef QMM_M16_PF
#define QMM_M16_PF 1
#endif

template <typename U>
using qmm_m16_frag_t = typename metal::vec<U, 8>;

// T: element type of x and y (half or float). KS: simdgroups per 32-column
// block (2 or 4); `ks` is this simdgroup's slice index. `red0` and `red1`
// each hold 16 * 32 floats; with KS == 2 only `red0` is used.
template <typename T, int KS>
METAL_FUNC void qmm_m16_block(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const int K,
    const int N,
    const int rows,
    const int col0,
    const int k0,
    const int Kp,
    const uint ks,
    const uint simd_lid,
    threadgroup float* red0,
    threadgroup float* red1) {
  constexpr int GS = 128;
  typedef qmm_m16_frag_t<T> frag_t;
  typedef qmm_m16_frag_t<float> cfrag_t;

  const int K_w = K / 16; // uint32 words per weight row
  const int K_g = K / GS;

  // Fragment coordinate of this lane (BaseNAXFrag::get_coord): elements 0..3
  // sit at (fm, fn..fn+3), elements 4..7 at (fm + 8, fn..fn+3).
  const short qid = simd_lid >> 2;
  const short fm = ((qid & 4) | ((simd_lid >> 1) & 3));
  const short fn = ((qid & 2) | (simd_lid & 1)) * 4;
  // This lane's four k values of a 16-value word are byte (fn / 4).
  const ushort bsh = ushort(8 * (fn >> 2));

  // Weight rows col0 + fm + 8 * j (j = 0, 1 -> B0; 2, 3 -> B1), clamped.
  int wrow[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    wrow[j] = min(col0 + int(fm) + 8 * j, N - 1);
  }
  // Input rows fm and fm + 8, clamped to the live rows (never stored).
  const device T* xa0 = x + min(int(fm), rows - 1) * K + fn;
  const device T* xa1 = x + min(int(fm) + 8, rows - 1) * K + fn;

  // The accumulator lives in the tensor op's destination cooperative tensor
  // for the whole K loop; the fragments are copied out once at the end.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      true,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
#pragma unroll
  for (short i = 0; i < 16; i++) {
    ct_c[i] = 0.0f;
  }

  const int g_begin = k0 / GS;
  const int n_groups = Kp / GS;

  // Blocks of 256 k (two groups): per weight row the four lanes of a
  // fragment quad (same fm, fn = 0, 4, 8, 12) load four consecutive uint4,
  // i.e. one contiguous 64-byte line of the row per load instruction; at
  // each 16-k step the word holding this lane's four values is taken from
  // the quad lane that loaded it (simd_shuffle). QMM_M16_PF blocks are kept
  // in flight ahead of the block being consumed; the volatile barrier keeps
  // the compiler from sinking the prefetch loads down to their use.
  const ushort wq = ushort(fn >> 2); // this lane's uint4 within the line
  ushort qlane[4];
#pragma unroll
  for (int st = 0; st < 4; st++) {
    qlane[st] =
        ushort((simd_lid & ~0x9u) | uint(st & 1) | (uint(st >> 1) << 3));
  }
  // Blocks are numbered per simdgroup: block i covers groups
  // g_begin + 2 * (ks + i * KS) and the one after it.
  const int n_blocks_total = (n_groups + 1) / 2;
  const int my_blocks = max((n_blocks_total - int(ks) + KS - 1) / KS, 0);
  auto block_line = [&](int i, int j) -> uint4 {
    const int gb = g_begin + 2 * (int(ks) + i * KS);
    return *((const device uint4*)(w + wrow[j] * K_w + gb * 8) + wq);
  };
  uint4 ring[QMM_M16_PF + 1][4];
#pragma unroll
  for (int r = 0; r < QMM_M16_PF + 1; r++) {
    if (r < my_blocks) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        ring[r][j] = block_line(r, j);
      }
    }
  }

  for (int i = 0; i < my_blocks; i++) {
    const int gb = g_begin + 2 * (int(ks) + i * KS);
    volatile int compiler_barrier;
#pragma unroll
    for (int gh = 0; gh < 2; gh++) {
      const int g = gb + gh;
      if (g - g_begin >= n_groups) {
        break;
      }
      float s0[4];
      float s1[4];
      float s2[4];
      float s3[4];
      float b[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float s = float(scales[wrow[j] * K_g + g]);
        b[j] = float(biases[wrow[j] * K_g + g]);
        s0[j] = s;
        s1[j] = s * 0.25f;
        s2[j] = s * 0.0625f;
        s3[j] = s * 0.015625f;
      }
#pragma unroll
      for (int st8 = 0; st8 < 8; st8++) {
        const int st = gh * 8 + st8; // 0..15 within the block
        const int k = g * GS + st8 * 16;
        frag_t B0;
        frag_t B1;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint word = simd_shuffle(ring[0][j][st & 3], qlane[st >> 2]);
          const uint by = (word >> bsh) & 0xffu;
#if QMM_M16_DBG_NODQ
          const float v0 = float(by);
          const float v1 = float(by >> 1);
          const float v2 = float(by >> 2);
          const float v3 = float(by >> 3);
#else
          const float v0 = s0[j] * float(by & 0x03u) + b[j];
          const float v1 = s1[j] * float(by & 0x0cu) + b[j];
          const float v2 = s2[j] * float(by & 0x30u) + b[j];
          const float v3 = s3[j] * float(by & 0xc0u) + b[j];
#endif
          if (j < 2) {
            B0[4 * j + 0] = T(v0);
            B0[4 * j + 1] = T(v1);
            B0[4 * j + 2] = T(v2);
            B0[4 * j + 3] = T(v3);
          } else {
            B1[4 * (j - 2) + 0] = T(v0);
            B1[4 * (j - 2) + 1] = T(v1);
            B1[4 * (j - 2) + 2] = T(v2);
            B1[4 * (j - 2) + 3] = T(v3);
          }
        }
#if QMM_M16_DBG_NOA
#pragma unroll
        for (int q = 0; q < 4; q++) {
          ct_a[q] = T(float(k + q) * 0.001f);
          ct_a[4 + q] = T(float(k - q) * 0.001f);
        }
#else
#pragma unroll
        for (int q = 0; q < 4; q++) {
          ct_a[q] = xa0[k + q];
          ct_a[4 + q] = xa1[k + q];
        }
#endif
#pragma unroll
        for (short q = 0; q < 8; q++) {
          ct_b[q] = B0[q];
          ct_b[8 + q] = B1[q];
        }
#if QMM_M16_DBG_NOMMA
        if (st8 == 7) {
          gemm_op.run(ct_a, ct_b, ct_c);
        }
#else
        gemm_op.run(ct_a, ct_b, ct_c);
#endif
      }
    }
    (void)compiler_barrier;
#pragma unroll
    for (int r = 0; r < QMM_M16_PF; r++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        ring[r][j] = ring[r + 1][j];
      }
    }
    if (i + QMM_M16_PF + 1 < my_blocks) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        ring[QMM_M16_PF][j] = block_line(i + QMM_M16_PF + 1, j);
      }
    }
  }

  cfrag_t C0;
  cfrag_t C1;
#pragma unroll
  for (short i = 0; i < 8; i++) {
    C0[i] = ct_c[i];
    C1[i] = ct_c[8 + i];
  }
  // Reduce the KS partials onto slice 0: 1 -> 0 (red0) and 3 -> 2 (red1),
  // then 2 -> 0 (red0). Identical fragment layouts line up element by
  // element.
  {
    threadgroup float* red = (ks & 2) ? red1 : red0;
    if (ks & 1) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        red[i * 32 + simd_lid] = C0[i];
        red[(8 + i) * 32 + simd_lid] = C1[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!(ks & 1)) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        C0[i] += red[i * 32 + simd_lid];
        C1[i] += red[(8 + i) * 32 + simd_lid];
      }
    }
    if (KS == 4) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ks == 2) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          red0[i * 32 + simd_lid] = C0[i];
          red0[(8 + i) * 32 + simd_lid] = C1[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ks == 0) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          C0[i] += red0[i * 32 + simd_lid];
          C1[i] += red0[(8 + i) * 32 + simd_lid];
        }
      }
    }
  }
  if (ks != 0) {
    return;
  }
  // C0[i] holds rows fm + (i / 4) * 8, columns col0 + fn + i % 4; C1 the
  // columns 16 further.
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int v = int(fm) + (i / 4) * 8;
    const int c = col0 + int(fn) + (i % 4);
    if (v < rows) {
      if (c < N) {
        y[v * N + c] = static_cast<T>(C0[i]);
      }
      if (c + 16 < N) {
        y[v * N + c + 16] = static_cast<T>(C1[i]);
      }
    }
  }
}
#endif // MLX_QMM_M16_NAX
