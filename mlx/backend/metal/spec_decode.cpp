// Copyright © 2024 Apple Inc.

#include <algorithm>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

bool SpecDecodeVerify::use_fallback(Stream s) {
  // Fused kernel only on the GPU; CPU uses the op-level composition fallback.
  return s.device == Device::cpu;
}

void SpecDecodeVerify::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& draft = inputs[0]; // [B, K] int32
  auto& target = inputs[1]; // [B, K+1] int32
  auto& n_accepted = outputs[0]; // [B] int32
  auto& committed = outputs[1]; // [B, K+1] int32

  n_accepted.set_data(allocator::malloc(n_accepted.nbytes()));
  committed.set_data(allocator::malloc(committed.nbytes()));

  int B = draft.shape(0);
  int K = draft.shape(1);

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto kernel = d.get_kernel("spec_decode_verify");
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(draft, 0);
  compute_encoder.set_input_array(target, 1);
  compute_encoder.set_output_array(n_accepted, 2);
  compute_encoder.set_output_array(committed, 3);
  compute_encoder.set_bytes(K, 4);
  compute_encoder.set_bytes(B, 5);

  // One thread per batch row (K is tiny; no reduction).
  int tgroup = std::min(B, 256);
  MTL::Size grid_dims = MTL::Size(B, 1, 1);
  MTL::Size group_dims = MTL::Size(tgroup, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
