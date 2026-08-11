#include "aospectrum_cuda_ops.h"

#include <stdio.h>

namespace {

__global__ void gather_shifted_real32(
      const float *hamiltonian_values,
      const float *overlap_values,
      const int32_t *source_positions,
      int64_t count,
      float shift_hartree,
      float *shifted_values) {
   const int64_t output =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (output >= count) {
      return;
   }
   const int32_t source = source_positions[output];
   shifted_values[output] =
      hamiltonian_values[source] -
      shift_hartree * overlap_values[source];
}

}  // namespace

extern "C" int aospectrum_cuda_gather_shifted_real32(
      const float *hamiltonian_values,
      const float *overlap_values,
      const int32_t *source_positions,
      int64_t count,
      float shift_hartree,
      float *shifted_values,
      cudaStream_t stream,
      char *error_message,
      size_t error_capacity) {
   if (hamiltonian_values == nullptr || overlap_values == nullptr ||
         source_positions == nullptr || shifted_values == nullptr ||
         count <= 0) {
      if (error_message != nullptr && error_capacity > 0) {
         snprintf(
            error_message,
            error_capacity,
            "invalid device shifted-value request");
      }
      return 1;
   }
   constexpr int threads = 256;
   const int blocks = static_cast<int>((count + threads - 1) / threads);
   gather_shifted_real32<<<blocks, threads, 0, stream>>>(
      hamiltonian_values,
      overlap_values,
      source_positions,
      count,
      shift_hartree,
      shifted_values);
   const cudaError_t status = cudaGetLastError();
   if (status != cudaSuccess) {
      if (error_message != nullptr && error_capacity > 0) {
         snprintf(
            error_message,
            error_capacity,
            "shifted-value kernel launch failed: %s",
            cudaGetErrorString(status));
      }
      return 1;
   }
   return 0;
}
