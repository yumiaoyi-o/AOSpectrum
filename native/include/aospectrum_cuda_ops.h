#ifndef AOSPECTRUM_CUDA_OPS_H
#define AOSPECTRUM_CUDA_OPS_H

#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int aospectrum_cuda_gather_shifted_real32(
      const float *hamiltonian_values,
      const float *overlap_values,
      const int32_t *source_positions,
      int64_t count,
      float shift_hartree,
      float *shifted_values,
      cudaStream_t stream,
      char *error_message,
      size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
