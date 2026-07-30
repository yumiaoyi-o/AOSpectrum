#ifndef AOSPECTRUM_CUDSS_SHIFT_FACTOR_H
#define AOSPECTRUM_CUDSS_SHIFT_FACTOR_H

#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aospectrum_cudss_shift_factor aospectrum_cudss_shift_factor;

typedef struct {
   int64_t upper_nnz;
   int64_t factor_nnz;
   int32_t pivot_count;
   int32_t inertia_positive;
   int32_t inertia_negative;
   int32_t solve_calls;
   int64_t solved_rhs;
   double preparation_seconds;
   double upload_seconds;
   double analysis_seconds;
   double factorization_seconds;
   double solve_seconds;
   uint64_t permanent_device_bytes;
   uint64_t peak_device_bytes;
   uint64_t permanent_host_bytes;
   uint64_t peak_host_bytes;
   uint64_t free_device_bytes_before_factorization;
   uint64_t free_device_bytes_after_factorization;
} aospectrum_cudss_shift_factor_stats;

int aospectrum_cudss_shift_factor_create(
      aospectrum_cudss_shift_factor **factor,
      int64_t n,
      int64_t nnz,
      const int32_t *indptr,
      const int32_t *indices,
      const float *hamiltonian_values,
      const float *overlap_values,
      float shift_hartree,
      int32_t maximum_rhs,
      cudaStream_t stream,
      char *error_message,
      size_t error_capacity);

int aospectrum_cudss_shift_factor_solve(
      aospectrum_cudss_shift_factor *factor,
      const float *rhs,
      int64_t leading_rhs,
      float *solution,
      int64_t leading_solution,
      int32_t rhs_count,
      char *error_message,
      size_t error_capacity);

void aospectrum_cudss_shift_factor_get_stats(
      const aospectrum_cudss_shift_factor *factor,
      aospectrum_cudss_shift_factor_stats *stats);

void aospectrum_cudss_shift_factor_destroy(
      aospectrum_cudss_shift_factor *factor);

#ifdef __cplusplus
}
#endif

#endif
