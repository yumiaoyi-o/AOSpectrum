#ifndef AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_H
#define AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_ABI_VERSION 1u
#define AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_ERROR_CAPACITY 512u

typedef struct {
   float real;
   float imag;
} aospectrum_ubatch_complex64;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   int64_t n;
   int64_t nnz;
   int32_t batch_count;
   int32_t rhs_count;
   const int32_t *indptr;
   const int32_t *indices;
   const aospectrum_ubatch_complex64 *values;
   const aospectrum_ubatch_complex64 *rhs;
   aospectrum_ubatch_complex64 *solution;
} aospectrum_cudss_complex64_ubatch_request_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   int32_t status;
   int32_t cudss_status;
   int64_t n;
   int64_t nnz;
   int64_t diagonal_count;
   int64_t reported_factor_nnz;
   int32_t reported_pivot_count;
   int32_t batch_count;
   int32_t rhs_count;
   int32_t iterative_refinement_steps;
   double upload_seconds;
   double descriptor_setup_seconds;
   double analysis_seconds;
   double factorization_seconds;
   double solve_seconds;
   double download_seconds;
   double total_seconds;
   uint64_t permanent_device_bytes;
   uint64_t peak_device_bytes;
   uint64_t permanent_host_bytes;
   uint64_t peak_host_bytes;
   uint64_t free_device_bytes_before;
   uint64_t free_device_bytes_after_upload;
   uint64_t free_device_bytes_after_factorization;
   uint64_t free_device_bytes_after_solve;
   uint64_t total_device_bytes;
   char error_message[AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_ERROR_CAPACITY];
} aospectrum_cudss_complex64_ubatch_result_v1;

int aospectrum_cudss_solve_complex64_ubatch_v1(
      const aospectrum_cudss_complex64_ubatch_request_v1 *request,
      aospectrum_cudss_complex64_ubatch_result_v1 *result);

#ifdef __cplusplus
}
#endif

#endif
