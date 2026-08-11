#ifndef AOSPECTRUM_PRIMME_CUDA_H
#define AOSPECTRUM_PRIMME_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AOSPECTRUM_PRIMME_CUDA_ABI_VERSION 1u
#define AOSPECTRUM_PRIMME_CUDA_ERROR_CAPACITY 512u

#define AOSPECTRUM_PRIMME_PRECONDITIONER_NONE 0
#define AOSPECTRUM_PRIMME_PRECONDITIONER_DIAGONAL 1
#define AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT 2

#define AOSPECTRUM_PRIMME_TARGET_CLOSEST_ABS 0
#define AOSPECTRUM_PRIMME_TARGET_CLOSEST_LEQ 1
#define AOSPECTRUM_PRIMME_TARGET_CLOSEST_GEQ 2

#define AOSPECTRUM_PRIMME_OPERATOR_HAMILTONIAN 0
#define AOSPECTRUM_PRIMME_OPERATOR_OVERLAP 1

/*
 * The v1 session is deliberately limited to real float32 Gamma-point
 * generalized Hermitian problems. A session is single-threaded: callers must
 * not update or solve it concurrently.
 */
typedef struct aospectrum_primme_cuda_real32_session_v1
   aospectrum_primme_cuda_real32_session_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   int64_t n;
   int64_t nnz;
   const int32_t *indptr;
   const int32_t *indices;
} aospectrum_primme_cuda_session_create_request_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   const float *hamiltonian_values;
   const float *overlap_values;
   float target_shift_hartree;
   int32_t preconditioner_kind;
   float diagonal_floor_hartree;
   int32_t maximum_preconditioner_block_size;
} aospectrum_primme_cuda_numeric_update_request_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   const float *device_hamiltonian_values;
   const float *device_overlap_values;
   uintptr_t producer_stream;
   float hamiltonian_scale_to_hartree;
} aospectrum_primme_cuda_device_values_request_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   float target_shift_hartree;
   int32_t preconditioner_kind;
   float diagonal_floor_hartree;
   int32_t maximum_preconditioner_block_size;
} aospectrum_primme_cuda_resident_factor_request_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   uint64_t numeric_epoch;
   int32_t num_evals;
   int32_t target_mode;
   float target_shift_hartree;
   float tolerance;
   int64_t max_matvecs;
   int32_t max_basis_size;
   int32_t min_restart_size;
   int32_t max_block_size;
   int32_t init_size;
   const float *initial_vectors;
   int32_t return_vectors;
   int32_t print_level;
} aospectrum_primme_cuda_solve_request_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   uint64_t numeric_epoch;
   int32_t operator_kind;
   int32_t block_size;
} aospectrum_primme_cuda_operator_request_v1;

typedef struct {
   uint32_t abi_version;
   uint32_t struct_size;
   int32_t status;
   int32_t primme_status;
   int32_t converged;
   int32_t target_mode;
   uint64_t numeric_epoch;
   uint64_t factor_numeric_epoch;
   uint64_t solve_index;
   int32_t factor_reused;
   int32_t preconditioner_kind;
   float target_shift_hartree;
   int32_t reserved;
   int64_t outer_iterations;
   int64_t restarts;
   int64_t matvecs;
   int64_t preconditions;
   double upload_seconds;
   double solve_seconds;
   double download_seconds;
   double primme_matvec_seconds;
   double primme_precondition_seconds;
   double primme_orthogonalization_seconds;
   uint64_t static_device_bytes;
   uint64_t free_device_bytes_before_solve;
   uint64_t free_device_bytes_after_solve;
   uint64_t total_device_bytes;
   int64_t cudss_upper_nnz;
   int64_t cudss_factor_nnz;
   int64_t cudss_solved_rhs;
   int32_t cudss_pivot_count;
   int32_t cudss_inertia_positive;
   int32_t cudss_inertia_negative;
   int32_t cudss_solve_calls;
   double cudss_preparation_seconds;
   double cudss_upload_seconds;
   double cudss_analysis_seconds;
   double cudss_factorization_seconds;
   double cudss_solve_seconds;
   uint64_t cudss_permanent_device_bytes;
   uint64_t cudss_peak_device_bytes;
   uint64_t cudss_permanent_host_bytes;
   uint64_t cudss_peak_host_bytes;
   uint64_t free_device_bytes_after_factorization;
   char error_message[AOSPECTRUM_PRIMME_CUDA_ERROR_CAPACITY];
} aospectrum_primme_cuda_result_v1;

int aospectrum_primme_cuda_real32_session_create_v1(
      const aospectrum_primme_cuda_session_create_request_v1 *request,
      aospectrum_primme_cuda_real32_session_v1 **session,
      aospectrum_primme_cuda_result_v1 *result);

/*
 * Numeric update is transactional from the solver's perspective. It
 * invalidates any old preconditioner first, uploads both value arrays, rebuilds
 * the requested preconditioner at target_shift_hartree, and only then commits
 * a new numeric_epoch. A failed update leaves the session unable to solve
 * until a later update succeeds.
 */
int aospectrum_primme_cuda_real32_session_update_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_numeric_update_request_v1 *request,
      aospectrum_primme_cuda_result_v1 *result);

/*
 * Admit one frame's device-resident H/S values. The call orders against the
 * producer stream, copies the values device-to-device into session-owned
 * buffers, and synchronizes before returning so the borrowed source buffers
 * may be released immediately.
 */
int aospectrum_primme_cuda_real32_session_load_device_values_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_device_values_request_v1 *request,
      aospectrum_primme_cuda_result_v1 *result);

/*
 * Factor a new shift from the resident H/S values. The first call creates the
 * cuDSS symbolic analysis; later calls with the same session topology and
 * block size reuse it and perform numerical factorization only.
 */
int aospectrum_primme_cuda_real32_session_factor_resident_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_resident_factor_request_v1 *request,
      aospectrum_primme_cuda_result_v1 *result);

/*
 * The solve request must repeat the committed numeric_epoch and exact target
 * shift. Multiple sequential solves may therefore reuse one cuDSS factor while
 * choosing closest_abs, closest_leq, or closest_geq independently.
 */
int aospectrum_primme_cuda_real32_session_solve_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_solve_request_v1 *request,
      float *eigenvalues_hartree,
      float *eigenvectors,
      float *residual_norms,
      aospectrum_primme_cuda_result_v1 *result);

/*
 * Apply the exact device CSR operator used by PRIMME to a host column-major
 * block. This is a diagnostic boundary: it does not alter the numeric epoch,
 * factor, or solve lifecycle.
 */
int aospectrum_primme_cuda_real32_session_apply_operator_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_operator_request_v1 *request,
      const float *input_vectors,
      float *output_vectors,
      aospectrum_primme_cuda_result_v1 *result);

void aospectrum_primme_cuda_real32_session_destroy_v1(
      aospectrum_primme_cuda_real32_session_v1 *session);

#ifdef __cplusplus
}
#endif

#endif
