#define _POSIX_C_SOURCE 200809L

#include "aospectrum_primme_cuda.h"
#include "aospectrum_cudss_shift_factor.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <limits.h>
#include <math.h>
#include <primme.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
   cusparseHandle_t handle;
   cusparseSpMatDescr_t descriptor;
   void *workspace;
   size_t workspace_size;
   int64_t n;
   int last_error;
} sparse_operator;

typedef struct {
   cublasHandle_t handle;
   const float *inverse_diagonal;
   int64_t n;
   int last_error;
} diagonal_preconditioner;

typedef struct {
   aospectrum_cudss_shift_factor *factor;
   int last_error;
   char error_message[AOSPECTRUM_PRIMME_CUDA_ERROR_CAPACITY];
} cudss_shift_preconditioner;

struct aospectrum_primme_cuda_real32_session_v1 {
   int64_t n;
   int64_t nnz;
   int device_ordinal;
   int32_t *host_indptr;
   int32_t *host_indices;
   int32_t *device_indptr;
   int32_t *device_indices;
   float *device_hamiltonian;
   float *device_overlap;
   float *device_inverse_diagonal;
   cudaStream_t stream;
   cublasHandle_t cublas_handle;
   cusparseHandle_t cusparse_handle;
   sparse_operator hamiltonian;
   sparse_operator overlap;
   diagonal_preconditioner diagonal_preconditioner;
   cudss_shift_preconditioner cudss_preconditioner;
   uint64_t numeric_epoch;
   uint64_t factor_numeric_epoch;
   uint64_t solve_index;
   uint64_t solves_in_numeric_epoch;
   float target_shift_hartree;
   float diagonal_floor_hartree;
   int32_t maximum_preconditioner_block_size;
   int32_t factor_maximum_rhs;
   int32_t preconditioner_kind;
   uint64_t values_epoch;
   int values_ready;
   int numeric_ready;
};

static double monotonic_seconds(void) {
   struct timespec value;
   if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
      return 0.0;
   }
   return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static void initialize_result_v1(aospectrum_primme_cuda_result_v1 *result) {
   if (result == NULL) {
      return;
   }
   memset(result, 0, sizeof(*result));
   result->abi_version = AOSPECTRUM_PRIMME_CUDA_ABI_VERSION;
   result->struct_size = (uint32_t)sizeof(*result);
}

static void set_error_v1(
      aospectrum_primme_cuda_result_v1 *result,
      int status,
      const char *format,
      ...) {
   va_list arguments;
   if (result == NULL) {
      return;
   }
   result->status = status;
   va_start(arguments, format);
   vsnprintf(
      result->error_message,
      AOSPECTRUM_PRIMME_CUDA_ERROR_CAPACITY,
      format,
      arguments);
   va_end(arguments);
}

static int check_cuda_v1(
      cudaError_t status,
      aospectrum_primme_cuda_result_v1 *result,
      const char *operation) {
   if (status == cudaSuccess) {
      return 0;
   }
   set_error_v1(
      result,
      20,
      "%s failed: %s",
      operation,
      cudaGetErrorString(status));
   return 1;
}

static int check_cublas_v1(
      cublasStatus_t status,
      aospectrum_primme_cuda_result_v1 *result,
      const char *operation) {
   if (status == CUBLAS_STATUS_SUCCESS) {
      return 0;
   }
   set_error_v1(
      result,
      21,
      "%s failed with cuBLAS status %d",
      operation,
      (int)status);
   return 1;
}

static int check_cusparse_v1(
      cusparseStatus_t status,
      aospectrum_primme_cuda_result_v1 *result,
      const char *operation) {
   if (status == CUSPARSE_STATUS_SUCCESS) {
      return 0;
   }
   set_error_v1(
      result,
      22,
      "%s failed with cuSPARSE status %d",
      operation,
      (int)status);
   return 1;
}

static int require_device_pointer_v1(
      const void *pointer,
      int expected_device,
      aospectrum_primme_cuda_result_v1 *result,
      const char *name) {
   struct cudaPointerAttributes attributes;
   if (pointer == NULL) {
      set_error_v1(result, 12, "%s device pointer is null", name);
      return 1;
   }
   const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
   if (status != cudaSuccess) {
      set_error_v1(
         result,
         20,
         "%s pointer is not CUDA-visible: %s",
         name,
         cudaGetErrorString(status));
      return 1;
   }
#if CUDART_VERSION >= 10000
   if (attributes.type != cudaMemoryTypeDevice &&
         attributes.type != cudaMemoryTypeManaged) {
      set_error_v1(result, 12, "%s pointer is not device memory", name);
      return 1;
   }
   if (attributes.type == cudaMemoryTypeDevice &&
         attributes.device != expected_device) {
      set_error_v1(result, 12, "%s pointer belongs to another GPU", name);
      return 1;
   }
#else
   if (attributes.memoryType != cudaMemoryTypeDevice ||
         attributes.device != expected_device) {
      set_error_v1(result, 12, "%s pointer belongs to another GPU", name);
      return 1;
   }
#endif
   return 0;
}

static uint64_t session_static_device_bytes(
      const aospectrum_primme_cuda_real32_session_v1 *session) {
   uint64_t total;
   if (session == NULL) {
      return 0;
   }
   total =
      ((uint64_t)session->n + 1u) * sizeof(int32_t) +
      (uint64_t)session->nnz * sizeof(int32_t) +
      (uint64_t)session->nnz * sizeof(float) * 2u +
      (uint64_t)session->hamiltonian.workspace_size +
      (uint64_t)session->overlap.workspace_size;
   if (session->device_inverse_diagonal != NULL) {
      total += (uint64_t)session->n * sizeof(float);
   }
   return total;
}

static void fill_session_metadata(
      const aospectrum_primme_cuda_real32_session_v1 *session,
      aospectrum_primme_cuda_result_v1 *result) {
   if (session == NULL || result == NULL) {
      return;
   }
   result->numeric_epoch = session->numeric_epoch;
   result->factor_numeric_epoch = session->factor_numeric_epoch;
   result->solve_index = session->solve_index;
   result->preconditioner_kind = session->preconditioner_kind;
   result->target_shift_hartree = session->target_shift_hartree;
   if (result->static_device_bytes == 0) {
      result->static_device_bytes = session_static_device_bytes(session);
   }
}

static void record_cudss_stats_v1(
      const aospectrum_cudss_shift_factor_stats *stats,
      aospectrum_primme_cuda_result_v1 *result) {
   if (stats == NULL || result == NULL) {
      return;
   }
   result->cudss_upper_nnz = stats->upper_nnz;
   result->cudss_factor_nnz = stats->factor_nnz;
   result->cudss_solved_rhs = stats->solved_rhs;
   result->cudss_pivot_count = stats->pivot_count;
   result->cudss_inertia_positive = stats->inertia_positive;
   result->cudss_inertia_negative = stats->inertia_negative;
   result->cudss_solve_calls = stats->solve_calls;
   result->cudss_preparation_seconds = stats->preparation_seconds;
   result->cudss_upload_seconds = stats->upload_seconds;
   result->cudss_analysis_seconds = stats->analysis_seconds;
   result->cudss_factorization_seconds = stats->factorization_seconds;
   result->cudss_solve_seconds = stats->solve_seconds;
   result->cudss_permanent_device_bytes = stats->permanent_device_bytes;
   result->cudss_peak_device_bytes = stats->peak_device_bytes;
   result->cudss_permanent_host_bytes = stats->permanent_host_bytes;
   result->cudss_peak_host_bytes = stats->peak_host_bytes;
   result->free_device_bytes_after_factorization =
      stats->free_device_bytes_after_factorization;
}

static void record_session_cudss_stats(
      const aospectrum_primme_cuda_real32_session_v1 *session,
      aospectrum_primme_cuda_result_v1 *result) {
   aospectrum_cudss_shift_factor_stats stats = {0};
   if (session == NULL ||
         session->cudss_preconditioner.factor == NULL ||
         result == NULL) {
      return;
   }
   aospectrum_cudss_shift_factor_get_stats(
      session->cudss_preconditioner.factor,
      &stats);
   record_cudss_stats_v1(&stats, result);
}

static int validate_topology(
      const aospectrum_primme_cuda_session_create_request_v1 *request,
      aospectrum_primme_cuda_result_v1 *result) {
   if (request->n <= 0 || request->n > INT32_MAX ||
         request->nnz <= 0 || request->nnz > INT32_MAX) {
      set_error_v1(
         result,
         11,
         "matrix dimensions exceed the real32 single-card ABI");
      return 1;
   }
   if (request->indptr == NULL || request->indices == NULL) {
      set_error_v1(result, 12, "CSR topology pointer is null");
      return 1;
   }
   if (request->indptr[0] != 0 ||
         request->indptr[request->n] != request->nnz) {
      set_error_v1(
         result,
         13,
         "CSR row offsets do not span the declared nnz");
      return 1;
   }
   for (int64_t row = 0; row < request->n; ++row) {
      const int32_t begin = request->indptr[row];
      const int32_t end = request->indptr[row + 1];
      if (begin < 0 || end < begin || end > request->nnz) {
         set_error_v1(
            result,
            14,
            "CSR row offsets are invalid at row %lld",
            (long long)row);
         return 1;
      }
      for (int32_t position = begin; position < end; ++position) {
         const int32_t column = request->indices[position];
         if (column < 0 || column >= request->n) {
            set_error_v1(
               result,
               15,
               "CSR column index is invalid at position %d",
               position);
            return 1;
         }
      }
   }
   return 0;
}

static void sparse_matvec(
      sparse_operator *operator_value,
      void *x,
      PRIMME_INT ldx,
      void *y,
      PRIMME_INT ldy,
      int block_size,
      int *error) {
   cusparseDnMatDescr_t x_descriptor = NULL;
   cusparseDnMatDescr_t y_descriptor = NULL;
   float alpha = 1.0f;
   float beta = 0.0f;
   size_t required_workspace = 0;
   cusparseStatus_t status;

   *error = 0;
   status = cusparseCreateDnMat(
      &x_descriptor,
      operator_value->n,
      block_size,
      ldx,
      x,
      CUDA_R_32F,
      CUSPARSE_ORDER_COL);
   if (status != CUSPARSE_STATUS_SUCCESS) {
      goto failure;
   }
   status = cusparseCreateDnMat(
      &y_descriptor,
      operator_value->n,
      block_size,
      ldy,
      y,
      CUDA_R_32F,
      CUSPARSE_ORDER_COL);
   if (status != CUSPARSE_STATUS_SUCCESS) {
      goto failure;
   }
   status = cusparseSpMM_bufferSize(
      operator_value->handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      operator_value->descriptor,
      x_descriptor,
      &beta,
      y_descriptor,
      CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT,
      &required_workspace);
   if (status != CUSPARSE_STATUS_SUCCESS) {
      goto failure;
   }
   if (required_workspace > operator_value->workspace_size) {
      if (operator_value->workspace != NULL) {
         cudaFree(operator_value->workspace);
      }
      operator_value->workspace = NULL;
      operator_value->workspace_size = 0;
      if (cudaMalloc(
            &operator_value->workspace,
            required_workspace) != cudaSuccess) {
         status = CUSPARSE_STATUS_ALLOC_FAILED;
         goto failure;
      }
      operator_value->workspace_size = required_workspace;
   }
   status = cusparseSpMM(
      operator_value->handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      operator_value->descriptor,
      x_descriptor,
      &beta,
      y_descriptor,
      CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT,
      operator_value->workspace);
   if (status != CUSPARSE_STATUS_SUCCESS) {
      goto failure;
   }

cleanup:
   if (x_descriptor != NULL) {
      cusparseDestroyDnMat(x_descriptor);
   }
   if (y_descriptor != NULL) {
      cusparseDestroyDnMat(y_descriptor);
   }
   return;

failure:
   operator_value->last_error = (int)status;
   *error = 1;
   goto cleanup;
}

static void hamiltonian_matvec(
      void *x,
      PRIMME_INT *ldx,
      void *y,
      PRIMME_INT *ldy,
      int *block_size,
      primme_params *primme,
      int *error) {
   sparse_matvec(
      (sparse_operator *)primme->matrix,
      x,
      *ldx,
      y,
      *ldy,
      *block_size,
      error);
}

static void overlap_matvec(
      void *x,
      PRIMME_INT *ldx,
      void *y,
      PRIMME_INT *ldy,
      int *block_size,
      primme_params *primme,
      int *error) {
   sparse_matvec(
      (sparse_operator *)primme->massMatrix,
      x,
      *ldx,
      y,
      *ldy,
      *block_size,
      error);
}

static void apply_diagonal_preconditioner(
      void *x,
      PRIMME_INT *ldx,
      void *y,
      PRIMME_INT *ldy,
      int *block_size,
      primme_params *primme,
      int *error) {
   diagonal_preconditioner *preconditioner =
      (diagonal_preconditioner *)primme->preconditioner;
   const cublasStatus_t status = cublasSdgmm(
      preconditioner->handle,
      CUBLAS_SIDE_LEFT,
      (int)preconditioner->n,
      *block_size,
      (const float *)x,
      (int)*ldx,
      preconditioner->inverse_diagonal,
      1,
      (float *)y,
      (int)*ldy);
   if (status == CUBLAS_STATUS_SUCCESS) {
      *error = 0;
   }
   else {
      preconditioner->last_error = (int)status;
      *error = 1;
   }
}

static void apply_cudss_shift_preconditioner(
      void *x,
      PRIMME_INT *ldx,
      void *y,
      PRIMME_INT *ldy,
      int *block_size,
      primme_params *primme,
      int *error) {
   cudss_shift_preconditioner *preconditioner =
      (cudss_shift_preconditioner *)primme->preconditioner;
   preconditioner->error_message[0] = '\0';
   preconditioner->last_error = aospectrum_cudss_shift_factor_solve(
      preconditioner->factor,
      (const float *)x,
      *ldx,
      (float *)y,
      *ldy,
      *block_size,
      preconditioner->error_message,
      sizeof(preconditioner->error_message));
   *error = preconditioner->last_error;
}

static int build_inverse_diagonal(
      const aospectrum_primme_cuda_real32_session_v1 *session,
      const float *hamiltonian_values,
      const float *overlap_values,
      float shift_hartree,
      float diagonal_floor_hartree,
      float **host_inverse,
      aospectrum_primme_cuda_result_v1 *result) {
   float *inverse = (float *)malloc((size_t)session->n * sizeof(float));
   if (inverse == NULL) {
      set_error_v1(result, 30, "cannot allocate host diagonal preconditioner");
      return 1;
   }
   for (int64_t row = 0; row < session->n; ++row) {
      int found = 0;
      float h_diagonal = 0.0f;
      float s_diagonal = 0.0f;
      for (int32_t position = session->host_indptr[row];
            position < session->host_indptr[row + 1];
            ++position) {
         if (session->host_indices[position] == row) {
            h_diagonal = hamiltonian_values[position];
            s_diagonal = overlap_values[position];
            found = 1;
            break;
         }
      }
      if (!found) {
         free(inverse);
         set_error_v1(
            result,
            31,
            "CSR row %lld has no diagonal entry",
            (long long)row);
         return 1;
      }
      float denominator =
         h_diagonal - shift_hartree * s_diagonal;
      if (fabsf(denominator) < diagonal_floor_hartree) {
         denominator = copysignf(
            diagonal_floor_hartree,
            denominator == 0.0f ? 1.0f : denominator);
      }
      inverse[row] = 1.0f / denominator;
   }
   *host_inverse = inverse;
   return 0;
}

static void clear_numeric_commit(
      aospectrum_primme_cuda_real32_session_v1 *session) {
   if (session == NULL) {
      return;
   }
   if (session->device_inverse_diagonal != NULL) {
      cudaFree(session->device_inverse_diagonal);
      session->device_inverse_diagonal = NULL;
   }
   session->diagonal_preconditioner.inverse_diagonal = NULL;
   session->factor_numeric_epoch = 0;
   session->preconditioner_kind = AOSPECTRUM_PRIMME_PRECONDITIONER_NONE;
   session->target_shift_hartree = NAN;
   session->diagonal_floor_hartree = 0.0f;
   session->maximum_preconditioner_block_size = 0;
   session->solves_in_numeric_epoch = 0;
   session->numeric_ready = 0;
   session->cudss_preconditioner.last_error = 0;
   session->cudss_preconditioner.error_message[0] = '\0';
}

static void invalidate_numeric_state(
      aospectrum_primme_cuda_real32_session_v1 *session) {
   if (session == NULL) {
      return;
   }
   clear_numeric_commit(session);
   if (session->cudss_preconditioner.factor != NULL) {
      aospectrum_cudss_shift_factor_destroy(
         session->cudss_preconditioner.factor);
      session->cudss_preconditioner.factor = NULL;
   }
   session->factor_maximum_rhs = 0;
   session->values_ready = 0;
}

void aospectrum_primme_cuda_real32_session_destroy_v1(
      aospectrum_primme_cuda_real32_session_v1 *session) {
   if (session == NULL) {
      return;
   }
   if (session->stream != NULL) {
      cudaStreamSynchronize(session->stream);
   }
   invalidate_numeric_state(session);
   if (session->hamiltonian.workspace != NULL) {
      cudaFree(session->hamiltonian.workspace);
   }
   if (session->overlap.workspace != NULL) {
      cudaFree(session->overlap.workspace);
   }
   if (session->hamiltonian.descriptor != NULL) {
      cusparseDestroySpMat(session->hamiltonian.descriptor);
   }
   if (session->overlap.descriptor != NULL) {
      cusparseDestroySpMat(session->overlap.descriptor);
   }
   if (session->device_hamiltonian != NULL) {
      cudaFree(session->device_hamiltonian);
   }
   if (session->device_overlap != NULL) {
      cudaFree(session->device_overlap);
   }
   if (session->device_indices != NULL) {
      cudaFree(session->device_indices);
   }
   if (session->device_indptr != NULL) {
      cudaFree(session->device_indptr);
   }
   if (session->cusparse_handle != NULL) {
      cusparseDestroy(session->cusparse_handle);
   }
   if (session->cublas_handle != NULL) {
      cublasDestroy(session->cublas_handle);
   }
   if (session->stream != NULL) {
      cudaStreamDestroy(session->stream);
   }
   free(session->host_indices);
   free(session->host_indptr);
   free(session);
}

int aospectrum_primme_cuda_real32_session_create_v1(
      const aospectrum_primme_cuda_session_create_request_v1 *request,
      aospectrum_primme_cuda_real32_session_v1 **session_output,
      aospectrum_primme_cuda_result_v1 *result) {
   int return_code = 1;
   double started;
   aospectrum_primme_cuda_real32_session_v1 *session = NULL;

   if (result == NULL) {
      return 1;
   }
   initialize_result_v1(result);
   if (request == NULL || session_output == NULL) {
      set_error_v1(result, 10, "create request or session output is null");
      return 1;
   }
   *session_output = NULL;
   if (request->abi_version != AOSPECTRUM_PRIMME_CUDA_ABI_VERSION ||
         request->struct_size != sizeof(*request)) {
      set_error_v1(result, 10, "create request ABI mismatch");
      return 1;
   }
   if (validate_topology(request, result)) {
      return 1;
   }

   session = (aospectrum_primme_cuda_real32_session_v1 *)calloc(
      1,
      sizeof(*session));
   if (session == NULL) {
      set_error_v1(result, 16, "cannot allocate real32 session");
      return 1;
   }
   session->n = request->n;
   session->nnz = request->nnz;
   session->target_shift_hartree = NAN;
   if (check_cuda_v1(
         cudaGetDevice(&session->device_ordinal),
         result,
         "cudaGetDevice")) {
      goto cleanup;
   }
   session->host_indptr = (int32_t *)malloc(
      ((size_t)request->n + 1u) * sizeof(int32_t));
   session->host_indices = (int32_t *)malloc(
      (size_t)request->nnz * sizeof(int32_t));
   if (session->host_indptr == NULL || session->host_indices == NULL) {
      set_error_v1(result, 16, "cannot copy CSR topology");
      goto cleanup;
   }
   memcpy(
      session->host_indptr,
      request->indptr,
      ((size_t)request->n + 1u) * sizeof(int32_t));
   memcpy(
      session->host_indices,
      request->indices,
      (size_t)request->nnz * sizeof(int32_t));

   started = monotonic_seconds();
   /*
    * PRIMME's CUDA orthogonalization copies small Gram matrices through the
    * legacy default stream, so this stream must participate in its ordering.
    */
   if (check_cuda_v1(
         cudaStreamCreateWithFlags(
            &session->stream,
            cudaStreamDefault),
         result,
         "cudaStreamCreateWithFlags")) {
      goto cleanup;
   }
   if (check_cublas_v1(
         cublasCreate(&session->cublas_handle),
         result,
         "cublasCreate") ||
       check_cublas_v1(
         cublasSetStream(session->cublas_handle, session->stream),
         result,
         "cublasSetStream") ||
       check_cusparse_v1(
         cusparseCreate(&session->cusparse_handle),
         result,
         "cusparseCreate") ||
       check_cusparse_v1(
         cusparseSetStream(session->cusparse_handle, session->stream),
         result,
         "cusparseSetStream")) {
      goto cleanup;
   }
   if (check_cuda_v1(
         cudaMalloc(
            (void **)&session->device_indptr,
            ((size_t)request->n + 1u) * sizeof(int32_t)),
         result,
         "cudaMalloc(indptr)") ||
       check_cuda_v1(
         cudaMalloc(
            (void **)&session->device_indices,
            (size_t)request->nnz * sizeof(int32_t)),
         result,
         "cudaMalloc(indices)") ||
       check_cuda_v1(
         cudaMalloc(
            (void **)&session->device_hamiltonian,
            (size_t)request->nnz * sizeof(float)),
         result,
         "cudaMalloc(H)") ||
       check_cuda_v1(
         cudaMalloc(
            (void **)&session->device_overlap,
            (size_t)request->nnz * sizeof(float)),
         result,
         "cudaMalloc(S)") ||
       check_cuda_v1(
         cudaMemcpyAsync(
            session->device_indptr,
            session->host_indptr,
            ((size_t)request->n + 1u) * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            session->stream),
         result,
         "cudaMemcpyAsync(indptr)") ||
       check_cuda_v1(
         cudaMemcpyAsync(
            session->device_indices,
            session->host_indices,
            (size_t)request->nnz * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            session->stream),
         result,
         "cudaMemcpyAsync(indices)")) {
      goto cleanup;
   }

   session->hamiltonian.handle = session->cusparse_handle;
   session->hamiltonian.n = request->n;
   session->overlap.handle = session->cusparse_handle;
   session->overlap.n = request->n;
   if (check_cusparse_v1(
         cusparseCreateCsr(
            &session->hamiltonian.descriptor,
            request->n,
            request->n,
            request->nnz,
            session->device_indptr,
            session->device_indices,
            session->device_hamiltonian,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F),
         result,
         "cusparseCreateCsr(H)") ||
       check_cusparse_v1(
         cusparseCreateCsr(
            &session->overlap.descriptor,
            request->n,
            request->n,
            request->nnz,
            session->device_indptr,
            session->device_indices,
            session->device_overlap,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F),
         result,
         "cusparseCreateCsr(S)") ||
       check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(topology upload)")) {
      goto cleanup;
   }
   result->upload_seconds = monotonic_seconds() - started;
   result->status = 0;
   result->error_message[0] = '\0';
   fill_session_metadata(session, result);
   *session_output = session;
   session = NULL;
   return_code = 0;

cleanup:
   aospectrum_primme_cuda_real32_session_destroy_v1(session);
   return return_code;
}

int aospectrum_primme_cuda_real32_session_update_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_numeric_update_request_v1 *request,
      aospectrum_primme_cuda_result_v1 *result) {
   int return_code = 1;
   float *host_inverse_diagonal = NULL;
   double started;

   if (result == NULL) {
      return 1;
   }
   initialize_result_v1(result);
   if (session == NULL || request == NULL) {
      set_error_v1(result, 10, "numeric update session or request is null");
      return 1;
   }
   fill_session_metadata(session, result);
   if (request->abi_version != AOSPECTRUM_PRIMME_CUDA_ABI_VERSION ||
         request->struct_size != sizeof(*request)) {
      set_error_v1(result, 10, "numeric update request ABI mismatch");
      return 1;
   }
   if (request->hamiltonian_values == NULL ||
         request->overlap_values == NULL) {
      set_error_v1(result, 12, "numeric H/S value pointer is null");
      return 1;
   }
   if (!isfinite(request->target_shift_hartree)) {
      set_error_v1(result, 14, "numeric update target shift is invalid");
      return 1;
   }
   if (request->preconditioner_kind < AOSPECTRUM_PRIMME_PRECONDITIONER_NONE ||
         request->preconditioner_kind >
            AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT) {
      set_error_v1(result, 17, "preconditioner kind is invalid");
      return 1;
   }
   if (request->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_DIAGONAL &&
       (!isfinite(request->diagonal_floor_hartree) ||
        request->diagonal_floor_hartree <= 0.0f)) {
      set_error_v1(result, 14, "diagonal floor is invalid");
      return 1;
   }
   if (request->maximum_preconditioner_block_size <= 0 ||
         (request->preconditioner_kind ==
               AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT &&
          request->maximum_preconditioner_block_size > 256)) {
      set_error_v1(
         result,
         18,
         "maximum preconditioner block size is invalid for this preconditioner");
      return 1;
   }
   if (session->numeric_epoch == UINT64_MAX) {
      set_error_v1(result, 19, "numeric epoch is exhausted");
      return 1;
   }

   /*
    * The old factor is invalid before either device value buffer changes. If
    * any later step fails, numeric_ready remains false and solve is rejected.
    */
   if (check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(before numeric update)")) {
      invalidate_numeric_state(session);
      fill_session_metadata(session, result);
      return 1;
   }
   invalidate_numeric_state(session);
   started = monotonic_seconds();
   if (check_cuda_v1(
         cudaMemcpyAsync(
            session->device_hamiltonian,
            request->hamiltonian_values,
            (size_t)session->nnz * sizeof(float),
            cudaMemcpyHostToDevice,
            session->stream),
         result,
         "cudaMemcpyAsync(H values)") ||
       check_cuda_v1(
         cudaMemcpyAsync(
            session->device_overlap,
            request->overlap_values,
            (size_t)session->nnz * sizeof(float),
            cudaMemcpyHostToDevice,
            session->stream),
         result,
         "cudaMemcpyAsync(S values)") ||
       check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(numeric values)")) {
      goto cleanup;
   }

   if (request->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_DIAGONAL) {
      if (build_inverse_diagonal(
            session,
            request->hamiltonian_values,
            request->overlap_values,
            request->target_shift_hartree,
            request->diagonal_floor_hartree,
            &host_inverse_diagonal,
            result)) {
         goto cleanup;
      }
      if (check_cuda_v1(
            cudaMalloc(
               (void **)&session->device_inverse_diagonal,
               (size_t)session->n * sizeof(float)),
            result,
            "cudaMalloc(inverse diagonal)") ||
          check_cuda_v1(
            cudaMemcpyAsync(
               session->device_inverse_diagonal,
               host_inverse_diagonal,
               (size_t)session->n * sizeof(float),
               cudaMemcpyHostToDevice,
               session->stream),
            result,
            "cudaMemcpyAsync(inverse diagonal)") ||
          check_cuda_v1(
            cudaStreamSynchronize(session->stream),
            result,
            "cudaStreamSynchronize(inverse diagonal)")) {
         goto cleanup;
      }
      session->diagonal_preconditioner.handle = session->cublas_handle;
      session->diagonal_preconditioner.inverse_diagonal =
         session->device_inverse_diagonal;
      session->diagonal_preconditioner.n = session->n;
   }
   else if (request->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT) {
      char factor_error[AOSPECTRUM_PRIMME_CUDA_ERROR_CAPACITY] = {0};
      if (aospectrum_cudss_shift_factor_create_device(
            &session->cudss_preconditioner.factor,
            session->n,
            session->nnz,
            session->host_indptr,
            session->host_indices,
            session->device_hamiltonian,
            session->device_overlap,
            request->target_shift_hartree,
            request->maximum_preconditioner_block_size,
            session->stream,
            factor_error,
            sizeof(factor_error))) {
         set_error_v1(
            result,
            32,
            "cuDSS shift factor failed: %s",
            factor_error);
         goto cleanup;
      }
      session->factor_maximum_rhs =
         request->maximum_preconditioner_block_size;
   }
   result->upload_seconds = monotonic_seconds() - started;

   session->numeric_epoch += 1u;
   session->values_epoch += 1u;
   session->values_ready = 1;
   session->preconditioner_kind = request->preconditioner_kind;
   session->target_shift_hartree = request->target_shift_hartree;
   session->diagonal_floor_hartree = request->diagonal_floor_hartree;
   session->maximum_preconditioner_block_size =
      request->maximum_preconditioner_block_size;
   session->numeric_ready = 1;
   if (request->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT) {
      session->factor_numeric_epoch = session->numeric_epoch;
   }

   result->status = 0;
   result->error_message[0] = '\0';
   fill_session_metadata(session, result);
   record_session_cudss_stats(session, result);
   return_code = 0;

cleanup:
   if (return_code != 0) {
      /*
       * Host update buffers are borrowed only for this call. Drain any copy
       * that was queued before a later CUDA error before returning to caller.
       */
      cudaStreamSynchronize(session->stream);
      invalidate_numeric_state(session);
      fill_session_metadata(session, result);
   }
   free(host_inverse_diagonal);
   return return_code;
}

int aospectrum_primme_cuda_real32_session_load_device_values_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_device_values_request_v1 *request,
      aospectrum_primme_cuda_result_v1 *result) {
   int return_code = 1;
   cudaEvent_t producer_ready = NULL;

   if (result == NULL) {
      return 1;
   }
   initialize_result_v1(result);
   if (session == NULL || request == NULL) {
      set_error_v1(result, 10, "device-value session or request is null");
      return 1;
   }
   fill_session_metadata(session, result);
   if (request->abi_version != AOSPECTRUM_PRIMME_CUDA_ABI_VERSION ||
         request->struct_size != sizeof(*request)) {
      set_error_v1(result, 10, "device-value request ABI mismatch");
      return 1;
   }
   if (!isfinite(request->hamiltonian_scale_to_hartree) ||
         request->hamiltonian_scale_to_hartree <= 0.0f) {
      set_error_v1(result, 14, "Hamiltonian device scale is invalid");
      return 1;
   }
   if (session->values_epoch == UINT64_MAX) {
      set_error_v1(result, 19, "device-value epoch is exhausted");
      return 1;
   }
   if (require_device_pointer_v1(
         request->device_hamiltonian_values,
         session->device_ordinal,
         result,
         "Hamiltonian") ||
       require_device_pointer_v1(
         request->device_overlap_values,
         session->device_ordinal,
         result,
         "overlap")) {
      return 1;
   }
   if (check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(before device admission)")) {
      invalidate_numeric_state(session);
      fill_session_metadata(session, result);
      return 1;
   }
   clear_numeric_commit(session);

   const double started = monotonic_seconds();
   const cudaStream_t producer_stream =
      (cudaStream_t)request->producer_stream;
   if (check_cuda_v1(
         cudaEventCreateWithFlags(&producer_ready, cudaEventDisableTiming),
         result,
         "cudaEventCreate(device admission)") ||
       check_cuda_v1(
         cudaEventRecord(producer_ready, producer_stream),
         result,
         "cudaEventRecord(producer stream)") ||
       check_cuda_v1(
         cudaStreamWaitEvent(session->stream, producer_ready, 0),
         result,
         "cudaStreamWaitEvent(device admission)") ||
       check_cuda_v1(
         cudaMemcpyAsync(
            session->device_hamiltonian,
            request->device_hamiltonian_values,
            (size_t)session->nnz * sizeof(float),
            cudaMemcpyDeviceToDevice,
            session->stream),
         result,
         "cudaMemcpyAsync(H device values)") ||
       check_cuda_v1(
         cudaMemcpyAsync(
            session->device_overlap,
            request->device_overlap_values,
            (size_t)session->nnz * sizeof(float),
            cudaMemcpyDeviceToDevice,
            session->stream),
         result,
         "cudaMemcpyAsync(S device values)") ||
       check_cublas_v1(
         cublasSscal(
            session->cublas_handle,
            (int)session->nnz,
            &request->hamiltonian_scale_to_hartree,
            session->device_hamiltonian,
            1),
         result,
         "cublasSscal(H device values)") ||
       check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(device admission)")) {
      goto cleanup;
   }
   session->values_epoch += 1u;
   session->values_ready = 1;
   result->upload_seconds = monotonic_seconds() - started;
   result->status = 0;
   result->error_message[0] = '\0';
   fill_session_metadata(session, result);
   return_code = 0;

cleanup:
   if (producer_ready != NULL) {
      cudaEventDestroy(producer_ready);
   }
   if (return_code != 0) {
      cudaStreamSynchronize(session->stream);
      invalidate_numeric_state(session);
      fill_session_metadata(session, result);
   }
   return return_code;
}

int aospectrum_primme_cuda_real32_session_factor_resident_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_resident_factor_request_v1 *request,
      aospectrum_primme_cuda_result_v1 *result) {
   int return_code = 1;
   int reuse_analysis = 0;
   char factor_error[AOSPECTRUM_PRIMME_CUDA_ERROR_CAPACITY] = {0};

   if (result == NULL) {
      return 1;
   }
   initialize_result_v1(result);
   if (session == NULL || request == NULL) {
      set_error_v1(result, 10, "resident-factor session or request is null");
      return 1;
   }
   fill_session_metadata(session, result);
   if (request->abi_version != AOSPECTRUM_PRIMME_CUDA_ABI_VERSION ||
         request->struct_size != sizeof(*request)) {
      set_error_v1(result, 10, "resident-factor request ABI mismatch");
      return 1;
   }
   if (!session->values_ready) {
      set_error_v1(result, 18, "resident H/S values are not ready");
      return 1;
   }
   if (!isfinite(request->target_shift_hartree)) {
      set_error_v1(result, 14, "resident factor shift is invalid");
      return 1;
   }
   if (request->preconditioner_kind !=
         AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT ||
       request->maximum_preconditioner_block_size <= 0 ||
       request->maximum_preconditioner_block_size > 256) {
      set_error_v1(
         result,
         18,
         "resident device path requires a valid cuDSS block size");
      return 1;
   }
   if (session->numeric_epoch == UINT64_MAX) {
      set_error_v1(result, 19, "numeric epoch is exhausted");
      return 1;
   }

   reuse_analysis =
      session->cudss_preconditioner.factor != NULL &&
      session->factor_maximum_rhs ==
         request->maximum_preconditioner_block_size;
   clear_numeric_commit(session);
   if (!reuse_analysis && session->cudss_preconditioner.factor != NULL) {
      aospectrum_cudss_shift_factor_destroy(
         session->cudss_preconditioner.factor);
      session->cudss_preconditioner.factor = NULL;
      session->factor_maximum_rhs = 0;
   }

   if (reuse_analysis) {
      if (aospectrum_cudss_shift_factor_refactor_device(
            session->cudss_preconditioner.factor,
            session->device_hamiltonian,
            session->device_overlap,
            request->target_shift_hartree,
            factor_error,
            sizeof(factor_error))) {
         set_error_v1(
            result,
            32,
            "cuDSS shift refactor failed: %s",
            factor_error);
         goto cleanup;
      }
   }
   else if (aospectrum_cudss_shift_factor_create_device(
         &session->cudss_preconditioner.factor,
         session->n,
         session->nnz,
         session->host_indptr,
         session->host_indices,
         session->device_hamiltonian,
         session->device_overlap,
         request->target_shift_hartree,
         request->maximum_preconditioner_block_size,
         session->stream,
         factor_error,
         sizeof(factor_error))) {
      set_error_v1(
         result,
         32,
         "cuDSS shift factor failed: %s",
         factor_error);
      goto cleanup;
   }

   session->factor_maximum_rhs =
      request->maximum_preconditioner_block_size;
   session->numeric_epoch += 1u;
   session->preconditioner_kind = request->preconditioner_kind;
   session->target_shift_hartree = request->target_shift_hartree;
   session->diagonal_floor_hartree = request->diagonal_floor_hartree;
   session->maximum_preconditioner_block_size =
      request->maximum_preconditioner_block_size;
   session->numeric_ready = 1;
   session->factor_numeric_epoch = session->numeric_epoch;
   result->status = 0;
   result->error_message[0] = '\0';
   fill_session_metadata(session, result);
   record_session_cudss_stats(session, result);
   return_code = 0;

cleanup:
   if (return_code != 0) {
      clear_numeric_commit(session);
      if (session->cudss_preconditioner.factor != NULL) {
         aospectrum_cudss_shift_factor_destroy(
            session->cudss_preconditioner.factor);
         session->cudss_preconditioner.factor = NULL;
      }
      session->factor_maximum_rhs = 0;
      fill_session_metadata(session, result);
   }
   return return_code;
}

int aospectrum_primme_cuda_real32_session_apply_operator_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_operator_request_v1 *request,
      const float *input_vectors,
      float *output_vectors,
      aospectrum_primme_cuda_result_v1 *result) {
   int return_code = 1;
   int matvec_error = 0;
   float *device_input = NULL;
   float *device_output = NULL;
   sparse_operator *operator_value = NULL;
   size_t vector_bytes;
   double started;

   if (result == NULL) {
      return 1;
   }
   initialize_result_v1(result);
   if (session == NULL || request == NULL) {
      set_error_v1(result, 10, "operator session or request is null");
      return 1;
   }
   fill_session_metadata(session, result);
   if (request->abi_version != AOSPECTRUM_PRIMME_CUDA_ABI_VERSION ||
         request->struct_size != sizeof(*request)) {
      set_error_v1(result, 10, "operator request ABI mismatch");
      return 1;
   }
   if (!session->numeric_ready ||
         request->numeric_epoch != session->numeric_epoch) {
      set_error_v1(result, 19, "operator request numeric epoch is stale");
      return 1;
   }
   if (request->operator_kind == AOSPECTRUM_PRIMME_OPERATOR_HAMILTONIAN) {
      operator_value = &session->hamiltonian;
   }
   else if (request->operator_kind == AOSPECTRUM_PRIMME_OPERATOR_OVERLAP) {
      operator_value = &session->overlap;
   }
   else {
      set_error_v1(result, 17, "operator kind is invalid");
      return 1;
   }
   if (request->block_size <= 0 ||
         (size_t)session->n >
            SIZE_MAX / ((size_t)request->block_size * sizeof(float))) {
      set_error_v1(result, 18, "operator block size is invalid");
      return 1;
   }
   if (input_vectors == NULL || output_vectors == NULL) {
      set_error_v1(result, 12, "operator vector pointer is null");
      return 1;
   }
   vector_bytes =
      (size_t)session->n * (size_t)request->block_size * sizeof(float);

   if (check_cuda_v1(
         cudaMalloc((void **)&device_input, vector_bytes),
         result,
         "cudaMalloc(operator input)") ||
       check_cuda_v1(
         cudaMalloc((void **)&device_output, vector_bytes),
         result,
         "cudaMalloc(operator output)")) {
      goto cleanup;
   }

   started = monotonic_seconds();
   if (check_cuda_v1(
         cudaMemcpyAsync(
            device_input,
            input_vectors,
            vector_bytes,
            cudaMemcpyHostToDevice,
            session->stream),
         result,
         "cudaMemcpyAsync(operator input)") ||
       check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(operator input)")) {
      goto cleanup;
   }
   result->upload_seconds = monotonic_seconds() - started;

   operator_value->last_error = 0;
   started = monotonic_seconds();
   sparse_matvec(
      operator_value,
      device_input,
      session->n,
      device_output,
      session->n,
      request->block_size,
      &matvec_error);
   if (matvec_error) {
      set_error_v1(
         result,
         22,
         "operator sparse_matvec failed with cuSPARSE status %d",
         operator_value->last_error);
      goto cleanup;
   }
   if (check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(operator matvec)")) {
      goto cleanup;
   }
   result->solve_seconds = monotonic_seconds() - started;

   started = monotonic_seconds();
   if (check_cuda_v1(
         cudaMemcpyAsync(
            output_vectors,
            device_output,
            vector_bytes,
            cudaMemcpyDeviceToHost,
            session->stream),
         result,
         "cudaMemcpyAsync(operator output)") ||
       check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(operator output)")) {
      goto cleanup;
   }
   result->download_seconds = monotonic_seconds() - started;
   result->static_device_bytes =
      session_static_device_bytes(session) + (uint64_t)vector_bytes * 2u;
   result->status = 0;
   result->error_message[0] = '\0';
   fill_session_metadata(session, result);
   return_code = 0;

cleanup:
   if (return_code != 0) {
      cudaStreamSynchronize(session->stream);
   }
   if (device_output != NULL) {
      cudaFree(device_output);
   }
   if (device_input != NULL) {
      cudaFree(device_input);
   }
   return return_code;
}

static primme_target target_mode_to_primme(int32_t target_mode) {
   if (target_mode == AOSPECTRUM_PRIMME_TARGET_CLOSEST_LEQ) {
      return primme_closest_leq;
   }
   if (target_mode == AOSPECTRUM_PRIMME_TARGET_CLOSEST_GEQ) {
      return primme_closest_geq;
   }
   return primme_closest_abs;
}

static int validate_solve_request(
      const aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_solve_request_v1 *request,
      const float *eigenvalues,
      const float *eigenvectors,
      const float *residuals,
      aospectrum_primme_cuda_result_v1 *result) {
   if (request->abi_version != AOSPECTRUM_PRIMME_CUDA_ABI_VERSION ||
         request->struct_size != sizeof(*request)) {
      set_error_v1(result, 10, "solve request ABI mismatch");
      return 1;
   }
   if (!session->numeric_ready || session->numeric_epoch == 0) {
      set_error_v1(result, 33, "session has no committed numeric epoch");
      return 1;
   }
   if (request->numeric_epoch != session->numeric_epoch) {
      set_error_v1(
         result,
         34,
         "solve numeric epoch mismatch: requested %llu, current %llu",
         (unsigned long long)request->numeric_epoch,
         (unsigned long long)session->numeric_epoch);
      return 1;
   }
   if (!isfinite(request->target_shift_hartree) ||
         request->target_shift_hartree != session->target_shift_hartree) {
      set_error_v1(
         result,
         35,
         "solve target shift does not match the committed numeric factor");
      return 1;
   }
   if (request->target_mode < AOSPECTRUM_PRIMME_TARGET_CLOSEST_ABS ||
         request->target_mode > AOSPECTRUM_PRIMME_TARGET_CLOSEST_GEQ) {
      set_error_v1(result, 17, "target mode is invalid");
      return 1;
   }
   if (eigenvalues == NULL || residuals == NULL) {
      set_error_v1(result, 12, "required solve output pointer is null");
      return 1;
   }
   if (request->num_evals <= 0 || request->num_evals >= session->n ||
         request->max_matvecs <= 0 ||
         request->max_basis_size <= request->num_evals ||
         request->min_restart_size <= 0 ||
         request->min_restart_size >= request->max_basis_size ||
         request->max_block_size <= 0 ||
         request->max_block_size > request->num_evals) {
      set_error_v1(result, 13, "invalid PRIMME dimensions or iteration limits");
      return 1;
   }
   if (!isfinite(request->tolerance) || request->tolerance <= 0.0f) {
      set_error_v1(result, 14, "solve tolerance is invalid");
      return 1;
   }
   if (request->init_size < 0 || request->init_size > request->num_evals ||
         (request->init_size > 0 && request->initial_vectors == NULL)) {
      set_error_v1(result, 15, "initial-vector window is invalid");
      return 1;
   }
   if (request->return_vectors != 0 && request->return_vectors != 1) {
      set_error_v1(result, 16, "return_vectors must be zero or one");
      return 1;
   }
   if (request->return_vectors && eigenvectors == NULL) {
      set_error_v1(result, 44, "return_vectors requested with null output pointer");
      return 1;
   }
   if (session->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT) {
      if (session->cudss_preconditioner.factor == NULL ||
            session->factor_numeric_epoch != session->numeric_epoch) {
         set_error_v1(result, 36, "cuDSS factor is absent or stale");
         return 1;
      }
      if (request->max_block_size >
            session->maximum_preconditioner_block_size) {
         set_error_v1(
            result,
            37,
            "solve block size exceeds the committed cuDSS factor capacity");
         return 1;
      }
   }
   if (session->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_DIAGONAL &&
       session->device_inverse_diagonal == NULL) {
      set_error_v1(result, 38, "diagonal preconditioner is absent");
      return 1;
   }
   return 0;
}

int aospectrum_primme_cuda_real32_session_solve_v1(
      aospectrum_primme_cuda_real32_session_v1 *session,
      const aospectrum_primme_cuda_solve_request_v1 *request,
      float *eigenvalues_hartree,
      float *eigenvectors,
      float *residual_norms,
      aospectrum_primme_cuda_result_v1 *result) {
   int return_code = 1;
   int primme_initialized = 0;
   float *device_eigenvectors = NULL;
   size_t free_bytes = 0;
   size_t total_bytes = 0;
   size_t eigenvector_bytes;
   double target_shift;
   double started;
   primme_params primme;

   if (result == NULL) {
      return 1;
   }
   initialize_result_v1(result);
   if (session == NULL || request == NULL) {
      set_error_v1(result, 10, "solve session or request is null");
      return 1;
   }
   fill_session_metadata(session, result);
   if (validate_solve_request(
         session,
         request,
         eigenvalues_hartree,
         eigenvectors,
         residual_norms,
         result)) {
      return 1;
   }
   if ((uint64_t)session->n >
         (uint64_t)SIZE_MAX /
            ((uint64_t)request->num_evals * sizeof(float))) {
      set_error_v1(result, 39, "eigenvector workspace size overflows size_t");
      return 1;
   }
   eigenvector_bytes =
      (size_t)session->n * (size_t)request->num_evals * sizeof(float);
   result->target_mode = request->target_mode;
   result->factor_reused =
      session->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT &&
      session->solves_in_numeric_epoch > 0;

   started = monotonic_seconds();
   if (check_cuda_v1(
         cudaMalloc((void **)&device_eigenvectors, eigenvector_bytes),
         result,
         "cudaMalloc(eigenvectors)")) {
      goto cleanup;
   }
   if (request->init_size > 0) {
      if (check_cuda_v1(
            cudaMemcpyAsync(
               device_eigenvectors,
               request->initial_vectors,
               (size_t)session->n *
                  (size_t)request->init_size *
                  sizeof(float),
               cudaMemcpyHostToDevice,
               session->stream),
            result,
            "cudaMemcpyAsync(initial vectors)")) {
         goto cleanup;
      }
   }
   if (check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(solve upload)")) {
      goto cleanup;
   }
   result->upload_seconds = monotonic_seconds() - started;

   session->hamiltonian.last_error = 0;
   session->overlap.last_error = 0;
   session->diagonal_preconditioner.last_error = 0;
   session->cudss_preconditioner.last_error = 0;
   session->cudss_preconditioner.error_message[0] = '\0';

   primme_initialize(&primme);
   primme_initialized = 1;

   /*
    * Give the preset enough dimensions to choose coherent defaults, then
    * reassert every user/session field that the preset is allowed to rewrite.
    */
   primme.n = session->n;
   primme.numEvals = request->num_evals;
   primme.maxBasisSize = request->max_basis_size;
   primme.minRestartSize = request->min_restart_size;
   primme.maxBlockSize = request->max_block_size;
   primme.correctionParams.precondition =
      session->preconditioner_kind != AOSPECTRUM_PRIMME_PRECONDITIONER_NONE;
   if (primme_set_method(PRIMME_JDQMR_ETol, &primme) != 0) {
      set_error_v1(result, 40, "primme_set_method(PRIMME_JDQMR_ETol) failed");
      goto cleanup;
   }

   primme.n = session->n;
   primme.numEvals = request->num_evals;
   primme.eps = request->tolerance;
   primme.maxMatvecs = request->max_matvecs;
   primme.maxBasisSize = request->max_basis_size;
   primme.minRestartSize = request->min_restart_size;
   primme.maxBlockSize = request->max_block_size;
   primme.initSize = request->init_size;
   primme.target = target_mode_to_primme(request->target_mode);
   /* PRIMME rejects closest_leq/closest_geq when locking is disabled. */
   if (request->target_mode != AOSPECTRUM_PRIMME_TARGET_CLOSEST_ABS) {
      primme.locking = 1;
   }
   target_shift = request->target_shift_hartree;
   primme.numTargetShifts = 1;
   primme.targetShifts = &target_shift;
   primme.matrixMatvec = hamiltonian_matvec;
   primme.matrixMatvec_type = primme_op_float;
   primme.matrix = &session->hamiltonian;
   primme.massMatrixMatvec = overlap_matvec;
   primme.massMatrixMatvec_type = primme_op_float;
   primme.massMatrix = &session->overlap;
   primme.internalPrecision = primme_op_float;
   primme.queue = &session->cublas_handle;
   primme.printLevel = request->print_level;
   primme.applyPreconditioner = NULL;
   primme.preconditioner = NULL;
   primme.correctionParams.precondition = 0;
   if (session->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_DIAGONAL) {
      primme.applyPreconditioner = apply_diagonal_preconditioner;
      primme.applyPreconditioner_type = primme_op_float;
      primme.preconditioner = &session->diagonal_preconditioner;
      primme.correctionParams.precondition = 1;
   }
   else if (session->preconditioner_kind ==
         AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT) {
      primme.applyPreconditioner = apply_cudss_shift_preconditioner;
      primme.applyPreconditioner_type = primme_op_float;
      primme.preconditioner = &session->cudss_preconditioner;
      primme.correctionParams.precondition = 1;
   }

   if (check_cuda_v1(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         result,
         "cudaMemGetInfo(before solve)")) {
      goto cleanup;
   }
   result->free_device_bytes_before_solve = free_bytes;
   result->total_device_bytes = total_bytes;
   result->static_device_bytes =
      session_static_device_bytes(session) + (uint64_t)eigenvector_bytes;

   session->solve_index += 1u;
   session->solves_in_numeric_epoch += 1u;
   result->solve_index = session->solve_index;
   started = monotonic_seconds();
   result->primme_status = cublas_sprimme(
      eigenvalues_hartree,
      device_eigenvectors,
      residual_norms,
      &primme);
   if (check_cuda_v1(
         cudaStreamSynchronize(session->stream),
         result,
         "cudaStreamSynchronize(solve)")) {
      goto cleanup;
   }
   result->solve_seconds = monotonic_seconds() - started;
   result->converged = primme.initSize;
   result->outer_iterations = primme.stats.numOuterIterations;
   result->restarts = primme.stats.numRestarts;
   result->matvecs = primme.stats.numMatvecs;
   result->preconditions = primme.stats.numPreconds;
   result->primme_matvec_seconds = primme.stats.timeMatvec;
   result->primme_precondition_seconds = primme.stats.timePrecond;
   result->primme_orthogonalization_seconds = primme.stats.timeOrtho;
   record_session_cudss_stats(session, result);

   if (session->hamiltonian.last_error ||
         session->overlap.last_error ||
         session->diagonal_preconditioner.last_error ||
         session->cudss_preconditioner.last_error) {
      set_error_v1(
         result,
         41,
         "GPU callback failed: H=%d S=%d diagonal=%d cuDSS=%d %s",
         session->hamiltonian.last_error,
         session->overlap.last_error,
         session->diagonal_preconditioner.last_error,
         session->cudss_preconditioner.last_error,
         session->cudss_preconditioner.error_message);
      goto cleanup;
   }
   if (result->primme_status != 0) {
      set_error_v1(
         result,
         42,
         "cublas_sprimme returned %d after converging %d of %d eigenpairs",
         result->primme_status,
         result->converged,
         request->num_evals);
      goto cleanup;
   }
   if (result->converged != request->num_evals) {
      set_error_v1(
         result,
         43,
         "PRIMME converged %d of %d requested eigenpairs",
         result->converged,
         request->num_evals);
      goto cleanup;
   }

   started = monotonic_seconds();
   if (request->return_vectors) {
      if (check_cuda_v1(
            cudaMemcpyAsync(
               eigenvectors,
               device_eigenvectors,
               eigenvector_bytes,
               cudaMemcpyDeviceToHost,
               session->stream),
            result,
            "cudaMemcpyAsync(eigenvectors)") ||
          check_cuda_v1(
            cudaStreamSynchronize(session->stream),
            result,
            "cudaStreamSynchronize(download)")) {
         goto cleanup;
      }
   }
   if (check_cuda_v1(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         result,
         "cudaMemGetInfo(after solve)")) {
      goto cleanup;
   }
   result->free_device_bytes_after_solve = free_bytes;
   result->download_seconds = monotonic_seconds() - started;
   result->status = 0;
   result->error_message[0] = '\0';
   return_code = 0;

cleanup:
   record_session_cudss_stats(session, result);
   fill_session_metadata(session, result);
   /*
    * PRIMME and the upload/download paths borrow caller buffers only for this
    * call. Best-effort synchronization keeps that lifetime true on errors too.
    */
   cudaStreamSynchronize(session->stream);
   if (primme_initialized) {
      primme_free(&primme);
   }
   if (device_eigenvectors != NULL) {
      cudaFree(device_eigenvectors);
   }
   return return_code;
}
