#define _POSIX_C_SOURCE 200809L

#include "aospectrum_cudss_shift_factor.h"

#include <cudss.h>

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct aospectrum_cudss_shift_factor {
   int64_t n;
   int32_t maximum_rhs;
   cudaStream_t stream;
   int32_t *device_indptr;
   int32_t *device_indices;
   float *device_values;
   cudssHandle_t handle;
   cudssConfig_t config;
   cudssData_t data;
   cudssMatrix_t matrix;
   aospectrum_cudss_shift_factor_stats stats;
};

static double monotonic_seconds(void) {
   struct timespec value;
   if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
      return 0.0;
   }
   return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static void set_error(
      char *message,
      size_t capacity,
      const char *format,
      ...) {
   if (message == NULL || capacity == 0) {
      return;
   }
   va_list arguments;
   va_start(arguments, format);
   vsnprintf(message, capacity, format, arguments);
   va_end(arguments);
}

static int check_cuda(
      cudaError_t status,
      char *message,
      size_t capacity,
      const char *operation) {
   if (status == cudaSuccess) {
      return 0;
   }
   set_error(
      message,
      capacity,
      "%s failed: %s",
      operation,
      cudaGetErrorString(status));
   return 1;
}

static int check_cudss(
      cudssStatus_t status,
      char *message,
      size_t capacity,
      const char *operation) {
   if (status == CUDSS_STATUS_SUCCESS) {
      return 0;
   }
   set_error(
      message,
      capacity,
      "%s failed with cuDSS status %d",
      operation,
      (int)status);
   return 1;
}

static int prepare_upper_triangle(
      int64_t n,
      int64_t nnz,
      const int32_t *indptr,
      const int32_t *indices,
      const float *hamiltonian_values,
      const float *overlap_values,
      float shift_hartree,
      int32_t **upper_indptr,
      int32_t **upper_indices,
      float **upper_values,
      int64_t *upper_nnz,
      char *error_message,
      size_t error_capacity) {
   int64_t count = 0;
   int64_t diagonals = 0;
   int32_t *output_indptr = calloc((size_t)n + 1, sizeof(*output_indptr));
   if (output_indptr == NULL) {
      set_error(
         error_message,
         error_capacity,
         "cannot allocate upper-triangle row offsets");
      return 1;
   }

   for (int64_t row = 0; row < n; ++row) {
      const int32_t begin = indptr[row];
      const int32_t end = indptr[row + 1];
      int has_diagonal = 0;
      if (begin < 0 || end < begin || end > nnz) {
         free(output_indptr);
         set_error(
            error_message,
            error_capacity,
            "CSR row offsets are not monotonic");
         return 1;
      }
      for (int32_t position = begin; position < end; ++position) {
         const int32_t column = indices[position];
         if (column < 0 || column >= n) {
            free(output_indptr);
            set_error(
               error_message,
               error_capacity,
               "CSR column index is outside the matrix");
            return 1;
         }
         if (column >= row) {
            ++count;
         }
         if (column == row) {
            has_diagonal = 1;
         }
      }
      if (count > INT32_MAX) {
         free(output_indptr);
         set_error(
            error_message,
            error_capacity,
            "upper-triangle nnz exceeds int32");
         return 1;
      }
      diagonals += has_diagonal;
      output_indptr[row + 1] = (int32_t)count;
   }
   if (diagonals != n) {
      free(output_indptr);
      set_error(
         error_message,
         error_capacity,
         "shifted matrix is missing diagonal entries: %lld/%lld",
         (long long)diagonals,
         (long long)n);
      return 1;
   }

   int32_t *output_indices = malloc((size_t)count * sizeof(*output_indices));
   float *output_values = malloc((size_t)count * sizeof(*output_values));
   if (output_indices == NULL || output_values == NULL) {
      free(output_values);
      free(output_indices);
      free(output_indptr);
      set_error(
         error_message,
         error_capacity,
         "cannot allocate upper-triangle CSR values");
      return 1;
   }

   int64_t output = 0;
   for (int64_t row = 0; row < n; ++row) {
      for (int32_t position = indptr[row];
            position < indptr[row + 1];
            ++position) {
         const int32_t column = indices[position];
         if (column < row) {
            continue;
         }
         const float value =
            hamiltonian_values[position] -
            shift_hartree * overlap_values[position];
         if (!isfinite(value)) {
            free(output_values);
            free(output_indices);
            free(output_indptr);
            set_error(
               error_message,
               error_capacity,
               "shifted matrix contains non-finite values");
            return 1;
         }
         output_indices[output] = column;
         output_values[output] = value;
         ++output;
      }
   }

   *upper_indptr = output_indptr;
   *upper_indices = output_indices;
   *upper_values = output_values;
   *upper_nnz = count;
   return 0;
}

static int query_data_info(
      aospectrum_cudss_shift_factor *factor,
      char *error_message,
      size_t error_capacity,
      const char *phase) {
   int info = 0;
   size_t bytes_written = 0;
   if (check_cudss(
         cudssDataGet(
            factor->handle,
            factor->data,
            CUDSS_DATA_INFO,
            &info,
            sizeof(info),
            &bytes_written),
         error_message,
         error_capacity,
         "cuDSS device-info query")) {
      return 1;
   }
   if (bytes_written != sizeof(info) || info != 0) {
      set_error(
         error_message,
         error_capacity,
         "%s reported cuDSS device info %d",
         phase,
         info);
      return 1;
   }
   return 0;
}

int aospectrum_cudss_shift_factor_create(
      aospectrum_cudss_shift_factor **factor_output,
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
      size_t error_capacity) {
   int return_code = 1;
   int32_t *host_indptr = NULL;
   int32_t *host_indices = NULL;
   float *host_values = NULL;
   float *dummy_rhs_values = NULL;
   float *dummy_solution_values = NULL;
   cudssMatrix_t dummy_rhs = NULL;
   cudssMatrix_t dummy_solution = NULL;
   aospectrum_cudss_shift_factor *factor = NULL;

   if (factor_output == NULL) {
      set_error(error_message, error_capacity, "factor output pointer is null");
      return 1;
   }
   *factor_output = NULL;
   if (n <= 0 || n > INT32_MAX || nnz <= 0 || nnz > INT32_MAX ||
         indptr == NULL || indices == NULL ||
         hamiltonian_values == NULL || overlap_values == NULL ||
         !isfinite(shift_hartree) ||
         maximum_rhs <= 0 || maximum_rhs > 256) {
      set_error(error_message, error_capacity, "invalid shift-factor request");
      return 1;
   }
   if (indptr[0] != 0 || indptr[n] != nnz) {
      set_error(
         error_message,
         error_capacity,
         "CSR row offsets do not span the declared nnz");
      return 1;
   }

   factor = calloc(1, sizeof(*factor));
   if (factor == NULL) {
      set_error(error_message, error_capacity, "cannot allocate factor state");
      return 1;
   }
   factor->n = n;
   factor->maximum_rhs = maximum_rhs;
   factor->stream = stream;

   double started = monotonic_seconds();
   if (prepare_upper_triangle(
         n,
         nnz,
         indptr,
         indices,
         hamiltonian_values,
         overlap_values,
         shift_hartree,
         &host_indptr,
         &host_indices,
         &host_values,
         &factor->stats.upper_nnz,
         error_message,
         error_capacity)) {
      goto cleanup;
   }
   factor->stats.preparation_seconds = monotonic_seconds() - started;

   size_t free_bytes = 0;
   size_t total_bytes = 0;
   if (check_cuda(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         error_message,
         error_capacity,
         "cudaMemGetInfo before factorization")) {
      goto cleanup;
   }
   factor->stats.free_device_bytes_before_factorization = free_bytes;

   started = monotonic_seconds();
   if (check_cuda(
         cudaMalloc(
            (void **)&factor->device_indptr,
            ((size_t)n + 1) * sizeof(*factor->device_indptr)),
         error_message,
         error_capacity,
         "shifted row-offset allocation") ||
       check_cuda(
         cudaMalloc(
            (void **)&factor->device_indices,
            (size_t)factor->stats.upper_nnz *
               sizeof(*factor->device_indices)),
         error_message,
         error_capacity,
         "shifted column allocation") ||
       check_cuda(
         cudaMalloc(
            (void **)&factor->device_values,
            (size_t)factor->stats.upper_nnz *
               sizeof(*factor->device_values)),
         error_message,
         error_capacity,
         "shifted value allocation") ||
       check_cuda(
         cudaMalloc(
            (void **)&dummy_rhs_values,
            (size_t)n * (size_t)maximum_rhs * sizeof(*dummy_rhs_values)),
         error_message,
         error_capacity,
         "dummy rhs allocation") ||
       check_cuda(
         cudaMalloc(
            (void **)&dummy_solution_values,
            (size_t)n * (size_t)maximum_rhs *
               sizeof(*dummy_solution_values)),
         error_message,
         error_capacity,
         "dummy solution allocation") ||
       check_cuda(
         cudaMemcpyAsync(
            factor->device_indptr,
            host_indptr,
            ((size_t)n + 1) * sizeof(*factor->device_indptr),
            cudaMemcpyHostToDevice,
            stream),
         error_message,
         error_capacity,
         "shifted row-offset upload") ||
       check_cuda(
         cudaMemcpyAsync(
            factor->device_indices,
            host_indices,
            (size_t)factor->stats.upper_nnz *
               sizeof(*factor->device_indices),
            cudaMemcpyHostToDevice,
            stream),
         error_message,
         error_capacity,
         "shifted column upload") ||
       check_cuda(
         cudaMemcpyAsync(
            factor->device_values,
            host_values,
            (size_t)factor->stats.upper_nnz *
               sizeof(*factor->device_values),
            cudaMemcpyHostToDevice,
            stream),
         error_message,
         error_capacity,
         "shifted value upload") ||
       check_cuda(
         cudaStreamSynchronize(stream),
         error_message,
         error_capacity,
         "shifted matrix upload synchronization")) {
      goto cleanup;
   }
   factor->stats.upload_seconds = monotonic_seconds() - started;

   if (check_cudss(
         cudssCreate(&factor->handle),
         error_message,
         error_capacity,
         "cuDSS handle creation") ||
       check_cudss(
         cudssSetStream(factor->handle, stream),
         error_message,
         error_capacity,
         "cuDSS stream binding") ||
       check_cudss(
         cudssConfigCreate(&factor->config),
         error_message,
         error_capacity,
         "cuDSS config creation") ||
       check_cudss(
         cudssDataCreate(factor->handle, &factor->data),
         error_message,
         error_capacity,
         "cuDSS data creation") ||
       check_cudss(
         cudssMatrixCreateCsr(
            &factor->matrix,
            n,
            n,
            factor->stats.upper_nnz,
            factor->device_indptr,
            NULL,
            factor->device_indices,
            factor->device_values,
            CUDSS_R_32I,
            CUDSS_R_32I,
            CUDSS_R_32F,
            CUDSS_MTYPE_SYMMETRIC,
            CUDSS_MVIEW_UPPER,
            CUDSS_BASE_ZERO),
         error_message,
         error_capacity,
         "cuDSS shifted matrix creation") ||
       check_cudss(
         cudssMatrixCreateDn(
            &dummy_rhs,
            n,
            maximum_rhs,
            n,
            dummy_rhs_values,
            CUDSS_R_32F,
            CUDSS_LAYOUT_COL_MAJOR),
         error_message,
         error_capacity,
         "cuDSS dummy rhs creation") ||
       check_cudss(
         cudssMatrixCreateDn(
            &dummy_solution,
            n,
            maximum_rhs,
            n,
            dummy_solution_values,
            CUDSS_R_32F,
            CUDSS_LAYOUT_COL_MAJOR),
         error_message,
         error_capacity,
         "cuDSS dummy solution creation")) {
      goto cleanup;
   }

   started = monotonic_seconds();
   if (check_cudss(
         cudssExecute(
            factor->handle,
            CUDSS_PHASE_ANALYSIS,
            factor->config,
            factor->data,
            factor->matrix,
            dummy_solution,
            dummy_rhs),
         error_message,
         error_capacity,
         "cuDSS symbolic analysis") ||
       check_cuda(
         cudaStreamSynchronize(stream),
         error_message,
         error_capacity,
         "cuDSS analysis synchronization") ||
       query_data_info(
         factor,
         error_message,
         error_capacity,
         "cuDSS analysis")) {
      goto cleanup;
   }
   factor->stats.analysis_seconds = monotonic_seconds() - started;

   int64_t memory_estimates[16] = {0};
   size_t bytes_written = 0;
   if (check_cudss(
         cudssDataGet(
            factor->handle,
            factor->data,
            CUDSS_DATA_MEMORY_ESTIMATES,
            memory_estimates,
            sizeof(memory_estimates),
            &bytes_written),
         error_message,
         error_capacity,
         "cuDSS memory-estimate query") ||
       bytes_written < 4 * sizeof(memory_estimates[0])) {
      if (error_message != NULL && error_message[0] == '\0') {
         set_error(
            error_message,
            error_capacity,
            "cuDSS returned an incomplete memory estimate");
      }
      goto cleanup;
   }
   factor->stats.permanent_device_bytes =
      memory_estimates[0] > 0 ? (uint64_t)memory_estimates[0] : 0;
   factor->stats.peak_device_bytes =
      memory_estimates[1] > 0 ? (uint64_t)memory_estimates[1] : 0;
   factor->stats.permanent_host_bytes =
      memory_estimates[2] > 0 ? (uint64_t)memory_estimates[2] : 0;
   factor->stats.peak_host_bytes =
      memory_estimates[3] > 0 ? (uint64_t)memory_estimates[3] : 0;

   started = monotonic_seconds();
   if (check_cudss(
         cudssExecute(
            factor->handle,
            CUDSS_PHASE_FACTORIZATION,
            factor->config,
            factor->data,
            factor->matrix,
            dummy_solution,
            dummy_rhs),
         error_message,
         error_capacity,
         "cuDSS numerical factorization") ||
       check_cuda(
         cudaStreamSynchronize(stream),
         error_message,
         error_capacity,
         "cuDSS factorization synchronization") ||
       query_data_info(
         factor,
         error_message,
         error_capacity,
         "cuDSS factorization")) {
      goto cleanup;
   }
   factor->stats.factorization_seconds = monotonic_seconds() - started;

   int64_t factor_nnz = 0;
   int32_t pivot_count = 0;
   int32_t inertia[2] = {0, 0};
   if (check_cudss(
         cudssDataGet(
            factor->handle,
            factor->data,
            CUDSS_DATA_LU_NNZ,
            &factor_nnz,
            sizeof(factor_nnz),
            &bytes_written),
         error_message,
         error_capacity,
         "cuDSS factor-nnz query") ||
       check_cudss(
         cudssDataGet(
            factor->handle,
            factor->data,
            CUDSS_DATA_NPIVOTS,
            &pivot_count,
            sizeof(pivot_count),
            &bytes_written),
         error_message,
         error_capacity,
         "cuDSS pivot-count query") ||
       check_cudss(
         cudssDataGet(
            factor->handle,
            factor->data,
            CUDSS_DATA_INERTIA,
            inertia,
            sizeof(inertia),
            &bytes_written),
         error_message,
         error_capacity,
         "cuDSS inertia query")) {
      goto cleanup;
   }
   factor->stats.factor_nnz = factor_nnz;
   factor->stats.pivot_count = pivot_count;
   factor->stats.inertia_positive = inertia[0];
   factor->stats.inertia_negative = inertia[1];

   if (check_cuda(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         error_message,
         error_capacity,
         "cudaMemGetInfo after factorization")) {
      goto cleanup;
   }
   factor->stats.free_device_bytes_after_factorization = free_bytes;
   *factor_output = factor;
   factor = NULL;
   return_code = 0;

cleanup:
   if (dummy_solution != NULL) {
      cudssMatrixDestroy(dummy_solution);
   }
   if (dummy_rhs != NULL) {
      cudssMatrixDestroy(dummy_rhs);
   }
   cudaFree(dummy_solution_values);
   cudaFree(dummy_rhs_values);
   free(host_values);
   free(host_indices);
   free(host_indptr);
   if (factor != NULL) {
      aospectrum_cudss_shift_factor_destroy(factor);
   }
   return return_code;
}

int aospectrum_cudss_shift_factor_solve(
      aospectrum_cudss_shift_factor *factor,
      const float *rhs,
      int64_t leading_rhs,
      float *solution,
      int64_t leading_solution,
      int32_t rhs_count,
      char *error_message,
      size_t error_capacity) {
   cudssMatrix_t rhs_matrix = NULL;
   cudssMatrix_t solution_matrix = NULL;
   int return_code = 1;

   if (factor == NULL || rhs == NULL || solution == NULL ||
         rhs_count <= 0 || rhs_count > factor->maximum_rhs ||
         leading_rhs < factor->n || leading_solution < factor->n) {
      set_error(error_message, error_capacity, "invalid cuDSS solve request");
      return 1;
   }

   if (check_cudss(
         cudssMatrixCreateDn(
            &rhs_matrix,
            factor->n,
            rhs_count,
            leading_rhs,
            (void *)rhs,
            CUDSS_R_32F,
            CUDSS_LAYOUT_COL_MAJOR),
         error_message,
         error_capacity,
         "cuDSS rhs descriptor creation") ||
       check_cudss(
         cudssMatrixCreateDn(
            &solution_matrix,
            factor->n,
            rhs_count,
            leading_solution,
            solution,
            CUDSS_R_32F,
            CUDSS_LAYOUT_COL_MAJOR),
         error_message,
         error_capacity,
         "cuDSS solution descriptor creation")) {
      goto cleanup;
   }

   double started = monotonic_seconds();
   if (check_cudss(
         cudssExecute(
            factor->handle,
            CUDSS_PHASE_SOLVE,
            factor->config,
            factor->data,
            factor->matrix,
            solution_matrix,
            rhs_matrix),
         error_message,
         error_capacity,
         "cuDSS shifted solve") ||
       check_cuda(
         cudaStreamSynchronize(factor->stream),
         error_message,
         error_capacity,
         "cuDSS shifted-solve synchronization") ||
       query_data_info(
         factor,
         error_message,
         error_capacity,
         "cuDSS shifted solve")) {
      goto cleanup;
   }
   factor->stats.solve_seconds += monotonic_seconds() - started;
   ++factor->stats.solve_calls;
   factor->stats.solved_rhs += rhs_count;
   return_code = 0;

cleanup:
   if (solution_matrix != NULL) {
      cudssMatrixDestroy(solution_matrix);
   }
   if (rhs_matrix != NULL) {
      cudssMatrixDestroy(rhs_matrix);
   }
   return return_code;
}

void aospectrum_cudss_shift_factor_get_stats(
      const aospectrum_cudss_shift_factor *factor,
      aospectrum_cudss_shift_factor_stats *stats) {
   if (stats == NULL) {
      return;
   }
   if (factor == NULL) {
      memset(stats, 0, sizeof(*stats));
      return;
   }
   *stats = factor->stats;
}

void aospectrum_cudss_shift_factor_destroy(
      aospectrum_cudss_shift_factor *factor) {
   if (factor == NULL) {
      return;
   }
   if (factor->matrix != NULL) {
      cudssMatrixDestroy(factor->matrix);
   }
   if (factor->data != NULL && factor->handle != NULL) {
      cudssDataDestroy(factor->handle, factor->data);
   }
   if (factor->config != NULL) {
      cudssConfigDestroy(factor->config);
   }
   if (factor->handle != NULL) {
      cudssDestroy(factor->handle);
   }
   cudaFree(factor->device_values);
   cudaFree(factor->device_indices);
   cudaFree(factor->device_indptr);
   free(factor);
}
