#define _POSIX_C_SOURCE 200809L

#include "aospectrum_cudss_complex64_ubatch.h"

#include <cudss.h>
#include <cuda_runtime.h>

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(
   sizeof(aospectrum_ubatch_complex64) == 2 * sizeof(float),
   "aospectrum_ubatch_complex64 must match NumPy complex64 storage");

static double monotonic_seconds(void) {
   struct timespec value;
   if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
      return 0.0;
   }
   return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int trace_enabled(void) {
   const char *value = getenv("AOSPECTRUM_CUDSS_TRACE");
   return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void trace_event(
      const aospectrum_cudss_complex64_ubatch_request_v1 *request,
      double call_started,
      const char *phase,
      const char *event) {
   if (!trace_enabled()) {
      return;
   }
   fprintf(
      stderr,
      "AOSPECTRUM_CUDSS_TRACE|elapsed_seconds=%.9f|phase=%s|event=%s"
      "|n=%lld|nnz=%lld|batch=%d|rhs=%d\n",
      monotonic_seconds() - call_started,
      phase,
      event,
      (long long)request->n,
      (long long)request->nnz,
      request->batch_count,
      request->rhs_count);
   fflush(stderr);
}

static void set_error(
      aospectrum_cudss_complex64_ubatch_result_v1 *result,
      int status,
      const char *format,
      ...) {
   va_list arguments;
   result->status = status;
   va_start(arguments, format);
   vsnprintf(
      result->error_message,
      AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_ERROR_CAPACITY,
      format,
      arguments);
   va_end(arguments);
}

static int check_cuda(
      cudaError_t status,
      aospectrum_cudss_complex64_ubatch_result_v1 *result,
      const char *operation) {
   if (status == cudaSuccess) {
      return 0;
   }
   set_error(
      result,
      20,
      "%s failed: %s",
      operation,
      cudaGetErrorString(status));
   return 1;
}

static int check_cudss(
      cudssStatus_t status,
      aospectrum_cudss_complex64_ubatch_result_v1 *result,
      const char *operation) {
   result->cudss_status = (int32_t)status;
   if (status == CUDSS_STATUS_SUCCESS) {
      return 0;
   }
   set_error(
      result,
      21,
      "%s failed with cuDSS status %d",
      operation,
      (int)status);
   return 1;
}

static int query_data_info(
      cudssHandle_t handle,
      cudssData_t data,
      aospectrum_cudss_complex64_ubatch_result_v1 *result,
      const char *phase) {
   int info = 0;
   size_t bytes_written = 0;
   if (check_cudss(
         cudssDataGet(
            handle,
            data,
            CUDSS_DATA_INFO,
            &info,
            sizeof(info),
            &bytes_written),
         result,
         "cuDSS device-info query")) {
      return 1;
   }
   if (bytes_written != sizeof(info) || info != 0) {
      set_error(
         result,
         22,
         "%s reported cuDSS device info %d",
         phase,
         info);
      return 1;
   }
   return 0;
}

static int validate_request(
      const aospectrum_cudss_complex64_ubatch_request_v1 *request,
      aospectrum_cudss_complex64_ubatch_result_v1 *result) {
   if (request == NULL || result == NULL) {
      return 1;
   }
   memset(result, 0, sizeof(*result));
   result->abi_version = AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_ABI_VERSION;
   result->struct_size = (uint32_t)sizeof(*result);
   if (request->abi_version != AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_ABI_VERSION ||
         request->struct_size != sizeof(*request)) {
      set_error(result, 10, "request ABI mismatch");
      return 1;
   }
   if (request->n <= 0 || request->n > INT32_MAX ||
         request->nnz <= 0 || request->nnz > INT32_MAX) {
      set_error(result, 11, "matrix dimensions exceed the int32 CSR ABI");
      return 1;
   }
   if (request->batch_count <= 0 || request->batch_count > 64 ||
         request->rhs_count <= 0 || request->rhs_count > 256) {
      set_error(result, 12, "invalid uniform-batch dimensions");
      return 1;
   }
   if (request->indptr == NULL || request->indices == NULL ||
         request->values == NULL || request->rhs == NULL ||
         request->solution == NULL) {
      set_error(result, 13, "required input or output pointer is null");
      return 1;
   }
   if (request->indptr[0] != 0 ||
         request->indptr[request->n] != request->nnz) {
      set_error(result, 14, "CSR row offsets do not span the declared nnz");
      return 1;
   }

   int64_t diagonal_count = 0;
   for (int64_t row = 0; row < request->n; ++row) {
      const int32_t begin = request->indptr[row];
      const int32_t end = request->indptr[row + 1];
      if (begin < 0 || end < begin || end > request->nnz) {
         set_error(result, 30, "CSR row offsets are not monotonic");
         return 1;
      }
      int32_t previous_column = -1;
      int has_diagonal = 0;
      for (int32_t position = begin; position < end; ++position) {
         const int32_t column = request->indices[position];
         if (column < 0 || column >= request->n) {
            set_error(result, 31, "CSR column index is outside the matrix");
            return 1;
         }
         if (column <= previous_column) {
            set_error(
               result,
               32,
               "CSR columns must be sorted and duplicate-free");
            return 1;
         }
         previous_column = column;
         has_diagonal |= column == row;
      }
      diagonal_count += has_diagonal;
   }
   if (diagonal_count != request->n) {
      set_error(
         result,
         33,
         "shifted matrices are missing diagonal entries: %lld/%lld",
         (long long)diagonal_count,
         (long long)request->n);
      return 1;
   }

   const uint64_t matrix_entries =
      (uint64_t)request->batch_count * (uint64_t)request->nnz;
   const uint64_t dense_entries =
      (uint64_t)request->batch_count * (uint64_t)request->n *
      (uint64_t)request->rhs_count;
   if (matrix_entries > SIZE_MAX / sizeof(aospectrum_ubatch_complex64) ||
         dense_entries > SIZE_MAX / sizeof(aospectrum_ubatch_complex64)) {
      set_error(result, 34, "uniform-batch storage exceeds size_t");
      return 1;
   }
   for (uint64_t index = 0; index < matrix_entries; ++index) {
      const aospectrum_ubatch_complex64 value = request->values[index];
      if (!isfinite(value.real) || !isfinite(value.imag)) {
         set_error(result, 35, "shifted matrix batch contains non-finite values");
         return 1;
      }
   }
   for (uint64_t index = 0; index < dense_entries; ++index) {
      const aospectrum_ubatch_complex64 value = request->rhs[index];
      if (!isfinite(value.real) || !isfinite(value.imag)) {
         set_error(result, 36, "right-hand-side batch contains non-finite values");
         return 1;
      }
   }

   result->n = request->n;
   result->nnz = request->nnz;
   result->diagonal_count = diagonal_count;
   result->batch_count = request->batch_count;
   result->rhs_count = request->rhs_count;
   return 0;
}

int aospectrum_cudss_solve_complex64_ubatch_v1(
      const aospectrum_cudss_complex64_ubatch_request_v1 *request,
      aospectrum_cudss_complex64_ubatch_result_v1 *result) {
   int32_t *device_indptr = NULL;
   int32_t *device_indices = NULL;
   aospectrum_ubatch_complex64 *device_values = NULL;
   aospectrum_ubatch_complex64 *device_rhs = NULL;
   aospectrum_ubatch_complex64 *device_solution = NULL;
   cudaStream_t stream = NULL;
   cudssHandle_t handle = NULL;
   cudssConfig_t config = NULL;
   cudssData_t data = NULL;
   cudssMatrix_t matrix = NULL;
   cudssMatrix_t rhs = NULL;
   cudssMatrix_t solution = NULL;
   int return_code = 1;
   double total_started = monotonic_seconds();

   if (validate_request(request, result)) {
      return 1;
   }
   trace_event(request, total_started, "native_call", "begin");

   size_t free_bytes = 0;
   size_t total_bytes = 0;
   if (check_cuda(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         result,
         "initial cudaMemGetInfo")) {
      goto cleanup;
   }
   result->free_device_bytes_before = free_bytes;
   result->total_device_bytes = total_bytes;

   const size_t indptr_bytes =
      ((size_t)request->n + 1) * sizeof(*device_indptr);
   const size_t indices_bytes =
      (size_t)request->nnz * sizeof(*device_indices);
   const size_t values_bytes =
      (size_t)request->batch_count * (size_t)request->nnz *
      sizeof(*device_values);
   const size_t dense_bytes =
      (size_t)request->batch_count * (size_t)request->n *
      (size_t)request->rhs_count * sizeof(*device_rhs);

   double started = monotonic_seconds();
   if (check_cuda(cudaStreamCreate(&stream), result, "stream creation") ||
       check_cuda(
         cudaMalloc((void **)&device_indptr, indptr_bytes),
         result,
         "row-offset allocation") ||
       check_cuda(
         cudaMalloc((void **)&device_indices, indices_bytes),
         result,
         "column-index allocation") ||
       check_cuda(
         cudaMalloc((void **)&device_values, values_bytes),
         result,
         "matrix-value batch allocation") ||
       check_cuda(
         cudaMalloc((void **)&device_rhs, dense_bytes),
         result,
         "right-hand-side batch allocation") ||
       check_cuda(
         cudaMalloc((void **)&device_solution, dense_bytes),
         result,
         "solution batch allocation") ||
       check_cuda(
         cudaMemcpyAsync(
            device_indptr,
            request->indptr,
            indptr_bytes,
            cudaMemcpyHostToDevice,
            stream),
         result,
         "row-offset upload") ||
       check_cuda(
         cudaMemcpyAsync(
            device_indices,
            request->indices,
            indices_bytes,
            cudaMemcpyHostToDevice,
            stream),
         result,
         "column-index upload") ||
       check_cuda(
         cudaMemcpyAsync(
            device_values,
            request->values,
            values_bytes,
            cudaMemcpyHostToDevice,
            stream),
         result,
         "matrix-value batch upload") ||
       check_cuda(
         cudaMemcpyAsync(
            device_rhs,
            request->rhs,
            dense_bytes,
            cudaMemcpyHostToDevice,
            stream),
         result,
         "right-hand-side batch upload") ||
       check_cuda(
         cudaMemsetAsync(device_solution, 0, dense_bytes, stream),
         result,
         "solution batch initialization") ||
       check_cuda(
         cudaStreamSynchronize(stream),
         result,
         "input batch upload synchronization")) {
      goto cleanup;
   }
   result->upload_seconds = monotonic_seconds() - started;
   trace_event(request, total_started, "upload", "complete");
   if (check_cuda(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         result,
         "post-upload cudaMemGetInfo")) {
      goto cleanup;
   }
   result->free_device_bytes_after_upload = free_bytes;

   started = monotonic_seconds();
   if (check_cudss(cudssCreate(&handle), result, "handle creation") ||
       check_cudss(cudssSetStream(handle, stream), result, "stream binding") ||
       check_cudss(cudssConfigCreate(&config), result, "config creation") ||
       check_cudss(cudssDataCreate(handle, &data), result, "data creation")) {
      goto cleanup;
   }
   int uniform_batch_size = request->batch_count;
   int uniform_batch_index = -1;
   int iterative_refinement_steps = 2;
   result->iterative_refinement_steps = iterative_refinement_steps;
   if (check_cudss(
         cudssConfigSet(
            config,
            CUDSS_CONFIG_UBATCH_SIZE,
            &uniform_batch_size,
            sizeof(uniform_batch_size)),
         result,
         "uniform-batch size configuration") ||
       check_cudss(
         cudssConfigSet(
            config,
            CUDSS_CONFIG_UBATCH_INDEX,
            &uniform_batch_index,
            sizeof(uniform_batch_index)),
         result,
         "uniform-batch all-systems configuration") ||
       check_cudss(
         cudssConfigSet(
            config,
            CUDSS_CONFIG_IR_N_STEPS,
            &iterative_refinement_steps,
            sizeof(iterative_refinement_steps)),
         result,
         "fixed iterative-refinement configuration") ||
       check_cudss(
         cudssMatrixCreateCsr(
            &matrix,
            request->n,
            request->n,
            request->nnz,
            device_indptr,
            NULL,
            device_indices,
            device_values,
            CUDSS_R_32I,
            CUDSS_R_32I,
            CUDSS_C_32F,
            CUDSS_MTYPE_GENERAL,
            CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO),
         result,
         "uniform sparse matrix batch creation") ||
       check_cudss(
         cudssMatrixCreateDn(
            &rhs,
            request->n,
            request->rhs_count,
            request->n,
            device_rhs,
            CUDSS_C_32F,
            CUDSS_LAYOUT_COL_MAJOR),
         result,
         "uniform right-hand-side batch creation") ||
       check_cudss(
         cudssMatrixCreateDn(
            &solution,
            request->n,
            request->rhs_count,
            request->n,
            device_solution,
            CUDSS_C_32F,
            CUDSS_LAYOUT_COL_MAJOR),
         result,
         "uniform solution batch creation")) {
      goto cleanup;
   }
   result->descriptor_setup_seconds = monotonic_seconds() - started;
   trace_event(request, total_started, "descriptor_setup", "complete");

   started = monotonic_seconds();
   trace_event(request, total_started, "analysis_execute", "begin");
   if (check_cudss(
         cudssExecute(
            handle,
            CUDSS_PHASE_ANALYSIS,
            config,
            data,
            matrix,
            solution,
            rhs),
         result,
         "uniform-batch symbolic analysis")) {
      goto cleanup;
   }
   trace_event(request, total_started, "analysis_execute", "complete");
   trace_event(request, total_started, "analysis_sync", "begin");
   if (check_cuda(
         cudaStreamSynchronize(stream),
         result,
         "uniform-batch analysis synchronization")) {
      goto cleanup;
   }
   trace_event(request, total_started, "analysis_sync", "complete");
   if (query_data_info(handle, data, result, "uniform-batch analysis")) {
      goto cleanup;
   }
   result->analysis_seconds = monotonic_seconds() - started;
   trace_event(request, total_started, "analysis", "complete");

   int64_t memory_estimates[16] = {0};
   size_t bytes_written = 0;
   if (check_cudss(
         cudssDataGet(
            handle,
            data,
            CUDSS_DATA_MEMORY_ESTIMATES,
            memory_estimates,
            sizeof(memory_estimates),
            &bytes_written),
         result,
         "uniform-batch memory-estimate query")) {
      goto cleanup;
   }
   if (bytes_written < 4 * sizeof(memory_estimates[0])) {
      set_error(
         result,
         40,
         "cuDSS returned an incomplete memory estimate: %zu bytes",
         bytes_written);
      goto cleanup;
   }
   result->permanent_device_bytes =
      memory_estimates[0] > 0 ? (uint64_t)memory_estimates[0] : 0;
   result->peak_device_bytes =
      memory_estimates[1] > 0 ? (uint64_t)memory_estimates[1] : 0;
   result->permanent_host_bytes =
      memory_estimates[2] > 0 ? (uint64_t)memory_estimates[2] : 0;
   result->peak_host_bytes =
      memory_estimates[3] > 0 ? (uint64_t)memory_estimates[3] : 0;
   trace_event(request, total_started, "memory_estimate", "complete");

   started = monotonic_seconds();
   trace_event(request, total_started, "factorization_execute", "begin");
   if (check_cudss(
         cudssExecute(
            handle,
            CUDSS_PHASE_FACTORIZATION,
            config,
            data,
            matrix,
            solution,
            rhs),
         result,
         "uniform-batch numerical factorization")) {
      goto cleanup;
   }
   trace_event(request, total_started, "factorization_execute", "complete");
   trace_event(request, total_started, "factorization_sync", "begin");
   if (check_cuda(
         cudaStreamSynchronize(stream),
         result,
         "uniform-batch factorization synchronization")) {
      goto cleanup;
   }
   trace_event(request, total_started, "factorization_sync", "complete");
   if (query_data_info(handle, data, result, "uniform-batch factorization")) {
      goto cleanup;
   }
   result->factorization_seconds = monotonic_seconds() - started;
   trace_event(request, total_started, "factorization", "complete");

   int64_t factor_nnz = 0;
   int32_t pivot_count = 0;
   if (check_cudss(
         cudssDataGet(
            handle,
            data,
            CUDSS_DATA_LU_NNZ,
            &factor_nnz,
            sizeof(factor_nnz),
            &bytes_written),
         result,
         "uniform-batch factor-nnz query") ||
       check_cudss(
         cudssDataGet(
            handle,
            data,
            CUDSS_DATA_NPIVOTS,
            &pivot_count,
            sizeof(pivot_count),
            &bytes_written),
         result,
         "uniform-batch pivot-count query")) {
      goto cleanup;
   }
   result->reported_factor_nnz = factor_nnz;
   result->reported_pivot_count = pivot_count;
   if (check_cuda(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         result,
         "post-factorization cudaMemGetInfo")) {
      goto cleanup;
   }
   result->free_device_bytes_after_factorization = free_bytes;

   started = monotonic_seconds();
   trace_event(request, total_started, "solve_execute", "begin");
   if (check_cudss(
         cudssExecute(
            handle,
            CUDSS_PHASE_SOLVE,
            config,
            data,
            matrix,
            solution,
            rhs),
         result,
         "uniform-batch multi-right-hand-side solve")) {
      goto cleanup;
   }
   trace_event(request, total_started, "solve_execute", "complete");
   trace_event(request, total_started, "solve_sync", "begin");
   if (check_cuda(
         cudaStreamSynchronize(stream),
         result,
         "uniform-batch solve synchronization")) {
      goto cleanup;
   }
   trace_event(request, total_started, "solve_sync", "complete");
   if (query_data_info(handle, data, result, "uniform-batch solve")) {
      goto cleanup;
   }
   result->solve_seconds = monotonic_seconds() - started;
   trace_event(request, total_started, "solve", "complete");
   if (check_cuda(
         cudaMemGetInfo(&free_bytes, &total_bytes),
         result,
         "post-solve cudaMemGetInfo")) {
      goto cleanup;
   }
   result->free_device_bytes_after_solve = free_bytes;

   started = monotonic_seconds();
   if (check_cuda(
         cudaMemcpyAsync(
            request->solution,
            device_solution,
            dense_bytes,
            cudaMemcpyDeviceToHost,
            stream),
         result,
         "solution batch download") ||
       check_cuda(
         cudaStreamSynchronize(stream),
         result,
         "solution batch download synchronization")) {
      goto cleanup;
   }
   result->download_seconds = monotonic_seconds() - started;
   trace_event(request, total_started, "download", "complete");
   result->status = 0;
   return_code = 0;

cleanup:
   result->total_seconds = monotonic_seconds() - total_started;
   trace_event(
      request,
      total_started,
      "native_call",
      return_code == 0 ? "complete" : "cleanup");
   if (stream != NULL) {
      /*
       * Uniform-batch factorization and solves may still be queued when an
       * API reports an error.  Complete outstanding work before releasing
       * descriptors and device buffers.
       */
      cudaStreamSynchronize(stream);
   }
   if (solution != NULL) {
      cudssMatrixDestroy(solution);
   }
   if (rhs != NULL) {
      cudssMatrixDestroy(rhs);
   }
   if (matrix != NULL) {
      cudssMatrixDestroy(matrix);
   }
   if (data != NULL && handle != NULL) {
      cudssDataDestroy(handle, data);
   }
   if (config != NULL) {
      cudssConfigDestroy(config);
   }
   if (handle != NULL) {
      cudssDestroy(handle);
   }
   cudaFree(device_solution);
   cudaFree(device_rhs);
   cudaFree(device_values);
   cudaFree(device_indices);
   cudaFree(device_indptr);
   if (stream != NULL) {
      cudaStreamDestroy(stream);
   }
   return return_code;
}
