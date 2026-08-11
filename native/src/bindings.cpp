#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aospectrum_cudss_complex64_ubatch.h"
#include "aospectrum_primme_cuda.h"

namespace py = pybind11;

namespace {

constexpr double kHartreeToEv = 27.211386245988;

struct ShiftSolveResult {
  py::array solutions;
  py::dict stage_seconds;
  std::uint64_t peak_gpu_memory_bytes;
  py::dict details;
};

struct NumericUpdateResult {
  int below;
  int zero;
  int above;
  std::uint64_t numeric_epoch;
  py::dict stage_seconds;
  py::dict details;
};

struct DeviceValuesResult {
  py::dict stage_seconds;
  py::dict details;
};

struct PrimmeSolveResult {
  double target_shift_ev;
  py::array energies_ev;
  py::array eigenvectors;
  py::dict stage_seconds;
  py::dict counters;
  py::dict details;
};

template <typename T>
std::vector<T> copy_int_array(const py::array &source, const char *name) {
  auto array = py::array_t<T, py::array::c_style | py::array::forcecast>(source);
  if (array.ndim() != 1 || array.size() == 0) {
    throw std::invalid_argument(std::string(name) + " must be nonempty 1D");
  }
  const auto *data = array.data();
  return std::vector<T>(data, data + array.size());
}

template <typename Scalar>
py::array_t<Scalar> c_array_3d(
    py::ssize_t first, py::ssize_t second, py::ssize_t third) {
  return py::array_t<Scalar>({first, second, third});
}

class CudssShiftSession {
 public:
  CudssShiftSession(
      const py::array &indptr,
      const py::array &indices,
      std::int64_t n,
      std::string scalar_dtype)
      : indptr_(copy_int_array<std::int32_t>(indptr, "indptr")),
        indices_(copy_int_array<std::int32_t>(indices, "indices")),
        n_(n),
        scalar_dtype_(std::move(scalar_dtype)) {
    if (n_ <= 0 || indptr_.size() != static_cast<std::size_t>(n_ + 1)) {
      throw std::invalid_argument("CSR dimension and indptr differ");
    }
    if (indptr_.front() != 0 ||
        indptr_.back() != static_cast<std::int32_t>(indices_.size())) {
      throw std::invalid_argument("CSR indptr endpoints are invalid");
    }
    if (scalar_dtype_ != "complex64") {
      throw std::invalid_argument("CudssShiftSession supports complex64 only");
    }
  }

  ShiftSolveResult solve(
      const py::array &hamiltonian,
      const py::array &overlap,
      const py::array &shifts,
      const py::array &rhs) {
    require_open();
    return solve_complex64(hamiltonian, overlap, shifts, rhs);
  }

  void close() { closed_ = true; }

 private:
  ShiftSolveResult solve_complex64(
      const py::array &hamiltonian,
      const py::array &overlap,
      const py::array &shifts_source,
      const py::array &rhs_source) {
    using Complex = std::complex<float>;
    auto h = py::array_t<Complex, py::array::c_style | py::array::forcecast>(
        hamiltonian);
    auto s = py::array_t<Complex, py::array::c_style | py::array::forcecast>(
        overlap);
    auto shifts =
        py::array_t<Complex, py::array::c_style | py::array::forcecast>(
            shifts_source);
    auto rhs =
        py::array_t<Complex, py::array::c_style | py::array::forcecast>(
            rhs_source);
    const auto nnz = static_cast<py::ssize_t>(indices_.size());
    if (h.ndim() != 1 || s.ndim() != 1 || h.size() != nnz ||
        s.size() != nnz) {
      throw std::invalid_argument("shifted H/S values differ from CSR pattern");
    }
    if (shifts.ndim() != 1 || shifts.size() < 1 || shifts.size() > 64) {
      throw std::invalid_argument("shift batch must contain 1 through 64 values");
    }
    if (rhs.ndim() != 2 || rhs.shape(0) != n_ || rhs.shape(1) <= 0 ||
        rhs.shape(1) > 256) {
      throw std::invalid_argument("RHS must have shape (n, 1..256)");
    }
    const auto batch = shifts.shape(0);
    const auto nrhs = rhs.shape(1);
    std::vector<Complex> values(
        static_cast<std::size_t>(batch * nnz));
    std::vector<Complex> packed_rhs(
        static_cast<std::size_t>(batch * n_ * nrhs));
    std::vector<Complex> packed_solution(packed_rhs.size());
    auto h_view = h.template unchecked<1>();
    auto s_view = s.template unchecked<1>();
    auto shift_view = shifts.template unchecked<1>();
    auto rhs_view = rhs.template unchecked<2>();
    for (py::ssize_t b = 0; b < batch; ++b) {
      const Complex shift = shift_view(b);
      for (py::ssize_t j = 0; j < nnz; ++j) {
        values[static_cast<std::size_t>(b * nnz + j)] =
            h_view(j) - shift * s_view(j);
      }
      for (py::ssize_t column = 0; column < nrhs; ++column) {
        for (std::int64_t row = 0; row < n_; ++row) {
          const auto offset =
              static_cast<std::size_t>(b * n_ * nrhs + column * n_ + row);
          packed_rhs[offset] = rhs_view(row, column);
        }
      }
    }
    auto started = std::chrono::steady_clock::now();
    ShiftSolveResult result;
    aospectrum_cudss_complex64_ubatch_request_v1 request{};
    request.abi_version = AOSPECTRUM_CUDSS_COMPLEX64_UBATCH_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.n = n_;
    request.nnz = nnz;
    request.batch_count = static_cast<std::int32_t>(batch);
    request.rhs_count = static_cast<std::int32_t>(nrhs);
    request.indptr = indptr_.data();
    request.indices = indices_.data();
    request.values = reinterpret_cast<const aospectrum_ubatch_complex64 *>(
        values.data());
    request.rhs = reinterpret_cast<const aospectrum_ubatch_complex64 *>(
        packed_rhs.data());
    request.solution = reinterpret_cast<aospectrum_ubatch_complex64 *>(
        packed_solution.data());
    aospectrum_cudss_complex64_ubatch_result_v1 native{};
    int status;
    {
      py::gil_scoped_release release;
      status = aospectrum_cudss_solve_complex64_ubatch_v1(&request, &native);
    }
    require_cudss_success(status, native.status, native.error_message);
    result.stage_seconds = stage_dict(native);
    result.peak_gpu_memory_bytes = native.peak_device_bytes;
    result.details = detail_dict(native);
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started)
                            .count();
    result.stage_seconds["binding_wall"] = wall;
    auto output = c_array_3d<Complex>(batch, n_, nrhs);
    auto output_view = output.template mutable_unchecked<3>();
    for (py::ssize_t b = 0; b < batch; ++b) {
      for (py::ssize_t column = 0; column < nrhs; ++column) {
        for (std::int64_t row = 0; row < n_; ++row) {
          const auto offset =
              static_cast<std::size_t>(b * n_ * nrhs + column * n_ + row);
          output_view(b, row, column) = packed_solution[offset];
        }
      }
    }
    result.solutions = std::move(output);
    return result;
  }

  template <typename Native>
  static py::dict stage_dict(const Native &value) {
    py::dict result;
    result["device_upload"] = value.upload_seconds;
    result["descriptor_setup"] = value.descriptor_setup_seconds;
    result["symbolic_analysis"] = value.analysis_seconds;
    result["numeric_factorization"] = value.factorization_seconds;
    result["multi_rhs_solve"] = value.solve_seconds;
    result["device_download"] = value.download_seconds;
    result["native_total"] = value.total_seconds;
    return result;
  }

  template <typename Native>
  static py::dict detail_dict(const Native &value) {
    py::dict result;
    result["batch_count"] = value.batch_count;
    result["rhs_count"] = value.rhs_count;
    result["reported_factor_nnz"] = value.reported_factor_nnz;
    result["reported_pivot_count"] = value.reported_pivot_count;
    result["permanent_device_bytes"] = value.permanent_device_bytes;
    result["peak_host_bytes"] = value.peak_host_bytes;
    return result;
  }

  static void require_cudss_success(
      int return_code, int native_status, const char *message) {
    if (return_code != 0 || native_status != 0) {
      throw std::runtime_error(
          "cuDSS shifted solve failed: return=" +
          std::to_string(return_code) + ", status=" +
          std::to_string(native_status) + ", message=" +
          std::string(message == nullptr ? "" : message));
    }
  }

  void require_open() const {
    if (closed_) {
      throw std::runtime_error("CudssShiftSession is closed");
    }
  }

  std::vector<std::int32_t> indptr_;
  std::vector<std::int32_t> indices_;
  std::int64_t n_;
  std::string scalar_dtype_;
  bool closed_{false};
};

class PrimmeStateSession {
 public:
  PrimmeStateSession(
      const py::array &indptr,
      const py::array &indices,
      std::int64_t n,
      const std::string &scalar_dtype)
      : indptr_(copy_int_array<std::int32_t>(indptr, "indptr")),
        indices_(copy_int_array<std::int32_t>(indices, "indices")),
        n_(n) {
    if (scalar_dtype != "float32") {
      throw std::invalid_argument(
          "PrimmeStateSession v1 supports Gamma-point float32 only");
    }
    aospectrum_primme_cuda_session_create_request_v1 request{};
    request.abi_version = AOSPECTRUM_PRIMME_CUDA_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.n = n_;
    request.nnz = static_cast<std::int64_t>(indices_.size());
    request.indptr = indptr_.data();
    request.indices = indices_.data();
    aospectrum_primme_cuda_result_v1 result{};
    const int status = aospectrum_primme_cuda_real32_session_create_v1(
        &request, &session_, &result);
    require_primme_success(status, result, "session create");
    if (session_ == nullptr) {
      throw std::runtime_error("PRIMME session create returned null");
    }
  }

  ~PrimmeStateSession() { close(); }

  NumericUpdateResult update_numeric(
      const py::array &hamiltonian,
      const py::array &overlap,
      double shift_ev) {
    require_open();
    auto h = py::array_t<float, py::array::c_style | py::array::forcecast>(
        hamiltonian);
    auto s = py::array_t<float, py::array::c_style | py::array::forcecast>(
        overlap);
    const auto nnz = static_cast<py::ssize_t>(indices_.size());
    if (h.ndim() != 1 || s.ndim() != 1 || h.size() != nnz ||
        s.size() != nnz) {
      throw std::invalid_argument("PRIMME H/S values differ from CSR pattern");
    }
    std::vector<float> h_hartree(static_cast<std::size_t>(nnz));
    std::vector<float> s_values(static_cast<std::size_t>(nnz));
    auto h_view = h.unchecked<1>();
    auto s_view = s.unchecked<1>();
    for (py::ssize_t index = 0; index < nnz; ++index) {
      h_hartree[static_cast<std::size_t>(index)] =
          static_cast<float>(h_view(index) / kHartreeToEv);
      s_values[static_cast<std::size_t>(index)] = s_view(index);
    }
    const float shift_hartree =
        static_cast<float>(shift_ev / kHartreeToEv);
    aospectrum_primme_cuda_numeric_update_request_v1 request{};
    request.abi_version = AOSPECTRUM_PRIMME_CUDA_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.hamiltonian_values = h_hartree.data();
    request.overlap_values = s_values.data();
    request.target_shift_hartree = shift_hartree;
    request.preconditioner_kind =
        AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT;
    request.diagonal_floor_hartree = 0.0F;
    request.maximum_preconditioner_block_size = 8;
    aospectrum_primme_cuda_result_v1 native{};
    int status;
    {
      py::gil_scoped_release release;
      status = aospectrum_primme_cuda_real32_session_update_v1(
          session_, &request, &native);
    }
    require_primme_success(status, native, "numeric update");
    numeric_epoch_ = native.numeric_epoch;
    shift_hartree_ = shift_hartree;
    const int below = native.cudss_inertia_negative;
    const int above = native.cudss_inertia_positive;
    const int zero = static_cast<int>(n_) - below - above;
    if (below < 0 || above < 0 || zero < 0) {
      throw std::runtime_error("cuDSS inertia counts are invalid");
    }
    NumericUpdateResult result{
        below, zero, above, native.numeric_epoch, py::dict(), py::dict()};
    result.stage_seconds["device_upload"] = native.upload_seconds;
    result.stage_seconds["cudss_preparation"] =
        native.cudss_preparation_seconds;
    result.stage_seconds["cudss_analysis"] = native.cudss_analysis_seconds;
    result.stage_seconds["cudss_factorization"] =
        native.cudss_factorization_seconds;
    result.details["factor_nnz"] = native.cudss_factor_nnz;
    result.details["pivot_count"] = native.cudss_pivot_count;
    result.details["peak_device_bytes"] = native.cudss_peak_device_bytes;
    return result;
  }

  DeviceValuesResult load_device_values(
      std::uintptr_t hamiltonian_pointer,
      std::uintptr_t overlap_pointer,
      std::int64_t nnz,
      std::uintptr_t producer_stream,
      double hamiltonian_scale_to_ev) {
    require_open();
    if (hamiltonian_pointer == 0 || overlap_pointer == 0 ||
        nnz != static_cast<std::int64_t>(indices_.size()) ||
        !std::isfinite(hamiltonian_scale_to_ev) ||
        hamiltonian_scale_to_ev <= 0.0) {
      throw std::invalid_argument(
          "device H/S values differ from the PRIMME CSR contract");
    }
    aospectrum_primme_cuda_device_values_request_v1 request{};
    request.abi_version = AOSPECTRUM_PRIMME_CUDA_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.device_hamiltonian_values =
        reinterpret_cast<const float *>(hamiltonian_pointer);
    request.device_overlap_values =
        reinterpret_cast<const float *>(overlap_pointer);
    request.producer_stream = producer_stream;
    request.hamiltonian_scale_to_hartree = static_cast<float>(
        hamiltonian_scale_to_ev / kHartreeToEv);
    aospectrum_primme_cuda_result_v1 native{};
    int status;
    {
      py::gil_scoped_release release;
      status =
          aospectrum_primme_cuda_real32_session_load_device_values_v1(
              session_, &request, &native);
    }
    require_primme_success(status, native, "device-value admission");
    DeviceValuesResult result{py::dict(), py::dict()};
    result.stage_seconds["device_admission"] = native.upload_seconds;
    result.details["static_device_bytes"] = native.static_device_bytes;
    return result;
  }

  NumericUpdateResult factor_resident(double shift_ev) {
    require_open();
    if (!std::isfinite(shift_ev)) {
      throw std::invalid_argument("resident factor shift is invalid");
    }
    const float shift_hartree =
        static_cast<float>(shift_ev / kHartreeToEv);
    aospectrum_primme_cuda_resident_factor_request_v1 request{};
    request.abi_version = AOSPECTRUM_PRIMME_CUDA_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.target_shift_hartree = shift_hartree;
    request.preconditioner_kind =
        AOSPECTRUM_PRIMME_PRECONDITIONER_CUDSS_SHIFT;
    request.diagonal_floor_hartree = 0.0F;
    request.maximum_preconditioner_block_size = 8;
    aospectrum_primme_cuda_result_v1 native{};
    int status;
    {
      py::gil_scoped_release release;
      status = aospectrum_primme_cuda_real32_session_factor_resident_v1(
          session_, &request, &native);
    }
    require_primme_success(status, native, "resident factorization");
    numeric_epoch_ = native.numeric_epoch;
    shift_hartree_ = shift_hartree;
    const int below = native.cudss_inertia_negative;
    const int above = native.cudss_inertia_positive;
    const int zero = static_cast<int>(n_) - below - above;
    if (below < 0 || above < 0 || zero < 0) {
      throw std::runtime_error("cuDSS inertia counts are invalid");
    }
    NumericUpdateResult result{
        below, zero, above, native.numeric_epoch, py::dict(), py::dict()};
    result.stage_seconds["device_value_prepare"] =
        native.cudss_preparation_seconds;
    result.stage_seconds["cudss_analysis"] = native.cudss_analysis_seconds;
    result.stage_seconds["cudss_factorization"] =
        native.cudss_factorization_seconds;
    result.details["factor_nnz"] = native.cudss_factor_nnz;
    result.details["pivot_count"] = native.cudss_pivot_count;
    result.details["peak_device_bytes"] = native.cudss_peak_device_bytes;
    result.details["analysis_reused"] =
        native.cudss_analysis_seconds == 0.0;
    return result;
  }

  PrimmeSolveResult solve(
      const std::string &target,
      int count,
      double tolerance,
      std::int64_t max_matvecs,
      int max_basis_size,
      int min_restart_size,
      int max_block_size,
      const py::object &initial_vectors_source) {
    require_open();
    if (numeric_epoch_ == 0 || count <= 0 || count >= n_) {
      throw std::invalid_argument("PRIMME solve state/count is invalid");
    }
    int target_code;
    if (target == "closest_leq") {
      target_code = AOSPECTRUM_PRIMME_TARGET_CLOSEST_LEQ;
    } else if (target == "closest_geq") {
      target_code = AOSPECTRUM_PRIMME_TARGET_CLOSEST_GEQ;
    } else {
      throw std::invalid_argument("unsupported PRIMME target");
    }
    std::vector<float> eigenvalues(static_cast<std::size_t>(count));
    std::vector<float> residuals(static_cast<std::size_t>(count));
    std::vector<float> eigenvectors(
        static_cast<std::size_t>(n_ * count));
    using FortranFloatArray = py::array_t<
        float,
        py::array::f_style | py::array::forcecast>;
    FortranFloatArray initial_vectors;
    int initial_size = 0;
    if (!initial_vectors_source.is_none()) {
      initial_vectors = FortranFloatArray(initial_vectors_source);
      if (initial_vectors.ndim() != 2 || initial_vectors.shape(0) != n_ ||
          initial_vectors.shape(1) <= 0 ||
          initial_vectors.shape(1) > count) {
        throw std::invalid_argument(
            "PRIMME initial vectors must have shape (n, 1:count)");
      }
      initial_size = static_cast<int>(initial_vectors.shape(1));
    }
    aospectrum_primme_cuda_solve_request_v1 request{};
    request.abi_version = AOSPECTRUM_PRIMME_CUDA_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.numeric_epoch = numeric_epoch_;
    request.num_evals = count;
    request.target_mode = target_code;
    request.target_shift_hartree = shift_hartree_;
    request.tolerance = static_cast<float>(tolerance);
    request.max_matvecs = max_matvecs;
    request.max_basis_size = max_basis_size;
    request.min_restart_size = min_restart_size;
    request.max_block_size = max_block_size;
    request.init_size = initial_size;
    request.initial_vectors =
        initial_size > 0 ? initial_vectors.data() : nullptr;
    request.return_vectors = 1;
    request.print_level = 0;
    aospectrum_primme_cuda_result_v1 native{};
    int status;
    const auto started = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      status = aospectrum_primme_cuda_real32_session_solve_v1(
          session_,
          &request,
          eigenvalues.data(),
          eigenvectors.data(),
          residuals.data(),
          &native);
    }
    require_primme_success(status, native, "directed solve");
    if (native.converged != count) {
      throw std::runtime_error("PRIMME did not converge every requested state");
    }
    std::vector<int> order(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
      order[static_cast<std::size_t>(index)] = index;
    }
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
      return eigenvalues[static_cast<std::size_t>(left)] <
             eigenvalues[static_cast<std::size_t>(right)];
    });
    py::array_t<float> energies({count});
    py::array_t<float> vectors({n_, static_cast<std::int64_t>(count)});
    std::vector<float> ordered_residuals(static_cast<std::size_t>(count));
    auto energy_view = energies.mutable_unchecked<1>();
    auto vector_view = vectors.mutable_unchecked<2>();
    for (int column = 0; column < count; ++column) {
      const int source_column = order[static_cast<std::size_t>(column)];
      ordered_residuals[static_cast<std::size_t>(column)] =
          residuals[static_cast<std::size_t>(source_column)];
      energy_view(column) =
          eigenvalues[static_cast<std::size_t>(source_column)] *
          static_cast<float>(kHartreeToEv);
      for (std::int64_t row = 0; row < n_; ++row) {
        vector_view(row, column) =
            eigenvectors[static_cast<std::size_t>(source_column * n_ + row)];
      }
    }
    PrimmeSolveResult result{
        static_cast<double>(shift_hartree_) * kHartreeToEv,
        std::move(energies),
        std::move(vectors),
        py::dict(),
        py::dict(),
        py::dict()};
    result.stage_seconds["binding_wall"] = std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                started)
                                                .count();
    result.stage_seconds["native_solve"] = native.solve_seconds;
    result.stage_seconds["device_download"] = native.download_seconds;
    result.counters["matvecs"] = native.matvecs;
    result.counters["preconditions"] = native.preconditions;
    result.counters["restarts"] = native.restarts;
    result.details["factor_reused"] = native.factor_reused;
    result.details["residual_norms_hartree"] = ordered_residuals;
    result.details["peak_device_bytes"] = native.cudss_peak_device_bytes;
    return result;
  }

  void close() {
    if (session_ != nullptr) {
      aospectrum_primme_cuda_real32_session_destroy_v1(session_);
      session_ = nullptr;
    }
  }

 private:
  static void require_primme_success(
      int return_code,
      const aospectrum_primme_cuda_result_v1 &result,
      const char *operation) {
    if (return_code != 0 || result.status != 0) {
      throw std::runtime_error(
          std::string("PRIMME/cuDSS ") + operation +
          " failed: return=" + std::to_string(return_code) +
          ", status=" + std::to_string(result.status) +
          ", primme=" + std::to_string(result.primme_status) +
          ", message=" + std::string(result.error_message));
    }
  }

  void require_open() const {
    if (session_ == nullptr) {
      throw std::runtime_error("PrimmeStateSession is closed");
    }
  }

  std::vector<std::int32_t> indptr_;
  std::vector<std::int32_t> indices_;
  std::int64_t n_;
  aospectrum_primme_cuda_real32_session_v1 *session_{nullptr};
  std::uint64_t numeric_epoch_{0};
  float shift_hartree_{0.0F};
};

}  // namespace

PYBIND11_MODULE(_aospectrum_cuda, module) {
  module.doc() = "AOSpectrum cuDSS/PRIMME CUDA sessions";

  py::class_<ShiftSolveResult>(module, "ShiftSolveResult")
      .def_readonly("solutions", &ShiftSolveResult::solutions)
      .def_readonly("stage_seconds", &ShiftSolveResult::stage_seconds)
      .def_readonly(
          "peak_gpu_memory_bytes",
          &ShiftSolveResult::peak_gpu_memory_bytes)
      .def_readonly("details", &ShiftSolveResult::details);

  py::class_<NumericUpdateResult>(module, "NumericUpdateResult")
      .def_readonly("below", &NumericUpdateResult::below)
      .def_readonly("zero", &NumericUpdateResult::zero)
      .def_readonly("above", &NumericUpdateResult::above)
      .def_readonly("numeric_epoch", &NumericUpdateResult::numeric_epoch)
      .def_readonly("stage_seconds", &NumericUpdateResult::stage_seconds)
      .def_readonly("details", &NumericUpdateResult::details);

  py::class_<DeviceValuesResult>(module, "DeviceValuesResult")
      .def_readonly("stage_seconds", &DeviceValuesResult::stage_seconds)
      .def_readonly("details", &DeviceValuesResult::details);

  py::class_<PrimmeSolveResult>(module, "PrimmeSolveResult")
      .def_readonly("target_shift_ev", &PrimmeSolveResult::target_shift_ev)
      .def_readonly("energies_ev", &PrimmeSolveResult::energies_ev)
      .def_readonly("eigenvectors", &PrimmeSolveResult::eigenvectors)
      .def_readonly("stage_seconds", &PrimmeSolveResult::stage_seconds)
      .def_readonly("counters", &PrimmeSolveResult::counters)
      .def_readonly("details", &PrimmeSolveResult::details);

  py::class_<CudssShiftSession>(module, "CudssShiftSession")
      .def(py::init<
           const py::array &,
           const py::array &,
           std::int64_t,
           std::string>())
      .def("solve", &CudssShiftSession::solve)
      .def("close", &CudssShiftSession::close);

  py::class_<PrimmeStateSession>(module, "PrimmeStateSession")
      .def(py::init<
           const py::array &,
           const py::array &,
           std::int64_t,
           const std::string &>())
      .def("update_numeric", &PrimmeStateSession::update_numeric)
      .def("load_device_values", &PrimmeStateSession::load_device_values)
      .def("factor_resident", &PrimmeStateSession::factor_resident)
      .def(
          "solve",
          &PrimmeStateSession::solve,
          py::arg("target"),
          py::arg("count"),
          py::arg("tolerance"),
          py::arg("max_matvecs"),
          py::arg("max_basis_size"),
          py::arg("min_restart_size"),
          py::arg("max_block_size"),
          py::arg("initial_vectors") = py::none())
      .def("close", &PrimmeStateSession::close);
}
