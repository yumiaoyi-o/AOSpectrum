find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(CUDSS_ROOT "" CACHE PATH "cuDSS installation prefix")
set(PRIMME_ROOT "" CACHE PATH "PRIMME CUDA installation prefix")

find_path(
  CUDSS_INCLUDE_DIR
  NAMES cudss.h
  HINTS "${CUDSS_ROOT}/include"
  REQUIRED
)
find_library(
  CUDSS_LIBRARY
  NAMES cudss libcudss.so.0
  HINTS "${CUDSS_ROOT}/lib" "${CUDSS_ROOT}/lib64"
  REQUIRED
)
find_path(
  PRIMME_INCLUDE_DIR
  NAMES primme.h
  HINTS "${PRIMME_ROOT}/include"
  REQUIRED
)
find_library(
  PRIMME_LIBRARY
  NAMES primme
  HINTS "${PRIMME_ROOT}/lib" "${PRIMME_ROOT}/lib64"
  REQUIRED
)

pybind11_add_module(
  _aospectrum_cuda
  MODULE
  src/bindings.cpp
  src/aospectrum_cudss_complex64_ubatch.c
  src/aospectrum_cudss_shift_factor.c
  src/aospectrum_cuda_ops.cu
  src/aospectrum_primme_cuda.c
)
target_compile_features(_aospectrum_cuda PRIVATE cxx_std_17)
target_include_directories(
  _aospectrum_cuda
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CUDSS_INCLUDE_DIR}"
  "${PRIMME_INCLUDE_DIR}"
)
target_link_libraries(
  _aospectrum_cuda
  PRIVATE
  "${CUDSS_LIBRARY}"
  "${PRIMME_LIBRARY}"
  CUDA::cudart
  CUDA::cusparse
  CUDA::cublas
  BLAS::BLAS
  LAPACK::LAPACK
)
set_target_properties(
  _aospectrum_cuda
  PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  INSTALL_RPATH_USE_LINK_PATH ON
)
install(TARGETS _aospectrum_cuda LIBRARY DESTINATION aospectrum)
