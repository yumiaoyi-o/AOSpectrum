#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
: "${CUDSS_ROOT:?Set CUDSS_ROOT to the cuDSS installation prefix}"
: "${PRIMME_ROOT:?Set PRIMME_ROOT to the PRIMME installation prefix}"

BUILD_DIR="${AOSPECTRUM_BUILD_DIR:-build/gpu}"
WHEEL_DIR="${AOSPECTRUM_WHEEL_DIR:-dist}"
EXTRA_CMAKE_ARGS="${AOSPECTRUM_CMAKE_ARGS:-}"
export CMAKE_GENERATOR="${AOSPECTRUM_CMAKE_GENERATOR:-Unix Makefiles}"

mkdir -p "${BUILD_DIR}" "${WHEEL_DIR}"
export CMAKE_ARGS="-DAOSPECTRUM_BUILD_CUDA=ON -DCUDSS_ROOT=${CUDSS_ROOT} -DPRIMME_ROOT=${PRIMME_ROOT} ${EXTRA_CMAKE_ARGS}"
export SKBUILD_BUILD_DIR="${BUILD_DIR}"

"${PYTHON}" -m pip wheel . --no-deps --wheel-dir "${WHEEL_DIR}"
