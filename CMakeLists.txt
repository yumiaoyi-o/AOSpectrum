cmake_minimum_required(VERSION 3.20)
project(AOSpectrum LANGUAGES C CXX)

option(AOSPECTRUM_BUILD_CUDA "Build the optional cuDSS/PRIMME extension" OFF)

if(AOSPECTRUM_BUILD_CUDA)
  enable_language(CUDA)
  add_subdirectory(native)
endif()
