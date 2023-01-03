cmake_minimum_required(VERSION 3.6)

project(cuda_cmm)

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )

# link_directories(
#     /'Program Files'/'NVIDIA GPU Computing Toolkit'/CUDA/v11.8/lib/x64
#     )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(cuda_cmm SHARED
  cuda_cmm.cu)

target_link_libraries(cuda_cmm
  ${PYTHON_LIBRARIES})

set_target_properties(cuda_cmm PROPERTIES PREFIX "")

add_custom_command(TARGET cuda_cmm POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_BINARY_DIR}/Release/cuda_cmm${CMAKE_SHARED_LIBRARY_SUFFIX}
  ${CMAKE_CURRENT_BINARY_DIR}/Release/cuda_cmm.pyd)
