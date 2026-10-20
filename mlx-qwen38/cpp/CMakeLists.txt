cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING "")
project(qwen38_cpp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Same MLX build as the Python loader (oMLX framework-mlx-base, mlx 0.32.2) for bit-identical kernels.
set(MLX_ROOT "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages/mlx" CACHE PATH "MLX install root")
find_package(MLX CONFIG REQUIRED PATHS "${MLX_ROOT}/share/cmake/MLX" NO_DEFAULT_PATH)
message(STATUS "MLX version: ${MLX_VERSION}  include: ${MLX_INCLUDE_DIRS}")

add_library(qwen38core STATIC
  src/kernels.cpp
  src/ple.cpp
  src/layers.cpp
  src/loader.cpp
  src/engine.cpp
  src/io_sched.cpp
  third_party/expert_io.c
)
target_include_directories(qwen38core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(qwen38core PUBLIC mlx)
target_compile_options(qwen38core PRIVATE -O3)

add_executable(mlx_hello tools/mlx_hello.cpp)
target_link_libraries(mlx_hello PRIVATE mlx)
set_target_properties(mlx_hello PROPERTIES BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

add_executable(qwen38_decode tools/qwen38_decode.cpp)
target_link_libraries(qwen38_decode PRIVATE qwen38core)
set_target_properties(qwen38_decode PROPERTIES BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels.cpp")
  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE qwen38core)
  set_target_properties(test_kernels PROPERTIES BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_prefill_qproj.cpp")
  add_executable(test_prefill_qproj tests/test_prefill_qproj.cpp)
  target_include_directories(test_prefill_qproj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_prefill_qproj PRIVATE mlx)
  set_target_properties(test_prefill_qproj PROPERTIES BUILD_RPATH "${MLX_ROOT}/lib" INSTALL_RPATH "${MLX_ROOT}/lib")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_budget.cpp")
  add_executable(test_budget tests/test_budget.cpp)
  target_include_directories(test_budget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(test_budget PRIVATE -O2 -UNDEBUG)  # asserts must stay live in Release
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_io_sched.cpp")
  add_executable(test_io_sched tests/test_io_sched.cpp src/io_sched.cpp)
  target_include_directories(test_io_sched PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(test_io_sched PRIVATE -O2 -UNDEBUG)  # asserts must stay live in Release
endif()

