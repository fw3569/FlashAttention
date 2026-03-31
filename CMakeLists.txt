cmake_minimum_required(VERSION 4.0)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_CUDA_HOST_COMPILER "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER_FORCED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
project(Proj LANGUAGES CXX CUDA)
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
add_library(
  attention_kernel
  SHARED
  ./src/csrc/cublas_handle.cu
  ./src/csrc/kernel/attention_kernel.cu
  ./src/csrc/kernel/softmax_kernel.cu
)
target_include_directories(
  attention_kernel
  PRIVATE
  ./include
)
find_package(CUDAToolkit REQUIRED)
target_link_directories(
  attention_kernel
  PRIVATE
  ${CUDAToolkit_LIBRARY_DIR}
)
target_link_libraries(
  attention_kernel
  PRIVATE
  CUDA::cudart
  CUDA::cublas
)
target_compile_options(
  attention_kernel
  PRIVATE
  $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math -Xcompiler /W0>
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-w -O3 --use_fast_math -Xcompiler "/Zi /W0">
)
target_link_options(
  attention_kernel
  PRIVATE
  $<$<CONFIG:RelWithDebInfo>:-Xlinker /DEBUG>
)
