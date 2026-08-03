cmake_minimum_required(VERSION 3.18)

# CUDA must be declared as a project language so that CMake detects NVCC,
# defines CMAKE_CUDA_COMPILER_VERSION, and compiles .cu source files.
project(OPM_MVS LANGUAGES CXX CUDA)

# ------------------------------------------------------------
# Toolchain checks
# ------------------------------------------------------------
if(NOT MSVC)
    message(FATAL_ERROR
        "This configuration is intended for Microsoft Visual Studio.")
endif()

# Visual Studio 2017 uses MSVC 191x.
if(MSVC_VERSION LESS 1910 OR MSVC_VERSION GREATER_EQUAL 1920)
    message(WARNING
        "This CMake file was prepared for Visual Studio 2017 (MSVC 191x). "
        "Detected MSVC_VERSION=${MSVC_VERSION}.")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR
        "CUDA 11.0 or newer is required. "
        "Detected CUDA ${CMAKE_CUDA_COMPILER_VERSION}.")
endif()

# CUDA 11.8 is recommended for the Visual Studio 2017 toolchain.
if(MSVC_VERSION LESS 1920 AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9)
    message(FATAL_ERROR
        "The detected CUDA toolkit does not support Visual Studio 2017. "
        "Use CUDA 11.8 for this toolchain.")
endif()

# ------------------------------------------------------------
# Language standards
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# GeForce RTX 2060: Turing, compute capability 7.5.
set(CMAKE_CUDA_ARCHITECTURES 75)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(OpenCV 3.3 REQUIRED)
find_package(OpenMP QUIET)

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------
# Use an underscore internally for a convenient CMake target name.
# OUTPUT_NAME below preserves the executable name OPM-MVS.exe.
add_executable(OPM_MVS
    main.cpp
    main.h
    OCMM.cpp
    OCMM.cu
    OCMM.h
)

target_include_directories(OPM_MVS PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(OPM_MVS PRIVATE
    ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(OPM_MVS PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enable M_PI in Microsoft headers and prevent Windows min/max macros.
target_compile_definitions(OPM_MVS PRIVATE
    _USE_MATH_DEFINES
    NOMINMAX
)

# ------------------------------------------------------------
# Compiler options
# ------------------------------------------------------------
target_compile_options(OPM_MVS PRIVATE
    # Microsoft C++ compiler warnings.
    $<$<COMPILE_LANGUAGE:CXX>:/W4>

    # CUDA/NVCC options.
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=128>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>

    # CUDA optimisation for Release builds.
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
)

set_target_properties(OPM_MVS PROPERTIES
    OUTPUT_NAME "OPM-MVS"
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION OFF
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA architecture: sm_75 (RTX 2060)")
