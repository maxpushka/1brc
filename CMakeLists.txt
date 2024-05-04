cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(1brc LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 20)
add_executable(${PROJECT_NAME}
        src/main.cc
        src/thread_pool.h
)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "IPO / LTO not supported: <${error}>")
endif ()

# Enable SSE2 instructions
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-msse2" COMPILER_SUPPORTS_SSE2)
if(COMPILER_SUPPORTS_SSE2)
    target_compile_options(${PROJECT_NAME} PRIVATE -msse2)
endif()
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(${PROJECT_NAME} PRIVATE -mavx2)
endif()
CHECK_CXX_COMPILER_FLAG("-mavx512f -mavx512bw" COMPILER_SUPPORTS_AVX512BW)
if(COMPILER_SUPPORTS_AVX512BW)
    target_compile_options(${PROJECT_NAME} PRIVATE -mavx512f -mavx512bw)
endif()
