cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(1brc LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 20)

# Specify ARM and Neon flags
if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd -O3")
endif ()

add_executable(${PROJECT_NAME}
        src/main.cc
        src/thread_pool.h
)
target_compile_definitions(${PROJECT_NAME} PRIVATE HARDWARE_CONCURRENCY=${HARDWARE_CONCURRENCY})
