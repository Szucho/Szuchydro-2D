cmake_minimum_required(VERSION 3.12)
project(Szuchydro-2D CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for native architecture and release builds by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add native architecture optimization flag if supported by the compiler
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/headers)

# Automatically glob all .cpp files in the sources directory (including main.cpp)
file(GLOB SOURCES "sources/*.cpp")

# Define the output executable location (putting it in builds/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/builds)

# Create the executable target using all source files found in sources/
add_executable(szuchydro2d ${SOURCES})
