cmake_minimum_required(VERSION 3.16)
project(YogaMudraVisualizer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV — must be installed (handled by Dockerfile)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# nlohmann/json is header-only; we bundle it in cpp/
include_directories(${CMAKE_SOURCE_DIR})

add_executable(visualizer main.cpp)

target_link_libraries(visualizer
    ${OpenCV_LIBS}
)

# Optimised release build
set(CMAKE_BUILD_TYPE Release)
target_compile_options(visualizer PRIVATE -O2 -Wall -Wextra)

message(STATUS "OpenCV version : ${OpenCV_VERSION}")
message(STATUS "Build type     : ${CMAKE_BUILD_TYPE}")
