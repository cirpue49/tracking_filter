cmake_minimum_required(VERSION 3.8)
project(help_filter)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp ukf.h ukf.cpp measurement_package.h tools.cpp tools.h ground_truth_package.h)
add_executable(help_filter ${SOURCE_FILES})