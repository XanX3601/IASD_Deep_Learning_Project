cmake_minimum_required(VERSION 3.0)
project(golois)

include_directories(
  ./include
)

set(source_files
)

set(test_files
)

add_subdirectory(libs/pybind11)
pybind11_add_module(golois src/golois.cpp)
