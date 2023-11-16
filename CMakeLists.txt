cmake_minimum_required(VERSION 3.24)
project(proyecto_final)

set(CMAKE_CXX_STANDARD 20)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


add_executable(proyecto_final main.cpp util/read.hpp util/selection.hpp lib/matrix.hpp util/libraries.hpp)
target_link_libraries (proyecto_final Eigen3::Eigen)
