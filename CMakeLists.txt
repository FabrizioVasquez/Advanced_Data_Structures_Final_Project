cmake_minimum_required(VERSION 3.24)
project(proyecto_final)

include(GNUInstallDirs)
include_directories(${PROJECT_SOURCE_DIR}/include "/home/miunmn/Documents/utec/eda/simpleini")

add_library(mnist ${PROJECT_SOURCE_DIR}/src/MNIST_Dataset.cpp ${PROJECT_SOURCE_DIR}/src/MNIST_Image.cpp)


configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(
        FILES config.ini
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

set(CMAKE_CXX_STANDARD 20)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(proyecto_final main.cpp util/read.hpp util/selection.hpp lib/matrix.hpp util/libraries.hpp lib/hashes.hpp lib/bloomfilter.hpp include/MNIST_Image.hpp include/MNIST_Dataset.hpp src/MNIST_Image.cpp src/MNIST_Dataset.cpp)
target_link_libraries (proyecto_final Eigen3::Eigen)