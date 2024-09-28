cmake_minimum_required(VERSION 3.5.0)
project(CarNumerics VERSION 0.1.0 LANGUAGES C CXX)

find_package (Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(CarNumerics Solver.cpp main.cpp)

target_link_libraries (CarNumerics ${ARMADILLO_LIBRARIES})

include(CTest)
enable_testing()

