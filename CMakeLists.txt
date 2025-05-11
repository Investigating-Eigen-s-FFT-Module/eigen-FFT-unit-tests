cmake_minimum_required(VERSION 3.8)

project(eigen-fft-unit-tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/findFFTW")

# GTest
find_package(GTest REQUIRED)

# Eigen
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen-test")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")

# FFTW
if(UNITTEST_BUILD_FFTW)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB FLOAT_LIB)
endif()

add_library(UNITTEST_BASE INTERFACE)
target_link_libraries(UNITTEST_BASE INTERFACE
    Eigen3::Eigen
    ${GTEST_LIBRARIES}
)
target_include_directories(UNITTEST_BASE INTERFACE
    ${GTEST_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/ext"
    "${CMAKE_SOURCE_DIR}/include/ext/pocketfft"
)

add_executable(run_tests "main.cpp")
target_link_libraries(run_tests PRIVATE UNITTEST_BASE)