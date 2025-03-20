cmake_minimum_required(VERSION 3.8)

project(eigen-fft-unit-tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/findFFTW")

option(UNITTEST_BUILD_KISSFFT ON)
option(UNITTEST_BUILD_POCKETFFT ON)
option(UNITTEST_BUILD_FFTW ON)
option(UNITTEST_BUILD_MKL ON) # TODO: ensure nr of threads is 1 for now

# Boost Testing
find_package(Boost CONFIG COMPONENTS unit_test_framework included_unit_test_framework REQUIRED)

message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_UNIT_TEST_FRAMEWORK_LIBRARY: ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}")

# Eigen
if(UNITTEST_USE_TESTING_VERSION)
    set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen-test")
else()
    set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen-official")
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")

# FFTW
if(UNITTEST_BUILD_FFTW)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB FLOAT_LIB)
endif()

# MKL
if(UNITTEST_BUILD_MKL)
    # TODO? enable tests with threading
    set(MKL_THREADING "sequential")
    set(MKL_SYCL_THREADING "sequential")
    find_package(MKL CONFIG REQUIRED)
    # message(STATUS "${MKL_IMPORTED_TARGETS}")
endif()

add_library(UNITTEST_BASE INTERFACE)
target_link_libraries(UNITTEST_BASE INTERFACE Eigen3::Eigen ${Boost_LIBRARIES})
target_include_directories(UNITTEST_BASE INTERFACE
    Boost::boost
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/ext"
    "${CMAKE_SOURCE_DIR}/include/ext/pocketfft"
)

# Executables
set(SRC "main.cpp")

if(UNITTEST_BUILD_KISSFFT)
    add_executable(unittest_kissfftd ${SRC})
    target_link_libraries(unittest_kissfftd UNITTEST_BASE)
endif()

if(UNITTEST_BUILD_POCKETFFT)
    add_executable(unittest_pocketfftd ${SRC})
    target_compile_definitions(unittest_pocketfftd PUBLIC
        EIGEN_POCKETFFT_DEFAULT
    )
    target_link_libraries(unittest_pocketfftd UNITTEST_BASE)
endif()

if(UNITTEST_BUILD_FFTW)
    add_executable(unittest_fftwd ${SRC})
    target_compile_definitions(unittest_fftwd PUBLIC
        EIGEN_FFTW_DEFAULT
    )
    target_link_libraries(unittest_fftwd UNITTEST_BASE FFTW::Double FFTW::Float)
endif()

if(UNITTEST_BUILD_MKL)
    add_executable(unittest_mkld ${SRC})
    target_compile_definitions(unittest_mkld PUBLIC
        EIGEN_MKL_DEFAULT
    )
    target_compile_options(unittest_mkld PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(unittest_mkld PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

    target_link_libraries(unittest_mkld PUBLIC $<LINK_ONLY:MKL::MKL> UNITTEST_BASE)
endif()
