cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
project(hessian_transform)
set(CMAKE_BUILD_TYPE Release)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_definitions(-std=c++11 -fPIC)
add_definitions(-Wno-deprecated)

add_subdirectory(test)

include_directories(${GIT_REPO_LOC}/math/lib/eigen_3.2.4/)
include_directories(${GIT_REPO_LOC}/LinearResponseVariationalBayes.cpp/include)

add_library(transform_hessian transform_hessian.cpp)
