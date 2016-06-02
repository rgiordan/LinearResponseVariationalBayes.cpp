cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
project(hessian_transform)
set(CMAKE_BUILD_TYPE Release)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_definitions(-std=c++11 -fPIC)
add_definitions(-Wno-deprecated)

add_subdirectory(test)

include_directories(${GIT_REPO_LOC}stan/)
include_directories(${GIT_REPO_LOC}math/)
include_directories(${GIT_REPO_LOC}math/lib/eigen_3.2.4/)
include_directories(${GIT_REPO_LOC}math/lib/boost_1.58.0/)
include_directories(${GIT_REPO_LOC}/LinearResponseVariationalBayes.cpp/include)

add_library(transform_hessian transform_hessian.cpp)
# add_library(variational_parameters variational_parameters.cpp)
# add_library(exponential_families exponential_families.cpp)
