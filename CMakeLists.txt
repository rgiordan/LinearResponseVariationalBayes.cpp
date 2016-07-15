cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
project(LinearResponseVariationalBayes)
set(CMAKE_BUILD_TYPE Release)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})
set(EIGEN_HEADER ${GIT_REPO_LOC}/math/lib/eigen_3.2.8/)
set(BOOST_HEADER ${GIT_REPO_LOC}/math/lib/boost_1.60.0/)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)
set(VERSION_STRING ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

add_subdirectory(src)
