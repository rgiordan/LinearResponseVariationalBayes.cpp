cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)
project(LinearResponseVariationalBayes)
set(CMAKE_BUILD_TYPE Release)

set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

# TOOD: can you glob these?
set(EIGEN_HEADER ${GIT_REPO_LOC}/math/lib/eigen_3.2.9/)
set(BOOST_HEADER ${GIT_REPO_LOC}/math/lib/boost_1.60.0/)
set(CVODES_HEADER ${GIT_REPO_LOC}/math/lib/cvodes_2.8.2/)

set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)
set(VERSION_STRING ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

add_subdirectory(src)

# Uncomment to debug
set(CMAKE_BUILD_TYPE Debug)
