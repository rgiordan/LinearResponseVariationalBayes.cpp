# cmake_minimum_required(VERSION 2.8)
# cmake_policy(VERSION 2.8.0)
# project(LinearResponseVariationalBayes)
# set(CMAKE_BUILD_TYPE Release)

# set(GIT_REPO_LOC $ENV{GIT_REPO_LOC})

add_definitions(-std=c++11 -fPIC)
add_definitions(-Wno-deprecated)

add_subdirectory(test)

include_directories(${GIT_REPO_LOC}/stan/)
include_directories(${GIT_REPO_LOC}/math/)
include_directories(${EIGEN_HEADER})
include_directories(${BOOST_HEADER})
include_directories(${CVODES_HEADER})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CC_SOURCES kahan_summation.cpp
               exponential_families.cpp
               transform_hessian.cpp
               variational_parameters.cpp
               monte_carlo_parameters.cpp)

add_library(LinearResponseVariationalBayes SHARED ${CC_SOURCES})

set_target_properties(LinearResponseVariationalBayes
                      PROPERTIES VERSION ${VERSION_STRING}
                      SOVERSION ${VERSION_MAJOR})

# Run `sudo make install` to install.
install(TARGETS LinearResponseVariationalBayes DESTINATION /usr/local/lib)
