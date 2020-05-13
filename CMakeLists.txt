cmake_minimum_required(VERSION 3.14)
project(OPPlab3)

set(CMAKE_CXX_STANDARD 14)

add_executable(OPPlab3 main.cpp Test.cpp Test.h InitABC.h InitABC.cpp)
# Require MPI for this project:

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(OPPlab3 ${MPI_LIBRARIES})