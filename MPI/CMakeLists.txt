cmake_minimum_required(VERSION 3.12)

find_package(MPI REQUIRED)

add_executable(sort sort.cpp)

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

target_link_libraries(sort PRIVATE MPI::MPI_CXX)