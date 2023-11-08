cmake_minimum_required(VERSION 3.12)

project(bitonic LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)

add_executable(sort sort.cu)

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(sort PRIVATE MPI::MPI_CXX)
