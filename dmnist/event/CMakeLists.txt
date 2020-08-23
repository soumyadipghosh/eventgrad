cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(event)
#set(CMAKE_BUILD_TYPE Debug)

find_package(Torch REQUIRED)

find_package(MPI REQUIRED)

#include_directories(${MPI_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(event event.cpp)
#target_compile_features(event PUBLIC cxx_range_for)
target_link_libraries(event ${TORCH_LIBRARIES})
target_link_libraries(event ${MPI_LIBRARIES})
#set_property(TARGET event PROPERTY CXX_STANDARD 11)

if(MPI_COMPILE_FLAGS)
  set_target_properties(event PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(event PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
