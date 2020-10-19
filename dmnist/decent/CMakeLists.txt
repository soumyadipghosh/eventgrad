cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cwdecent)

find_package(Torch REQUIRED)

find_package(MPI REQUIRED)

#include_directories(${MPI_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(cwdecent cwdecent.cpp)
#target_compile_features(cwdecent PUBLIC cxx_range_for)
target_link_libraries(cwdecent ${TORCH_LIBRARIES})
target_link_libraries(cwdecent ${MPI_LIBRARIES})
#set_property(TARGET cwdecent PROPERTY CXX_STANDARD 11)

if(MPI_COMPILE_FLAGS)
  set_target_properties(cwdecent PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(cwdecent PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
