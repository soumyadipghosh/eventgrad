cmake_minimum_required(VERSION 3.1)

project(event)
set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

#include_directories(${MPI_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(${PROJECT_NAME} event.cpp)
target_link_libraries(${PROJECT_NAME} "${MPI_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

if(MPI_COMPILE_FLAGS)
  set_target_properties(event PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(event PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
