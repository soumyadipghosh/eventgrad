cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(ex)

find_package(Torch REQUIRED)

add_executable(ex ex.cpp)
#target_compile_features(ex PUBLIC cxx_range_for)
target_link_libraries(ex ${TORCH_LIBRARIES})
#set_property(TARGET ex PROPERTY CXX_STANDARD 11)

