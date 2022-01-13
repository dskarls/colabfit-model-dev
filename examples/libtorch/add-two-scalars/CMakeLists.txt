cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

list(APPEND CMAKE_PREFIX_PATH "/root/libtorch/libtorch/")

find_package(Torch REQUIRED)

add_executable(add-two-scalars add-two-scalars.cpp)
target_link_libraries(add-two-scalars "${TORCH_LIBRARIES}")
set_property(TARGET add-two-scalars PROPERTY CXX_STANDARD 14)
