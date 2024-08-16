cmake_minimum_required(VERSION 3.10)
project(custom_linear_extension)

set(CMAKE_CXX_STANDARD 14)

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Torch REQUIRED)

if(NOT TARGET pybind11)
    add_subdirectory(pybind11)
endif()

pybind11_add_module(custom_linear_cpp custom_linear.cpp)

target_link_libraries(custom_linear_cpp PRIVATE ${TORCH_LIBRARIES})

target_compile_options(custom_linear_cpp PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(custom_linear_cpp PRIVATE ${TORCH_INCLUDE_DIRS})

set_target_properties(custom_linear_cpp PROPERTIES PREFIX "")
