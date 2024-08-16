cmake_minimum_required(VERSION 3.10)
project(custom_linear_extension)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(Torch_DIR /mnt/e/AIsys/labs/microsoft/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(pybind11 REQUIRED)

pybind11_add_module(custom_linear_cpp custom_linear.cpp)

target_link_libraries(custom_linear_cpp PRIVATE ${TORCH_LIBRARIES})

target_compile_options(custom_linear_cpp PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(custom_linear_cpp PRIVATE ${TORCH_INCLUDE_DIRS})

set_target_properties(custom_linear_cpp PROPERTIES OUTPUT_NAME "custom_linear" SUFFIX ".so")
