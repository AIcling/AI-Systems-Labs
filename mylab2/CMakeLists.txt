cmake_minimum_required(VERSION 3.10)
project(custom_cpp_extension)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(Torch_DIR /mnt/e/AIsys/labs/microsoft/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(custom_extension
    custom_linear.cpp
    custom_conv.cpp
    custom_dropout2d.cpp
    bindings.cpp  
)

target_link_libraries(custom_extension PRIVATE ${TORCH_LIBRARIES})
target_compile_options(custom_extension PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(custom_extension PRIVATE ${TORCH_INCLUDE_DIRS})
set_target_properties(custom_extension PROPERTIES OUTPUT_NAME "custom_cpp_extension" SUFFIX ".so")