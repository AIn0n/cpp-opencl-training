cmake_minimum_required(VERSION 3.22)

project(
    training_opencl
    VERSION 0.0.0
    DESCRIPTION "just hacking around with openCL"
)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(train_cpp_cl ${sources})

target_include_directories(train_cpp_cl 
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    )

set_target_properties(train_cpp_cl PROPERTIES OUTPUT_NAME "train_cpp_cl")

add_executable(hello
    src/main.cpp
)

target_link_libraries(hello
    PUBLIC train_cpp_cl
    PUBLIC OpenCL::OpenCL
)

find_package(OpenCL REQUIRED)
