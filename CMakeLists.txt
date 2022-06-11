cmake_minimum_required(VERSION 3.15)

project(fast_convolution)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(3rdparty)

add_library(fast_convolution
        src/native_convolution.cpp
        )
target_include_directories(fast_convolution
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        )


add_executable(main main.cpp)
target_link_libraries(main PRIVATE fast_convolution kissfft)

if (ENABLE_ASAN)
    target_compile_options(main PRIVATE -fsanitize=address)
    target_link_options(main PRIVATE -fsanitize=address)
endif ()
