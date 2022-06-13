cmake_minimum_required(VERSION 3.15)
project(fast_convolution)

set(CMAKE_CXX_STANDARD 17)

if (ENABLE_ASAN)
    message(STATUS "build with ASAN")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif ()

add_subdirectory(3rdparty)

add_library(fast_convolution
        src/native_convolution.cpp
        )
target_include_directories(fast_convolution
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        )
target_link_libraries(fast_convolution PUBLIC kissfft)

add_subdirectory(example)

enable_testing()
add_subdirectory(tests)
