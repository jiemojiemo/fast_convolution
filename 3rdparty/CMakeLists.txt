set(KISSFFT_TOOLS OFF)
set(KISSFFT_TEST OFF)
add_subdirectory(kissfft)
add_subdirectory(googletest)
add_subdirectory(portaudio)

add_library(dr_libs INTERFACE)
target_include_directories(dr_libs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dr_libs)