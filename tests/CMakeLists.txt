add_executable(unit_test
        test_frequency_domain_delay_line.cpp
        test_realtime_uniformly_partitioned_conv.cpp
        test_realtime_uniformly_partitioned_conv_utilities.cpp
        test_utilities.cpp
        )

target_link_libraries(unit_test PRIVATE gmock_main fast_convolution)