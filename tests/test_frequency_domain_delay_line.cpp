//
// Created by user on 6/12/22.
//
#include "fast_convolution/frequency_domain_delay_line.h"
#include <gmock/gmock.h>

using namespace testing;

class AFrequencyDomainDelayLine : public Test {
public:
    int delay_line_size = 2;
    int freq_size = 5;
};


TEST_F(AFrequencyDomainDelayLine, InitWithSize) {
    FrequencyDomainDelayLine delay_line(delay_line_size, freq_size);

    ASSERT_THAT(delay_line.freqs.size(), Eq(delay_line_size));
    ASSERT_THAT(delay_line.freqs[0].size(), Eq(freq_size));
}

TEST_F(AFrequencyDomainDelayLine, CanGetNextWriteIndex) {
    FrequencyDomainDelayLine delay_line(delay_line_size, freq_size);

    int next_write_index = delay_line.getNextWriteIndex();

    ASSERT_THAT(next_write_index, Eq(0));
}

TEST_F(AFrequencyDomainDelayLine, NextWriteIndexIncreasedAfterGet) {
    FrequencyDomainDelayLine delay_line(delay_line_size, freq_size);

    int next_write_index = delay_line.getNextWriteIndex();
    next_write_index = delay_line.getNextWriteIndex();

    ASSERT_THAT(next_write_index, Eq(1));
}

TEST_F(AFrequencyDomainDelayLine, NextWriteIndexLooped) {
    FrequencyDomainDelayLine delay_line(delay_line_size, freq_size);

    int next_write_index = -1;
    int num_get = 3;
    for (int i = 0; i < num_get; ++i) {
        next_write_index = delay_line.getNextWriteIndex();
    }

    ASSERT_THAT(next_write_index, Eq(num_get % delay_line_size - 1));
}