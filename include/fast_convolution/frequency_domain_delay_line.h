//
// Created by user on 6/12/22.
//

#ifndef FAST_CONVOLUTION_FREQUENCY_DOMAIN_DELAY_LINE_H
#define FAST_CONVOLUTION_FREQUENCY_DOMAIN_DELAY_LINE_H

#pragma once
#include <complex>
#include <vector>

using Frequencies = std::vector<std::complex<float>>;

class FrequencyDomainDelayLine {
public:
    FrequencyDomainDelayLine() = default;

    FrequencyDomainDelayLine(int delay_line_size,
                             int freq_size)
        : delay_line_size_(delay_line_size),
          freqs_array(delay_line_size) {

        for (auto &f: freqs_array) {
            f.resize(freq_size, {0.0f, 0.0f});
        }
    }

    void resize(int delay_line_size,
                int freq_size)
    {
        delay_line_size_ = delay_line_size;
        freqs_array.resize(delay_line_size);

        for (auto &f: freqs_array) {
            f.resize(freq_size, {0.0f, 0.0f});
        }
    }

    int getDelayedFrequenciesIndex(int delay_n) const {
        delay_n %= delay_line_size_;
        return (write_index_ - 1 - delay_n + delay_line_size_) % delay_line_size_;
    }

    const Frequencies &getDelayedFrequencies(int delay_n) const {
        return freqs_array[getDelayedFrequenciesIndex(delay_n)];
    }

    int getNextWriteIndex() {
        int next = write_index_++;
        if (write_index_ >= freqs_array.size()) {
            write_index_ = 0;
        }
        return next;
    }

    std::vector<Frequencies> freqs_array;

private:
    int delay_line_size_{0};
    int write_index_{0};
};


#endif//FAST_CONVOLUTION_FREQUENCY_DOMAIN_DELAY_LINE_H
