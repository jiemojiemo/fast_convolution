//
// Created by user on 6/12/22.
//

#ifndef FAST_CONVOLUTION_FREQUENCY_DOMAIN_DELAY_LINE_H
#define FAST_CONVOLUTION_FREQUENCY_DOMAIN_DELAY_LINE_H

#pragma once
#include <vector>
#include <complex>
class FrequencyDomainDelayLine {
public:
    FrequencyDomainDelayLine(int delay_line_size,
                             int freq_size)
        : freqs(delay_line_size) {

        for (auto &f: freqs) {
            f.resize(freq_size, {0.0f, 0.0f});
        }
    }

    int getNextWriteIndex() {
        int next = write_index_++;
        if (write_index_ >= freqs.size()) {
            write_index_ = 0;
        }
        return next;
    }

    std::vector<std::vector<std::complex<float>>> freqs;

private:
    int write_index_{0};
};


#endif//FAST_CONVOLUTION_FREQUENCY_DOMAIN_DELAY_LINE_H
