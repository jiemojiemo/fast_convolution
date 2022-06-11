//
// Created by user on 6/11/22.
//

#ifndef FAST_CONVOLUTION_OVERLAP_ADD_CONVOLUTION_H
#define FAST_CONVOLUTION_OVERLAP_ADD_CONVOLUTION_H

#pragma once
#include "fft_convolution.h"
#include <cmath>
#include <vector>
class OverlapAddConvolution {
public:
    void prepare(int block_length,
                 int h_length) {
        block_length_ = block_length;
        fft_conv_.prepare(block_length, h_length);
        unwrap_block_x_.resize(block_length, 0.0f);

        const int block_conv_output_size = block_length + h_length - 1;
        block_y_.resize(block_conv_output_size, 0.0f);
    }
    void convolution(const std::vector<float> &x,
                     const std::vector<float> &h,
                     float *y) {
        const int num_block = ceil(1.0 * x.size() / block_length_);
        const int conv_output_size = x.size() + h.size() - 1;

        // read block
        for (int i = 0; i < num_block; ++i) {
            unwrap(x, i);

            fft_conv_.convolution(unwrap_block_x_, h, block_y_.data());
            
            // overlap add
            overlap_add(i, conv_output_size, y);
        }
    }

private:
    void unwrap(const std::vector<float> &x,
                int block_index) {
        std::fill(unwrap_block_x_.begin(), unwrap_block_x_.end(), 0.0f);

        int block_begin = block_index * block_length_;
        int block_end = block_begin + block_length_;
        if (block_end > x.size()) {
            block_end = x.size();
        }
        int current_block_size = block_end - block_begin;
        std::copy_n(x.data() + block_begin, current_block_size, unwrap_block_x_.data());
    }

    void overlap_add(int block_index,
                     int conv_output_size,
                     float *y) {
        int block_begin = block_index * block_length_;
        int block_end = block_begin + block_y_.size();
        block_end = std::min(conv_output_size, block_end);
        int jj = 0;
        for (int j = block_begin; j < block_end; ++j) {
            y[j] += block_y_[jj++];
        }
    }

private:
    int block_length_{0};
    FFTConvolution fft_conv_;
    std::vector<float> unwrap_block_x_;
    std::vector<float> block_y_;
};


#endif//FAST_CONVOLUTION_OVERLAP_ADD_CONVOLUTION_H
