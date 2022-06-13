//
// Created by user on 6/13/22.
//

#ifndef FAST_CONVOLUTION_REALTIME_UNIFORMLY_PARTITIONED_CONV_H
#define FAST_CONVOLUTION_REALTIME_UNIFORMLY_PARTITIONED_CONV_H

#pragma once
#include "fast_convolution/realtime_uniformly_partitioned_conv_utilities.h"
class RealtimeUniformlyPartitionedConv {
public:
    void prepare(size_t block_size,
                 const float *h, size_t h_size) {

        utilities_.prepare(block_size, h, h_size);
        input_buffer_.resize(block_size, 0.0f);
        output_buffer_.resize(block_size, 0.0f);

        input_write_pointer_ = 0;
        output_read_pointer_ = 0;
        block_size_ = block_size;
    }

    float processSample(float in) {
        input_buffer_[input_write_pointer_] = in;
        ++input_write_pointer_;
        if (input_write_pointer_ >= input_buffer_.size()) {
            input_write_pointer_ = 0;
        }

        float output = output_buffer_[output_read_pointer_];
        output_read_pointer_++;
        if (output_read_pointer_ >= output_buffer_.size()) {
            output_read_pointer_ = 0;
        }

        if (++counter_ >= block_size_) {
            utilities_.processBlock(input_buffer_.data(),
                                    output_buffer_.data(),
                                    block_size_);

            counter_ = 0;
        }

        return output;
    }

    const RealtimeUniformlyPartitionedConvolutionUtilities &getConvUtilities() const {
        return utilities_;
    }

    const std::vector<float> &getInputBuffer() const {
        return input_buffer_;
    }

    const std::vector<float> &getOutputBuffer() const {
        return output_buffer_;
    }

    size_t getInputWritePointer() const {
        return input_write_pointer_;
    }

    size_t getOutputReadPointer() const {
        return output_read_pointer_;
    }

    size_t getCounter() const {
        return counter_;
    }

private:
    RealtimeUniformlyPartitionedConvolutionUtilities utilities_;
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    size_t input_write_pointer_{0};
    size_t output_read_pointer_{0};
    size_t counter_{0};
    size_t block_size_{0};
};


#endif//FAST_CONVOLUTION_REALTIME_UNIFORMLY_PARTITIONED_CONV_H
