//
// Created by user on 6/13/22.
//

#ifndef FAST_CONVOLUTION_REALTIME_UNIFORMLY_PARTITIONED_CONV_UTILITIES_H
#define FAST_CONVOLUTION_REALTIME_UNIFORMLY_PARTITIONED_CONV_UTILITIES_H

#pragma once
#include "fast_convolution/fft.h"
#include "fast_convolution/frequency_domain_delay_line.h"
#include "fast_convolution/utilities.h"
#include <complex>
class RealtimeUniformlyPartitionedConvolutionUtilities {
public:
    void prepare(size_t block_size,
                 const float *h, size_t h_size) {
        block_size_ = block_size;
        fft_size_ = Utilities::nextPower2Number(2 * block_size);
        half_fft_size_ = fft_size_ / 2 + 1;

        fft_ = std::make_unique<FFT>(fft_size_);

        time_domain_buffer_.resize(fft_size_);
        sliding_block_.resize(2 * block_size);

        num_sub_filter_ = (h_size + block_size - 1) / block_size;
        sub_filters_fft_.resize(num_sub_filter_);
        for (auto &f: sub_filters_fft_) {
            f.resize(half_fft_size_, {0.0f, 0.0f});
        }
        precalcSubFiltersFFT(block_size, h, h_size);

        freqs_delay_line_.resize(num_sub_filter_, half_fft_size_);

        fft_summing_.resize(half_fft_size_, {0.0f, 0.0f});
    }

    void processBlock(const float *block_data,
                      float *output,
                      int block_size) {
        assert(block_size == block_size_);
        shiftAndCopyToSlidingBlock(block_data, block_size);

        // block fft
        int next_write_index = freqs_delay_line_.getNextWriteIndex();
        auto &next_writable_freqs = freqs_delay_line_.freqs_array[next_write_index];
        fft_->forward(sliding_block_.data(), (kiss_fft_cpx *) (next_writable_freqs.data()));

        // summing
        std::fill(fft_summing_.begin(), fft_summing_.end(), 0);

        for (int i = 0; i < num_sub_filter_; ++i) {
            const Frequencies &current_sub_filter = sub_filters_fft_[i];
            const Frequencies &delayed_freq = freqs_delay_line_.getDelayedFrequencies(i);

            for (int j = 0; j < half_fft_size_; ++j) {
                fft_summing_[j] += (current_sub_filter[j] * delayed_freq[j]);
            }
        }

        // ifft
        fft_->inverse((kiss_fft_cpx *) fft_summing_.data(), time_domain_buffer_.data());
        for (auto &item: time_domain_buffer_) {
            item /= fft_size_;
        }

        // copy last B samples to output
        std::copy_n(time_domain_buffer_.end() - block_size, block_size, output);
    }

    const std::vector<Frequencies> &getSubFiltersFFT() const {
        return sub_filters_fft_;
    }

    const FrequencyDomainDelayLine &getFrequencyDomainDelayLine() const {
        return freqs_delay_line_;
    }

    const FFT *getFFT() const {
        return fft_.get();
    }

    const std::vector<float> &getSlidingBlock() {
        return sliding_block_;
    }

    size_t getBlockSize() const {
        return block_size_;
    }

private:
    void precalcSubFiltersFFT(size_t block_size, const float *h, size_t h_size) {
        for (int i = 0; i < num_sub_filter_; ++i) {
            std::fill(time_domain_buffer_.begin(), time_domain_buffer_.end(), 0.0f);
            int begin_index = i * block_size;
            int end_index = begin_index + block_size;
            if (end_index > h_size) {
                end_index = h_size;
            }
            int actual_size = end_index - begin_index;

            std::copy_n(h + i * block_size, actual_size, time_domain_buffer_.data());
            fft_->forward(time_domain_buffer_.data(), (kiss_fft_cpx *) sub_filters_fft_[i].data());
        }
    }

    void shiftAndCopyToSlidingBlock(const float *block_data,
                                    int block_size) {
        Utilities::leftShift(sliding_block_, block_size, 0.0f);
        std::copy_n(block_data, block_size, sliding_block_.data() + block_size);
    }

    std::vector<Frequencies> sub_filters_fft_;
    std::unique_ptr<FFT> fft_;
    std::vector<float> time_domain_buffer_;
    std::vector<float> sliding_block_;
    std::vector<std::complex<float>> fft_summing_;
    FrequencyDomainDelayLine freqs_delay_line_;
    size_t num_sub_filter_{0};
    size_t block_size_{0};
    size_t fft_size_{0};
    size_t half_fft_size_{0};
};


#endif//FAST_CONVOLUTION_REALTIME_UNIFORMLY_PARTITIONED_CONV_UTILITIES_H
