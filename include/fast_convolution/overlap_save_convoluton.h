//
// Created by user on 6/11/22.
//

#ifndef FAST_CONVOLUTION_OVERLAP_SAVE_CONVOLUTON_H
#define FAST_CONVOLUTION_OVERLAP_SAVE_CONVOLUTON_H

#pragma once
#include "utilities.h"
#include "fft.h"
#include <complex>
#include <vector>
class OverlapSaveConvolution {
public:
    void prepare(int block_length,
                 int h_length) {
        block_length_ = block_length;
        const size_t K = block_length + h_length - 1;
        fft_size_ = Utilities::nextPower2Number(K);

        fft_ = std::make_unique<FFT>(fft_size_);

        padding_h_.resize(fft_size_, 0.0f);
        sliding_x_.resize(fft_size_, 0.0f);
        block_y_.resize(fft_size_, 0.0f);

        X_.resize(fft_size_ / 2 + 1, 0.0f);
        H_.resize(fft_size_ / 2 + 1, 0.0f);
    }
    void convolution(const std::vector<float> &x,
                     const std::vector<float> &h,
                     float *y) {

        const int K = x.size() + h.size() - 1;
        const int num_block = ceil(1.0 * x.size() / block_length_) + ceil(1.0 * K / block_length_) - 1;
        hZeroPadding(h);

        int k = 0;
        for (int i = 0; i < num_block; ++i) {
            unwrap(x, i);

            blockFFTConv();

            // copy block result to output
            // the last B size value is valid
            for (int j = block_y_.size() - block_length_; j < block_y_.size() && k < K; ++j) {
                y[k++] = block_y_[j];
            }
        }
    }

private:
    void unwrap(const std::vector<float> &x,
                int block_index) {

        // shift block_length
        Utilities::leftShift(sliding_x_, block_length_, 0.0f);

        int block_begin = block_index * block_length_;
        int block_end = block_begin + block_length_;
        if (block_end > x.size()) {
            block_end = x.size();
        }
        if (block_end > block_begin) {
            int current_block_size = block_end - block_begin;
            std::copy_n(x.data() + block_begin, current_block_size, sliding_x_.end() - block_length_);
        }
    }

    void hZeroPadding(const std::vector<float> &h) {
        std::fill(padding_h_.begin(), padding_h_.end(), 0.0f);
        std::copy_n(h.begin(), h.size(), padding_h_.data());
    }

    void blockFFTConv() {
        fft_->forward(sliding_x_.data(), (kiss_fft_cpx *) (X_.data()));
        fft_->forward(padding_h_.data(), (kiss_fft_cpx *) (H_.data()));
        for (int j = 0; j < X_.size(); ++j) {
            X_[j] *= H_[j];
        }

        fft_->inverse((kiss_fft_cpx *) (X_.data()), block_y_.data());
        for (int j = 0; j < block_y_.size(); ++j) {
            block_y_[j] /= fft_size_;
        }
    }

    int block_length_{0};
    size_t fft_size_;
    std::vector<float> sliding_x_;
    std::vector<float> padding_h_;
    std::vector<float> block_y_;
    std::unique_ptr<FFT> fft_{nullptr};
    std::vector<std::complex<float>> X_;
    std::vector<std::complex<float>> H_;
};


#endif//FAST_CONVOLUTION_OVERLAP_SAVE_CONVOLUTON_H
