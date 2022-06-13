//
// Created by user on 6/11/22.
//

#ifndef FAST_CONVOLUTION_FFT_CONVOLUTION_H
#define FAST_CONVOLUTION_FFT_CONVOLUTION_H

#pragma once
#include "fast_convolution/fft.h"
#include "fast_convolution/utilities.h"
#include "kiss_fftr.h"
#include <complex>
#include <memory>
#include <vector>
class FFTConvolution {
public:
    void prepare(int x_length,
                 int h_length) {
        K_ = x_length + h_length - 1;
        prepare(K_);
    }

    void prepare(int K) {
        K_ = K;
        fft_size_ = Utilities::nextPower2Number(K_);
        fft_ = std::make_unique<FFT>(fft_size_);

        // allocate buffers
        padding_x_.resize(fft_size_, 0.0f);
        padding_h_.resize(fft_size_, 0.0f);
        y_.resize(fft_size_, 0.0f);

        X_.resize(fft_size_ / 2 + 1, 0.0f);
        H_.resize(fft_size_ / 2 + 1, 0.0f);
    }

    /**
     * fft convolution
     * @param x The input signal with size M
     * @param h The filter signal with size N
     * @param y The output buffer, make sure the array size is at least (M + N - 1)
     */
    void convolution(const std::vector<float> &x,
                     const std::vector<float> &h,
                     float *y) {

        //        assert((x.size() + h.size() - 1) == K_);

        zeroPadding(x, h);

        // fft multiple
        fft_->forward(padding_x_.data(), (kiss_fft_cpx *) (X_.data()));
        fft_->forward(padding_h_.data(), (kiss_fft_cpx *) (H_.data()));
        for (int i = 0; i < X_.size(); ++i) {
            X_[i] *= H_[i];
        }

        // ifft & normalize the output
        fft_->inverse((kiss_fft_cpx *) (X_.data()), y_.data());
        for (int i = 0; i < K_; ++i) {
            y_[i] /= fft_size_;
        }

        // copy ifft results to y
        std::copy_n(y_.data(), K_, y);
    }

private:
    void zeroPadding(const std::vector<float> &x,
                     const std::vector<float> &h) {
        std::fill(padding_x_.begin(), padding_x_.end(), 0.0f);
        std::fill(padding_h_.begin(), padding_h_.end(), 0.0f);

        std::copy_n(x.begin(), x.size(), padding_x_.data());
        std::copy_n(h.begin(), h.size(), padding_h_.data());
    }

    size_t K_ = 0;
    size_t fft_size_{0};

    std::vector<float> padding_x_;
    std::vector<float> padding_h_;
    std::vector<float> y_;
    std::vector<std::complex<float>> X_;
    std::vector<std::complex<float>> H_;

    std::unique_ptr<FFT> fft_;
};


#endif//FAST_CONVOLUTION_FFT_CONVOLUTION_H
