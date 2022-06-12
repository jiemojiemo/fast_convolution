//
// Created by user on 6/11/22.
//

#ifndef FAST_CONVOLUTION_FFT_H
#define FAST_CONVOLUTION_FFT_H

#pragma once
#include "kiss_fftr.h"
class FFT {
public:
    FFT(int fft_size)
        : fft_size_(fft_size)
    {
        fft_forward_ = kiss_fftr_alloc(fft_size, 0, NULL, NULL);
        fft_inverse_ = kiss_fftr_alloc(fft_size, 1, NULL, NULL);
    }

    ~FFT() {
        kiss_fftr_free(fft_forward_);
        kiss_fftr_free(fft_inverse_);
    }

    size_t getFFTSize() const{
        return fft_size_;
    }

    void forward(const float *time_data,
                 kiss_fft_cpx *freq_data) {
        kiss_fftr(fft_forward_, time_data, freq_data);
    }

    void inverse(const kiss_fft_cpx *freq_data,
                 float *time_data) {
        kiss_fftri(fft_inverse_, freq_data, time_data);
    }

private:
    kiss_fftr_cfg fft_forward_{NULL};
    kiss_fftr_cfg fft_inverse_{NULL};
    size_t fft_size_{0};
};

#endif//FAST_CONVOLUTION_FFT_H
