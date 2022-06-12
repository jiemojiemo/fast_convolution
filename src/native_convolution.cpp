//
// Created by user on 6/11/22.
//
#include "fast_convolution/native_convolution.h"

namespace {
    float getFlip(const std::vector<float> &h,
                  int i) {
        return *(h.rbegin() + i);
    }

}// namespace

std::vector<float> nativeConvolution(
        const std::vector<float> &x,
        const std::vector<float> &h) {
    const auto M = x.size();
    const auto N = h.size();

    std::vector<float> zero_pad_x(M + 2 * (N - 1));
    std::copy_n(x.data(), x.size(), zero_pad_x.data() + (N - 1));

    const auto output_size = (M + N - 1);
    std::vector<float> y(output_size, 0.0f);

    for (int i = 0; i < output_size; ++i) {
        float output = 0.0f;
        for (int j = 0; j < N; ++j) {
            int x_index = j + i;
            output += zero_pad_x[x_index] * getFlip(h, j);
        }
        y[i] = output;
    }

    return y;
}