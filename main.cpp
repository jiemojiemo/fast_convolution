//
// Created by user on 6/9/22.
//
#include "fft.h"
#include "fft_convolution.h"
#include "native_convolution.h"
#include "overlap_add_convolution.h"
#include <complex>
#include <iostream>
#include <vector>

using namespace std;
int main() {
    vector<float> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> h{1, 2};
    const int conv_output_size = x.size() + h.size() - 1;

    //    {
    //        auto y = nativeConvolution(x, h);
    //        printVector(y);
    //    }
    //
    {
        FFTConvolution fft_conv;
        fft_conv.prepare(x.size(), h.size());
        std::vector<float> y(conv_output_size);

        fft_conv.convolution(x, h, y.data());

        Utilities::printVector(y);
    }

    {
        const int B = 5;
        OverlapAddConvolution overlap_add_conv;
        overlap_add_conv.prepare(B, h.size());
        std::vector<float> y(conv_output_size);

        overlap_add_conv.convolution(x, h, y.data());

        Utilities::printVector(y);
    }

    return 0;
}