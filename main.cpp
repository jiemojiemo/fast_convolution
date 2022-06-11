//
// Created by user on 6/9/22.
//
#include "fft.h"
#include "fft_convolution.h"
#include "native_convolution.h"
#include <complex>
#include <iostream>
#include <vector>


void printVector(const std::vector<float> &x) {
    for (const auto item: x) {
        std::cout << item << ",";
    }
    std::cout << std::endl;
}


using namespace std;
int main() {
    vector<float> x{0, 1, 2, 3};
    vector<float> h{1, 2};

    {
        auto y = nativeConvolution(x, h);
        printVector(y);
    }

    {
        FFTConvolution fft_conv;
        fft_conv.prepare(4, 2);
        std::vector<float> y(5);

        fft_conv.convolution(x, h, y.data());

        printVector(y);
    }
    return 0;
}