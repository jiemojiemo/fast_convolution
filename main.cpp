//
// Created by user on 6/9/22.
//
#include "native_convolution.h"

#include <iostream>
#include <vector>

void printVector(const std::vector<float> &x) {
    for (const auto item: x) {
        std::cout << item << ",";
    }
    std::cout << std::endl;
}


int main() {
    std::vector<float> x{0, 1, 2, 3};
    std::vector<float> h{1, 2};

    auto y = nativeConvolution(x, h);
    printVector(y);

    return 0;
}