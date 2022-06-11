//
// Created by user on 6/11/22.
//

#ifndef FAST_CONVOLUTION_UTILITIES_H
#define FAST_CONVOLUTION_UTILITIES_H

#pragma once
#include <vector>
#include <iostream>
class Utilities {
public:
    static size_t nextPower2Number(size_t k) {
        return 1 << int(log2(k - 1) + 1);
    }

    static void printVector(const std::vector<float> &x) {
        for (const auto item: x) {
            std::cout << item << ",";
        }
        std::cout << std::endl;
    }
};

#endif//FAST_CONVOLUTION_UTILITIES_H
