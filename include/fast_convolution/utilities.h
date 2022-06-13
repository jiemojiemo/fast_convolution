//
// Created by user on 6/11/22.
//

#ifndef FAST_CONVOLUTION_UTILITIES_H
#define FAST_CONVOLUTION_UTILITIES_H

#pragma once
#include <cmath>
#include <iostream>
#include <vector>
class Utilities {
public:
    static size_t nextPower2Number(size_t k) {
        if (k == 0) return 2;

        return 1 << int(log2(k - 1) + 1);
    }

    template<class T>
    static void printVector(const std::vector<T> &x) {
        for (const auto item: x) {
            std::cout << item << ",";
        }
        std::cout << std::endl;
    }

    static void leftShift(std::vector<float> &x, int shift, float pad_value) {
        shift = shift % x.size();

        std::copy(x.begin() + shift, x.end(), x.begin());

        std::fill(x.end() - shift, x.end(), pad_value);
    }
};

#endif//FAST_CONVOLUTION_UTILITIES_H
