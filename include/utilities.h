//
// Created by user on 6/11/22.
//

#ifndef FAST_CONVOLUTION_UTILITIES_H
#define FAST_CONVOLUTION_UTILITIES_H

#pragma once

class Utilities {
public:
    static size_t nextPower2Number(size_t k) {
        return 1 << int(log2(k - 1) + 1);
    }
};

#endif//FAST_CONVOLUTION_UTILITIES_H
