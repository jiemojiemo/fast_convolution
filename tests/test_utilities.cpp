//
// Created by user on 6/11/22.
//
#include "fast_convolution/utilities.h"
#include <gmock/gmock.h>

using namespace testing;
using namespace std;

class AUtilities : public Test {
public:
};

TEST_F(AUtilities, CanGetNextPowerOf2Number) {
    ASSERT_THAT(Utilities::nextPower2Number(10), Eq(16));
}

TEST_F(AUtilities, NextPowerOf2NumberReturns2IfInputIsZero) {
    ASSERT_THAT(Utilities::nextPower2Number(0), Eq(2));
}

TEST_F(AUtilities, CanLeftShiftVector)
{
    vector<float> x{1,2,3,4};

    Utilities::leftShift(x, 2, 0);

    vector<float> expected{3, 4, 0, 0};
    ASSERT_THAT(x, ContainerEq(expected));
}