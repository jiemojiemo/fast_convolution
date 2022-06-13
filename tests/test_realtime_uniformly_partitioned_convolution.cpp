//
// Created by user on 6/11/22.
//

#include "fast_convolution/realtime_uniformly_partitioned_conv_utilities.h"
#include <gmock/gmock.h>

using namespace testing;
using namespace std;
class ARealtimeUniformlyPartitionedConv : public Test {
public:
    vector<float> h{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11};
    RealtimeUniformlyPartitionedConvolutionUtilities conv;
    size_t B = 4;
};

MATCHER_P(FloatNearPointwise, tol, "Out of range") {
    return (std::get<0>(arg) > std::get<1>(arg) - tol && std::get<0>(arg) < std::get<1>(arg) + tol);
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareWithFilter) {
    conv.prepare(B, h.data(), h.size());
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareAllocateSubFilterBuffers) {
    conv.prepare(B, h.data(), h.size());

    int expected = ceil(1.0 * h.size() / B);
    ASSERT_THAT(conv.getSubFiltersFFT().size(), expected);
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareInitFFTConfig) {
    conv.prepare(B, h.data(), h.size());

    auto expected = Utilities::nextPower2Number(2 * B);
    ASSERT_THAT(conv.getFFT(), NotNull());
    ASSERT_THAT(conv.getFFT()->getFFTSize(), Eq(expected));
}

TEST_F(ARealtimeUniformlyPartitionedConv, PreparePrecalcSubFilterFFT) {
    conv.prepare(B, h.data(), h.size());

    int expected_size = (h.size() + B - 1) / B;
    ASSERT_THAT(conv.getSubFiltersFFT().size(), Eq(expected_size));
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareInitFrequencyDelayLine) {
    conv.prepare(B, h.data(), h.size());

    int expected_size = (h.size() + B - 1) / B;
    ASSERT_THAT(conv.getFrequencyDomainDelayLine().freqs_array.size(), Eq(expected_size));
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareAllocateSlidingBlockBuffer) {
    conv.prepare(B, h.data(), h.size());

    ASSERT_THAT(conv.getSlidingBlock().size(), Eq(2 * B));
}

TEST_F(ARealtimeUniformlyPartitionedConv, ProcessBlockWillShiftSlindingBlock) {
    conv.prepare(B, h.data(), h.size());
    std::vector<float> block(B, 1.0f);

    conv.processBlock(block.data(), block.data(), block.size());

    std::vector<float> expected = {0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_THAT(conv.getSlidingBlock(), ContainerEq(expected));
}

TEST_F(ARealtimeUniformlyPartitionedConv, ProcessBlockOutputCorrectly) {
    conv.prepare(B, h.data(), h.size());
    std::vector<float> block(B, 1.0f);

    conv.processBlock(block.data(), block.data(), block.size());

    std::vector<float> expected = {0, 1, 3, 6};
    ASSERT_THAT(block, Pointwise(FloatNearPointwise(1e-6), expected));
}