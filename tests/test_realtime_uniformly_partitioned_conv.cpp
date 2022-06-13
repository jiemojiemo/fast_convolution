//
// Created by user on 6/13/22.
//
#include "fast_convolution/realtime_uniformly_partitioned_conv.h"
#include <gmock/gmock.h>
#include <vector>
using namespace std;
using namespace testing;

class ARealtimeUniformlyPartitionedConv : public Test {
public:
    vector<float> h{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11};
    RealtimeUniformlyPartitionedConv conv;
    size_t B = 4;
};

TEST_F(ARealtimeUniformlyPartitionedConv, CanPrepareWithFilterData) {
    conv.prepare(B, h.data(), h.size());
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareAlsoMakeUtitliesPrepared) {
    conv.prepare(B, h.data(), h.size());

    ASSERT_THAT(conv.getConvUtilities().getFFT(), NotNull());
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareAllocateInternalBuffers) {
    conv.prepare(B, h.data(), h.size());

    ASSERT_THAT(conv.getInputBuffer().size(), Eq(B));
    ASSERT_THAT(conv.getOutputBuffer().size(), Eq(B));
}

TEST_F(ARealtimeUniformlyPartitionedConv, CanProcessSingleSample) {
    conv.prepare(B, h.data(), h.size());

    conv.processSample(0.0f);
}

TEST_F(ARealtimeUniformlyPartitionedConv, ProcessIncreaseInputWritePointer) {
    conv.prepare(B, h.data(), h.size());

    conv.processSample(0.0f);
    conv.processSample(0.0f);

    ASSERT_THAT(conv.getInputWritePointer(), Eq(2));
}

TEST_F(ARealtimeUniformlyPartitionedConv, InputWritePointerLoopBackIfOutofRange) {
    conv.prepare(B, h.data(), h.size());

    conv.processSample(0.0f);
    conv.processSample(0.0f);
    conv.processSample(0.0f);
    conv.processSample(0.0f);

    ASSERT_THAT(conv.getInputWritePointer(), Eq(0));
}

TEST_F(ARealtimeUniformlyPartitionedConv, CopyInToInputBufferWhenProcess) {
    conv.prepare(B, h.data(), h.size());

    conv.processSample(1.0f);
    conv.processSample(2.0f);

    ASSERT_THAT(conv.getInputBuffer(), Contains(1.0f));
    ASSERT_THAT(conv.getInputBuffer(), Contains(2.0f));
}

TEST_F(ARealtimeUniformlyPartitionedConv, ProcessIncreaseOutputWritePointer) {
    conv.prepare(B, h.data(), h.size());

    conv.processSample(0.0f);
    conv.processSample(0.0f);

    ASSERT_THAT(conv.getOutputReadPointer(), Eq(2));
}

TEST_F(ARealtimeUniformlyPartitionedConv, OutputReadPointerLoopBackIfOutofRange) {
    conv.prepare(B, h.data(), h.size());

    conv.processSample(0.0f);
    conv.processSample(0.0f);
    conv.processSample(0.0f);
    conv.processSample(0.0f);

    ASSERT_THAT(conv.getOutputReadPointer(), Eq(0));
}

TEST_F(ARealtimeUniformlyPartitionedConv, ProcessIncreaseCounter) {
    conv.prepare(B, h.data(), h.size());

    conv.processSample(0.0f);

    ASSERT_THAT(conv.getCounter(), Eq(1));
}

TEST_F(ARealtimeUniformlyPartitionedConv, CounterResetToZeroAfterProcessABlock) {
    conv.prepare(B, h.data(), h.size());

    for (int i = 0; i < B; ++i) {
        conv.processSample(1.0f);
    }

    ASSERT_THAT(conv.getCounter(), Eq(0));
}

TEST_F(ARealtimeUniformlyPartitionedConv, ProcessAsExpected) {
    conv.prepare(B, h.data(), h.size());

    // latency
    for (int i = 0; i < B; ++i) {
        conv.processSample(1.0f), Eq(0.0f);
    }

    ASSERT_THAT(conv.processSample(1.0f), FloatNear(0.0f, 1e-6));
    ASSERT_THAT(conv.processSample(1.0f), FloatNear(1.0f, 1e-6));
    ASSERT_THAT(conv.processSample(1.0f), FloatNear(3.0f, 1e-6));
    ASSERT_THAT(conv.processSample(1.0f), FloatNear(6.0f, 1e-6));
}