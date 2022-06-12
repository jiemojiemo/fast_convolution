//
// Created by user on 6/11/22.
//
#include "fast_convolution/fft.h"
#include "fast_convolution/utilities.h"
#include <complex>

#include <gmock/gmock.h>

using namespace testing;
using namespace std;

using frequncyArray = std::vector<std::complex<float>>;

class RealtimeUniformlyPartitionedConvolution {
public:
    void prepare(size_t block_size,
                 const float *h, size_t h_size) {
        auto fft_size = Utilities::nextPower2Number(2 * block_size);
        fft_ = std::make_unique<FFT>(fft_size);

        padding_block_h_.resize(fft_size);
        sliding_block_.resize(2 * block_size);

        num_sub_filter_ = (h_size + block_size - 1) / block_size;
        sub_filters_fft_.resize(num_sub_filter_);
        for (auto &f: sub_filters_fft_) {
            f.resize(fft_size / 2 + 1, {0.0f, 0.0f});
        }



        precalcSubFiltersFFT(block_size, h, h_size);
    }

    const std::vector<frequncyArray> &getSubFiltersFFT() const {
        return sub_filters_fft_;
    }

    const FFT *getFFT() const {
        return fft_.get();
    }

    const std::vector<float> &getSlidingBlock() {
        return sliding_block_;
    }

private:
    void precalcSubFiltersFFT(size_t block_size, const float *h, size_t h_size) {
        for (int i = 0; i < num_sub_filter_; ++i) {
            std::fill(padding_block_h_.begin(), padding_block_h_.end(), 0.0f);
            std::copy_n(h + i * block_size, block_size, padding_block_h_.data());
            fft_->forward(padding_block_h_.data(), (kiss_fft_cpx *) sub_filters_fft_[i].data());
        }
    }
    std::vector<frequncyArray> sub_filters_fft_;
    std::unique_ptr<FFT> fft_;
    std::vector<float> padding_block_h_;
    std::vector<float> sliding_block_;
    size_t num_sub_filter_{0};
};

class ARealtimeUniformlyPartitionedConv : public Test {
public:
    vector<float> h{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11};
    RealtimeUniformlyPartitionedConvolution conv;
    size_t B = 4;
};

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
}

TEST_F(ARealtimeUniformlyPartitionedConv, PrepareAllocateSlidingBlockBuffer) {
    conv.prepare(B, h.data(), h.size());

    ASSERT_THAT(conv.getSlidingBlock().size(), Eq(2 * B));
}