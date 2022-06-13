//
// Created by user on 6/13/22.
//
#include "fast_convolution/fft_convolution.h"
#include "fast_convolution/native_convolution.h"
#include "fast_convolution/overlap_add_convolution.h"
#include "fast_convolution/overlap_save_convoluton.h"
#include "fast_convolution/realtime_uniformly_partitioned_conv.h"
#include "fast_convolution/utilities.h"
#include <vector>
using namespace std;

int main() {
    vector<float> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> h{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const int conv_output_size = x.size() + h.size() - 1;

    {
        auto y = nativeConvolution(x, h);
        Utilities::printVector(y);
    }

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

    {
        const int B = 4;
        OverlapSaveConvolution overlap_save_conv;
        overlap_save_conv.prepare(B, h.size());
        std::vector<float> y(conv_output_size);

        overlap_save_conv.convolution(x, h, y.data());

        Utilities::printVector(y);
    }

    {
        const int B = 4;
        RealtimeUniformlyPartitionedConv uniform_conv;
        uniform_conv.prepare(B, h.data(), h.size());

        for (int i = 0; i < conv_output_size + B; ++i) {
            float in = (i >= x.size()) ? (0) : (x[i]);
            float output = uniform_conv.processSample(in);
            cout << output << ",";
        }
        cout << endl;
    }

    return 0;
}