# Fast Convolution

Python and C++ Implementation of fast convolution algorithm:

+ FFT convolution
+ Overlap-Add
+ Overlap-Save
+ Realtime Uniformly Partitioned Convolution

# How to build and run example

Changes the `input_file` and `impulse_file` in `example/realtime_conv.cpp`

```shell
git clone git@github.com:jiemojiemo/fast_convolution.git
cd fast_convolution
git submodule update --init --recursive
cmake -S . -B build -DKISSFFT_TEST=OFF -DKISSFFT_TOOLS=OFF
cmake --build build --target realtime_conv

./build/example/realtime_cov
```