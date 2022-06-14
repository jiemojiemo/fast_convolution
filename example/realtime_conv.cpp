//
// Created by user on 6/13/22.
//
#include "fast_convolution/realtime_uniformly_partitioned_conv.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "portaudio.h"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

typedef struct
{
    RealtimeUniformlyPartitionedConv conv;
    std::vector<float> input_samples;
    std::vector<float> impulse_samples;
    int input_read_pointer;
    char message[20];
} paTestData;

std::vector<float> loadAudioFile(const std::string &file_path,
                                 int *sr) {
    drwav wav;
    if (!drwav_init_file(&wav, file_path.c_str(), NULL)) {
        cerr << "can not open " << file_path << endl;
        exit(-1);
    }

    if (wav.channels != 1) {
        cerr << "only support mono file" << endl;
        exit(-1);
    }

    *sr = wav.sampleRate;

    std::vector<float> samples(wav.channels * wav.totalPCMFrameCount);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, samples.data());

    drwav_uninit(&wav);

    return samples;
}

static int patestCallback(const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo *timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData) {
    paTestData *data = (paTestData *) userData;
    float *out = (float *) outputBuffer;
    unsigned long i;

    (void) timeInfo; /* Prevent unused variable warnings. */
    (void) statusFlags;
    (void) inputBuffer;

    for (i = 0; i < framesPerBuffer; i++) {
        float in = data->input_samples[data->input_read_pointer++];
        float o = data->conv.processSample(in);
        out[i] = o;

        if (data->input_read_pointer >= data->input_samples.size()) {
            return paComplete;
        }
    }

    return paContinue;
}

static void streamFinished(void *userData) {
    paTestData *data = (paTestData *) userData;
    printf("Stream Completed: %s\n", data->message);
}

int main(int argc, char *argv[]) {

    const auto input_file = "/Users/user/Downloads/voice_mono.wav";
    const auto impulse_file = "/Users/user/Downloads/On a Star_01.wav";
    int input_sr;
    int impulse_sr;
    const int kNumSeconds = 10;
    const int kBlockSize = 256;

    paTestData data;
    data.input_samples = loadAudioFile(input_file, &input_sr);
    data.impulse_samples = loadAudioFile(impulse_file, &impulse_sr);
    assert(input_sr == impulse_sr);
    data.conv.prepare(kBlockSize, data.impulse_samples.data(), data.impulse_samples.size());

    PaStreamParameters outputParameters;
    PaStream *stream;
    PaError err = Pa_Initialize();
    if (err != paNoError) goto error;

    outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
    if (outputParameters.device == paNoDevice) {
        fprintf(stderr, "Error: No default output device.\n");
        goto error;
    }
    outputParameters.channelCount = 1;         /* mono output */
    outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;

    err = Pa_OpenStream(
            &stream,
            NULL, /* no input */
            &outputParameters,
            input_sr,
            1024,
            paClipOff, /* we won't output out of range samples so don't bother clipping them */
            patestCallback,
            &data);
    if (err != paNoError) goto error;


    sprintf(data.message, "No Message");
    err = Pa_SetStreamFinishedCallback(stream, &streamFinished);
    if (err != paNoError) goto error;

    err = Pa_StartStream(stream);
    if (err != paNoError) goto error;

    printf("Play for %d seconds.\n", kNumSeconds);
    Pa_Sleep(kNumSeconds * 1000);

    err = Pa_StopStream(stream);
    if (err != paNoError) goto error;

    err = Pa_CloseStream(stream);
    if (err != paNoError) goto error;

    Pa_Terminate();
    printf("Test finished.\n");
    return 0;

error:
    Pa_Terminate();
    fprintf(stderr, "An error occurred while using the portaudio stream\n");
    fprintf(stderr, "Error number: %d\n", err);
    fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
    return err;
}