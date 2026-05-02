#include "audio_io.h"

#include <sndfile.h>

#include <cstdio>

std::vector<float> audio_load_mono(const std::string& path, int target_sr) {
    SF_INFO info = {};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf) {
        fprintf(stderr, "Failed to open audio: %s (%s)\n",
                path.c_str(), sf_strerror(nullptr));
        return {};
    }

    // Read all frames as float
    std::vector<float> raw(info.frames * info.channels);
    sf_count_t read = sf_readf_float(sf, raw.data(), info.frames);
    sf_close(sf);

    if (read != info.frames) {
        fprintf(stderr, "Short read: %s (%lld of %lld frames)\n",
                path.c_str(), (long long)read, (long long)info.frames);
    }

    // Mix to mono if needed
    std::vector<float> mono(read);
    if (info.channels == 1) {
        mono.assign(raw.begin(), raw.begin() + read);
    } else {
        for (sf_count_t i = 0; i < read; i++) {
            float sum = 0.0f;
            for (int c = 0; c < info.channels; c++)
                sum += raw[i * info.channels + c];
            mono[i] = sum / info.channels;
        }
    }

    // Simple integer-ratio resampling (decimation) if needed
    if (info.samplerate != target_sr) {
        if (info.samplerate % target_sr != 0) {
            fprintf(stderr, "Cannot resample %d -> %d (non-integer ratio)\n",
                    info.samplerate, target_sr);
            return {};
        }
        int ratio = info.samplerate / target_sr;
        std::vector<float> resampled(mono.size() / ratio);
        for (size_t i = 0; i < resampled.size(); i++)
            resampled[i] = mono[i * ratio];
        return resampled;
    }
    return mono;
}

bool audio_save_wav(const std::string& path, const float* data, int64_t n,
                    int sample_rate) {
    SF_INFO info = {};
    info.samplerate = sample_rate;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE* sf = sf_open(path.c_str(), SFM_WRITE, &info);
    if (!sf) {
        fprintf(stderr, "Failed to write audio: %s (%s)\n",
                path.c_str(), sf_strerror(nullptr));
        return false;
    }
    sf_count_t wrote = sf_writef_float(sf, data, n);
    sf_close(sf);
    return wrote == n;
}
