#pragma once

// Audio file I/O via libsndfile. Used by the CLI / bench / test_streaming /
// eval_graph binaries; the public C API in `localvqe_api.h` operates on
// raw float / int16 buffers.

#include <cstdint>
#include <string>
#include <vector>

/// Load an audio file (FLAC, WAV, etc.) as mono float32 [-1,1].
/// Resamples to target_sr if the file's sample rate differs (simple decimation).
/// Returns empty vector on failure.
std::vector<float> audio_load_mono(const std::string& path,
                                    int target_sr = 16000);

/// Save mono float32 PCM as 16-bit WAV. Returns true on success.
bool audio_save_wav(const std::string& path, const float* data, int64_t n,
                    int sample_rate = 16000);
