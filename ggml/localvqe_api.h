#ifndef LOCALVQE_API_H
#define LOCALVQE_API_H

/**
 * LocalVQE C API — purego-compatible shared library interface.
 *
 * All functions use simple C types (no structs, no variadic args)
 * for compatibility with Go's purego FFI.
 *
 * Typical usage:
 *   uintptr_t ctx = localvqe_new("model.gguf");
 *   if (!ctx) { handle error }
 *   int ret = localvqe_process_f32(ctx, mic, ref, n_samples, out);
 *   localvqe_free(ctx);
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef LOCALVQE_BUILD
    #define LOCALVQE_API __declspec(dllexport)
  #else
    #define LOCALVQE_API __declspec(dllimport)
  #endif
#else
  #define LOCALVQE_API __attribute__((visibility("default")))
#endif

/**
 * Create a new LocalVQE context by loading a GGUF model file.
 * Defaults to the CPU backend, device 0. Returns an opaque handle,
 * or 0 on failure.
 */
LOCALVQE_API uintptr_t localvqe_new(const char* model_path);

/**
 * Create a new context with explicit backend + device selection.
 *
 * backend_name:  matches ggml_backend_reg_name (e.g. "CPU", "Vulkan", "CUDA").
 * device_index:  index into the chosen backend's device list (0-based).
 *
 * Use localvqe_list_devices() to see the available choices.
 * Returns an opaque handle, or 0 on failure.
 */
LOCALVQE_API uintptr_t localvqe_new_ex(const char* model_path,
                                       const char* backend_name,
                                       int device_index);

/**
 * Print every registered backend + device to stderr. No model required.
 * Useful for telling the user what to pass to localvqe_new_ex().
 */
LOCALVQE_API void localvqe_list_devices(void);

/**
 * Print memory budget + graph op-type histogram for the loaded model.
 * Diagnostic only; cheap (no inference). Output goes to stdout.
 */
LOCALVQE_API void localvqe_print_profile(uintptr_t ctx);

/**
 * Free a LocalVQE context and all associated resources.
 */
LOCALVQE_API void localvqe_free(uintptr_t ctx);

/**
 * Process audio through the AEC model (float32 version).
 *
 * mic:       Microphone input (mono, float32, [-1,1] range, 16kHz)
 * ref:       Far-end reference (mono, float32, [-1,1] range, 16kHz)
 * n_samples: Number of samples in both mic and ref (must be >= 512)
 * out:       Pre-allocated output buffer (n_samples floats)
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_process_f32(uintptr_t ctx,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out);

/**
 * Process audio through the AEC model (int16 PCM version).
 *
 * mic:       Microphone input (mono, int16 PCM, 16kHz)
 * ref:       Far-end reference (mono, int16 PCM, 16kHz)
 * n_samples: Number of samples in both mic and ref (must be >= 512)
 * out:       Pre-allocated output buffer (n_samples int16s)
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_process_s16(uintptr_t ctx,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out);

/**
 * Get the last error message, or empty string if no error.
 * The returned pointer is valid until the next API call on this context.
 */
LOCALVQE_API const char* localvqe_last_error(uintptr_t ctx);

/**
 * Get model sample rate (always 16000 currently).
 */
LOCALVQE_API int localvqe_sample_rate(uintptr_t ctx);

/**
 * Get hop length in samples (256).
 */
LOCALVQE_API int localvqe_hop_length(uintptr_t ctx);

/**
 * Get FFT size (512).
 */
LOCALVQE_API int localvqe_fft_size(uintptr_t ctx);

/**
 * Process a single hop of audio through the AEC model (float32 version).
 *
 * mic:         Microphone input (mono, float32, [-1,1], 16kHz)
 * ref:         Far-end reference (mono, float32, [-1,1], 16kHz)
 * hop_samples: Must equal hop_length (256)
 * out:         Pre-allocated output buffer (hop_samples floats)
 *
 * Returns 0 on success. First call outputs zeros (warmup).
 */
LOCALVQE_API int localvqe_process_frame_f32(uintptr_t ctx,
                                           const float* mic, const float* ref,
                                           int hop_samples, float* out);

/**
 * Process a single hop of audio through the AEC model (int16 PCM version).
 *
 * mic:         Microphone input (mono, int16 PCM, 16kHz)
 * ref:         Far-end reference (mono, int16 PCM, 16kHz)
 * hop_samples: Must equal hop_length (256)
 * out:         Pre-allocated output buffer (hop_samples int16s)
 *
 * Returns 0 on success. First call outputs zeros (warmup).
 */
LOCALVQE_API int localvqe_process_frame_s16(uintptr_t ctx,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out);

/**
 * Reset streaming state to initial zeros.
 * Call between utterances or when restarting processing.
 */
LOCALVQE_API void localvqe_reset(uintptr_t ctx);

/**
 * Configure the residual-echo noise gate.
 *
 * When enabled, any 256-sample output hop whose RMS sits at or below
 * `threshold_dbfs` (in dBFS) is replaced with zeros. Cleans up the
 * model's quiet residual on FE-only / silent-NE stretches that would
 * otherwise sound like "buffering" or amplified noise floor when the
 * downstream player peak-normalises. Operates on the OLA-synthesised
 * output, so it affects both the streaming and batch APIs.
 *
 * Off by default. Setting `threshold_dbfs = -45.0` is a reasonable
 * starting point: it gates frames that contain only model residual
 * (~-60 to -80 dBFS) but preserves typical speech (~-30 to -10 dBFS).
 *
 * Trade-off: a hard gate also mutes legitimate quiet speech below
 * threshold (distant or whispered NE). The model's NE-preservation
 * is the wrong place to fix this in the gate; tighten the threshold
 * (more negative) if the model is known to preserve such cases well,
 * loosen if not.
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_set_noise_gate(uintptr_t ctx,
                                        int enabled,
                                        float threshold_dbfs);

/**
 * Get the current noise-gate configuration.
 *
 * `enabled_out` and `threshold_dbfs_out` may each be NULL if the
 * caller doesn't want the corresponding value.
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_get_noise_gate(uintptr_t ctx,
                                        int* enabled_out,
                                        float* threshold_dbfs_out);

#ifdef __cplusplus
}
#endif

#endif /* LOCALVQE_API_H */
