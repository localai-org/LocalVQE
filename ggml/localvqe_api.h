#ifndef LOCALVQE_API_H
#define LOCALVQE_API_H

/**
 * LocalVQE C API — purego-compatible shared library interface.
 *
 * All handles are exposed as opaque uintptr_t aliases (localvqe_ctx_t,
 * localvqe_options_t). Underlying ABI is a flat integer for FFI consumers
 * (purego, ctypes); the typedefs are documentation for C consumers and
 * keep ctx vs. options visibly distinct in function signatures.
 *
 * Typical usage:
 *   localvqe_ctx_t ctx = localvqe_new("model.gguf");
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

typedef uintptr_t localvqe_ctx_t;
typedef uintptr_t localvqe_options_t;

/**
 * Create a new LocalVQE context by loading a GGUF model file.
 * Defaults to the CPU backend, device 0. Returns an opaque handle,
 * or 0 on failure.
 *
 * For anything other than the defaults, use the options-builder API
 * below — it is FFI-friendly (no struct layouts crossed) and can be
 * extended with new setters without breaking existing callers.
 */
LOCALVQE_API localvqe_ctx_t localvqe_new(const char* model_path);

/* ── Options builder ────────────────────────────────────────────────────────
 *
 * Opaque, extensible alternative to a positional constructor. Allocate an
 * options handle, call setters for the fields you care about, then pass the
 * handle to localvqe_new_with_options(). Callers never see the struct
 * layout, so adding new fields = adding new setters without breaking ABI.
 *
 *   localvqe_options_t opts = localvqe_options_new();
 *   localvqe_options_set_model_path(opts, "model.gguf");
 *   localvqe_options_set_backend(opts, "Vulkan");
 *   localvqe_options_set_device(opts, 1);
 *   localvqe_ctx_t ctx = localvqe_new_with_options(opts);
 *   localvqe_options_free(opts);
 *
 * Every setter returns 0 on success, -1 on a null handle, -2 on an invalid
 * argument value (null/empty string, negative index, etc.). The options
 * handle is owned by the caller and must be freed with
 * localvqe_options_free; new_with_options does not consume it.
 */

LOCALVQE_API localvqe_options_t localvqe_options_new(void);
LOCALVQE_API void               localvqe_options_free(localvqe_options_t opts);

LOCALVQE_API int localvqe_options_set_model_path(localvqe_options_t opts,
                                                 const char* model_path);
LOCALVQE_API int localvqe_options_set_backend(localvqe_options_t opts,
                                              const char* backend_name);
LOCALVQE_API int localvqe_options_set_device(localvqe_options_t opts,
                                             int device_index);

/**
 * Construct a context from a populated options handle. model_path must
 * have been set. Returns an opaque ctx handle, or 0 on failure.
 */
LOCALVQE_API localvqe_ctx_t localvqe_new_with_options(localvqe_options_t opts);

/**
 * Print every registered backend + device to stderr. No model required.
 * Useful for telling the user what to pass to localvqe_options_set_*.
 */
LOCALVQE_API void localvqe_list_devices(void);

/**
 * Print memory budget + graph op-type histogram for the loaded model.
 * Diagnostic only; cheap (no inference). Output goes to stdout.
 */
LOCALVQE_API void localvqe_print_profile(localvqe_ctx_t ctx);

/**
 * Free a LocalVQE context and all associated resources.
 */
LOCALVQE_API void localvqe_free(localvqe_ctx_t ctx);

/**
 * Process audio through the AEC model (float32 version).
 *
 * mic:       Microphone input (mono, float32, [-1,1] range, 16kHz)
 * ref:       Far-end reference (mono, float32, [-1,1] range, 16kHz)
 * n_samples: Number of samples in both mic and ref (must be >= 512)
 * out:       Pre-allocated output buffer (n_samples floats)
 *
 * Output is sample-aligned to input. The first hop is the
 * synthesis-window-tapered start of the first analysis frame: it
 * begins at zero and ramps up over `hop` samples, with no
 * discontinuity at any frame boundary.
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_process_f32(localvqe_ctx_t ctx,
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
 * Output is sample-aligned to input. The first hop is the
 * synthesis-window-tapered start of the first analysis frame: it
 * begins at zero and ramps up over `hop` samples, with no
 * discontinuity at any frame boundary.
 *
 * Returns 0 on success, negative on error.
 */
LOCALVQE_API int localvqe_process_s16(localvqe_ctx_t ctx,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out);

/**
 * Get the last error message, or empty string if no error.
 * The returned pointer is valid until the next API call on this context.
 */
LOCALVQE_API const char* localvqe_last_error(localvqe_ctx_t ctx);

/**
 * Get model sample rate (always 16000 currently).
 */
LOCALVQE_API int localvqe_sample_rate(localvqe_ctx_t ctx);

/**
 * Get hop length in samples (256).
 */
LOCALVQE_API int localvqe_hop_length(localvqe_ctx_t ctx);

/**
 * Get FFT size (512).
 */
LOCALVQE_API int localvqe_fft_size(localvqe_ctx_t ctx);

/**
 * Process a single hop of audio through the AEC model (float32 version).
 *
 * mic:         Microphone input (mono, float32, [-1,1], 16kHz)
 * ref:         Far-end reference (mono, float32, [-1,1], 16kHz)
 * hop_samples: Must equal hop_length (256)
 * out:         Pre-allocated output buffer (hop_samples floats)
 *
 * Returns 0 on success.
 */
LOCALVQE_API int localvqe_process_frame_f32(localvqe_ctx_t ctx,
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
 * Returns 0 on success.
 */
LOCALVQE_API int localvqe_process_frame_s16(localvqe_ctx_t ctx,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out);

/**
 * Reset streaming state to initial zeros.
 * Call between utterances or when restarting processing.
 */
LOCALVQE_API void localvqe_reset(localvqe_ctx_t ctx);

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
LOCALVQE_API int localvqe_set_noise_gate(localvqe_ctx_t ctx,
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
LOCALVQE_API int localvqe_get_noise_gate(localvqe_ctx_t ctx,
                                        int* enabled_out,
                                        float* threshold_dbfs_out);

#ifdef __cplusplus
}
#endif

#endif /* LOCALVQE_API_H */
