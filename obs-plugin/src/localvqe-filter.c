#include <obs-module.h>
#include <media-io/audio-resampler.h>
#include <util/deque.h>
#include <util/threading.h>
#include <util/platform.h>

#include <localvqe_api.h>

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define LVQE_RATE      16000u
#define LVQE_HOP       256u
#define LVQE_HOP_BYTES (LVQE_HOP * sizeof(float))

/* Sanity: anywhere we drain the hop loop, cap iterations so a bad timestamp
 * or queue-size invariant can never wedge filter_audio. At 16 kHz a single
 * OBS callback delivers at most ~10–20 ms of audio (one or two hops), so
 * many hundreds of hops in a single call already means something is wrong. */
#define LVQE_MAX_HOPS_PER_CALL 1024u

/* The downsampler can't legitimately produce more frames than the upper
 * bound below. Used to sanity-check audio_resampler_resample outputs and
 * to bound the up-resample scratch buffer. */
#define LVQE_MAX_FRAMES_PER_CHUNK 65536u

/* If successive chunks' timestamps leave a forward gap of more than this,
 * or jump backwards beyond the queued head, treat it as a discontinuity
 * (Media Source seek/loop, source restart, OBS clock reset) — flush
 * the queue so head_ts can re-stamp from the new chunk.
 *
 * Must be well above OBS's per-chunk buffering jitter, which we've
 * observed alternating ±60 ms on Desktop Audio with zero cumulative
 * drift. Tightening below that range causes a flush-per-chunk loop
 * that wrecks alignment instead of preserving it. 500 ms is small
 * enough to catch any user-initiated seek/loop. */
#define LVQE_TS_DISCONTINUITY_NS (500ull * 1000000ull)  /* 500 ms */

#define LVQE_BLOG(level, fmt, ...) \
	blog(level, "[obs-localvqe] " fmt, ##__VA_ARGS__)

_Static_assert(LVQE_HOP_BYTES == LVQE_HOP * sizeof(float),
	       "hop byte count must equal LVQE_HOP * sizeof(float)");
_Static_assert(LVQE_HOP > 0 && LVQE_HOP <= 4096,
	       "LVQE_HOP must be a reasonable positive frame count");
_Static_assert(LVQE_RATE > 0, "LVQE_RATE must be positive");

/* OBS's enum speaker_layout values happen to equal channel counts
 * (SPEAKERS_MONO=1, SPEAKERS_STEREO=2, …, SPEAKERS_7POINT1=8), which is
 * what lets `(enum speaker_layout)f->obs_channels` work in
 * build_io_resamplers. Pin that down — if OBS ever renumbers the enum,
 * this asserts at compile time instead of silently producing wrong audio. */
_Static_assert((int)SPEAKERS_MONO       == 1, "SPEAKERS_MONO must equal 1");
_Static_assert((int)SPEAKERS_STEREO     == 2, "SPEAKERS_STEREO must equal 2");
_Static_assert((int)SPEAKERS_2POINT1    == 3, "SPEAKERS_2POINT1 must equal 3");
_Static_assert((int)SPEAKERS_4POINT0    == 4, "SPEAKERS_4POINT0 must equal 4");
_Static_assert((int)SPEAKERS_4POINT1    == 5, "SPEAKERS_4POINT1 must equal 5");
_Static_assert((int)SPEAKERS_5POINT1    == 6, "SPEAKERS_5POINT1 must equal 6");
_Static_assert((int)SPEAKERS_7POINT1    == 8, "SPEAKERS_7POINT1 must equal 8");

/* If the OBS mix rate is below the model rate, our framing (256-sample
 * hops at LVQE_RATE) blows past one OBS chunk per hop, scratch-buffer
 * sizing assumes the opposite direction, and want_16k can exceed our
 * upper bound. None of this is supported. Refuse cleanly. */
#define LVQE_MIN_OBS_RATE LVQE_RATE

#define S_MODEL_PATH       "model_path"
#define S_NOISE_GATE_EN    "noise_gate_enabled"
#define S_NOISE_GATE_DB    "noise_gate_threshold_dbfs"
#define S_REF_SOURCE       "reference_source"
#define S_THREADS          "n_threads"

/* 0 = auto: the library caps at 4 and respects sched_getaffinity, so
 * taskset / cgroup / VM CPU limits aren't exceeded. We pick the default
 * conservatively because the model is small enough that going past ~4
 * threads has diminishing returns on every CPU we've measured, and a
 * hard-coded 4 would over-subscribe in resource-constrained envs.
 * Upper bound stays 32 — that's the API's own validation limit. */
#define LVQE_THREADS_DEFAULT 0
#define LVQE_THREADS_MIN     0
#define LVQE_THREADS_MAX     32

#define T_(s)              obs_module_text(s)

struct lvqe_filter {
	obs_source_t *parent;

	localvqe_ctx_t ctx;
	char *model_path;
	int   n_threads;  /* last value passed to localvqe_options_set_threads */

	uint32_t obs_rate;       /* OBS mix rate, typically 48000 */
	uint32_t obs_channels;   /* OBS mix channels */

	/* OBS rate × N planar  →  16 kHz mono float */
	audio_resampler_t *down;
	/* 16 kHz mono float    →  OBS rate × N planar */
	audio_resampler_t *up;

	/* 16 kHz mono float ring buffers (bytes). */
	struct deque in_16k;
	struct deque out_16k;
	uint64_t     in_16k_head_ts;   /* timestamp of in_16k[0], if has_ts */
	bool         in_16k_has_ts;

	/* OBS-rate planar float output rings, one per channel. Decouples
	 * the variable-count up-resample output from the fixed-size buffer
	 * OBS expects back: each filter_audio drains exactly `audio->frames`
	 * (or zero-fills on per-channel underflow). Without this, ±1-frame
	 * resampler jitter left stale mic tail bytes spliced into the
	 * output → constant chunk-boundary crackle. */
	struct deque out_obs[MAX_AV_PLANES];
	uint64_t output_underflows;

	float hop_mic[LVQE_HOP];
	float hop_ref[LVQE_HOP];
	float hop_out[LVQE_HOP];

	/* Up-resample scratch, sized once for the worst-case drain so the
	 * hot path never mallocs. ~256 KB per filter instance. */
	float scratch_up[LVQE_MAX_FRAMES_PER_CHUNK];

	/* Sidechain reference (AEC). NULL → zeros into the ref input. */
	obs_weak_source_t *weak_ref;
	pthread_mutex_t ref_lock;
	struct deque ref_16k;
	uint64_t     ref_16k_head_ts;
	bool         ref_16k_has_ts;
	audio_resampler_t *ref_down;

};

/* ── Helpers ────────────────────────────────────────────────────────── */

static inline uint64_t samples_to_ns(uint64_t n)
{
	return (n * 1000000000ull) / LVQE_RATE;
}

/* OBS pre-normalises every source to the global mix rate. If a stray
 * obs_reset_audio changed it since we built our resamplers, bail rather
 * than feed wrongly-rated audio into the model. `who` labels the log
 * line so the source of the bail is grep-able. Returns true if OK. */
static bool obs_mix_config_ok(const struct lvqe_filter *f, const char *who)
{
	audio_t *aout = obs_get_audio();
	if (aout &&
	    audio_output_get_sample_rate(aout) == f->obs_rate &&
	    audio_output_get_channels(aout)    == f->obs_channels)
		return true;
	LVQE_BLOG(LOG_WARNING,
		  "OBS audio config changed under us "
		  "(rate=%u→%u ch=%u→%u); %s",
		  f->obs_rate,
		  aout ? audio_output_get_sample_rate(aout) : 0,
		  f->obs_channels,
		  aout ? (uint32_t)audio_output_get_channels(aout) : 0,
		  who);
	return false;
}

/* Flush on a large forward gap, or when the new chunk jumps backwards
 * beyond the queued head.  An overlap with the predicted queue tail is
 * normal and must not flush it.  The next push re-stamps head_ts from
 * `chunk_ts`. `label` distinguishes mic vs ref in the log. */
static bool flush_on_ts_discontinuity(struct deque *q,
				      uint64_t *head_ts, bool *has_ts,
				      uint64_t chunk_ts, const char *label)
{
	if (!*has_ts || q->size == 0) return false;
	uint64_t queue_samples = q->size / sizeof(float);
	int64_t skew = (int64_t)chunk_ts -
		       (int64_t)(*head_ts + samples_to_ns(queue_samples));

	/* A negative skew normally means OBS delivered an overlapping source
	 * block (OBS 32 does this for monitored/global audio sources) or that
	 * the reference queue is deliberately ahead of the mic.  Neither is a
	 * clock reset.  Only regard it as a backwards discontinuity when the
	 * new block predates the queue *head*, not merely its predicted tail. */
	bool forward_gap = skew > (int64_t)LVQE_TS_DISCONTINUITY_NS;
	bool before_head = chunk_ts < *head_ts &&
			   *head_ts - chunk_ts > LVQE_TS_DISCONTINUITY_NS;
	if (!forward_gap && !before_head) return false;
	LVQE_BLOG(LOG_INFO,
		  "%s timestamp discontinuity (%lld ns skew); flushing queue",
		  label, (long long)skew);
	deque_free(q);
	*has_ts = false;
	return true;
}

static void release_resamplers(struct lvqe_filter *f)
{
	if (f->down) { audio_resampler_destroy(f->down); f->down = NULL; }
	if (f->up)   { audio_resampler_destroy(f->up);   f->up   = NULL; }
	if (f->ref_down) { audio_resampler_destroy(f->ref_down); f->ref_down = NULL; }
}

static void build_io_resamplers(struct lvqe_filter *f)
{
	assert(f != NULL);
	assert(f->obs_rate > 0 && f->obs_rate <= 384000);
	assert(f->obs_channels >= 1 && f->obs_channels <= MAX_AV_PLANES);

	struct resample_info obs_side = {
		.samples_per_sec = f->obs_rate,
		.format          = AUDIO_FORMAT_FLOAT_PLANAR,
		.speakers        = (enum speaker_layout)f->obs_channels,
	};
	struct resample_info lvqe_side = {
		.samples_per_sec = LVQE_RATE,
		.format          = AUDIO_FORMAT_FLOAT,
		.speakers        = SPEAKERS_MONO,
	};
	f->down     = audio_resampler_create(&lvqe_side, &obs_side);
	f->up       = audio_resampler_create(&obs_side, &lvqe_side);
	/* Same shape as `down`: the sidechain callback receives audio that
	 * OBS has already normalised to the mix rate, so the conversion is
	 * identical. Built once here — obs_rate is fixed for the filter's
	 * lifetime (only obs_reset_audio could change it, which we detect
	 * on the audio paths and bail on). */
	f->ref_down = audio_resampler_create(&lvqe_side, &obs_side);
	if (!f->down || !f->up || !f->ref_down) {
		LVQE_BLOG(LOG_ERROR,
			  "failed to build resamplers (rate=%u ch=%u)",
			  f->obs_rate, f->obs_channels);
	}
}

static void detach_reference(struct lvqe_filter *f);

/* ── Sidechain (reference / AEC) ────────────────────────────────────── */

static void ref_audio_cb(void *param, obs_source_t *src,
			 const struct audio_data *audio, bool muted)
{
	UNUSED_PARAMETER(src);
	struct lvqe_filter *f = param;
	assert(f != NULL);
	if (!audio || !audio->frames)
		return;
	if (audio->frames > LVQE_MAX_FRAMES_PER_CHUNK) {
		LVQE_BLOG(LOG_WARNING,
			  "ref chunk implausibly large (%u frames); dropping",
			  audio->frames);
		return;
	}
	if (!muted && !audio->data[0]) {
		/* OBS contract: data[0] is non-NULL for any audible chunk. */
		return;
	}

	if (!obs_mix_config_ok(f, "dropping ref chunk"))
		return;

	pthread_mutex_lock(&f->ref_lock);

	flush_on_ts_discontinuity(&f->ref_16k, &f->ref_16k_head_ts,
				  &f->ref_16k_has_ts, audio->timestamp, "ref");

	uint8_t *out[MAX_AV_PLANES] = {0};
	uint32_t out_frames = 0;
	uint64_t ts_off = 0;
	const uint8_t *in[MAX_AV_PLANES];
	for (size_t i = 0; i < MAX_AV_PLANES; ++i)
		in[i] = audio->data[i];

	bool was_empty = (f->ref_16k.size == 0);

	if (muted) {
		static const float zeros[LVQE_MAX_FRAMES_PER_CHUNK] = {0};
		uint32_t silent = (audio->frames * LVQE_RATE) / f->obs_rate;
		assert(silent <= LVQE_MAX_FRAMES_PER_CHUNK);
		deque_push_back(&f->ref_16k, zeros,
				(size_t)silent * sizeof(float));
		out_frames = silent;
	} else if (!f->ref_down) {
		out_frames = 0;
	} else if (!audio_resampler_resample(f->ref_down, out, &out_frames,
					    &ts_off, in, audio->frames)) {
		out_frames = 0;
	} else if (out_frames > LVQE_MAX_FRAMES_PER_CHUNK || !out[0]) {
		LVQE_BLOG(LOG_WARNING,
			  "ref resampler returned implausible output "
			  "(frames=%u, out[0]=%p); dropping",
			  out_frames, (void *)out[0]);
		out_frames = 0;
	} else {
		deque_push_back(&f->ref_16k, out[0],
				out_frames * sizeof(float));
	}

	if (was_empty && out_frames > 0) {
		/* Resampler ts_offset is small (sub-ms) and identical between
		 * the mic and ref resamplers, so it cancels in alignment math.
		 * Use the input chunk timestamp directly. */
		f->ref_16k_head_ts = audio->timestamp;
		f->ref_16k_has_ts = true;
	}

	pthread_mutex_unlock(&f->ref_lock);
}

static void detach_reference(struct lvqe_filter *f)
{
	if (f->weak_ref) {
		obs_source_t *src = obs_weak_source_get_source(f->weak_ref);
		if (src) {
			obs_source_remove_audio_capture_callback(
				src, ref_audio_cb, f);
			obs_source_release(src);
		}
		obs_weak_source_release(f->weak_ref);
		f->weak_ref = NULL;
	}
	pthread_mutex_lock(&f->ref_lock);
	deque_free(&f->ref_16k);
	f->ref_16k_has_ts = false;
	pthread_mutex_unlock(&f->ref_lock);
}

static void attach_reference(struct lvqe_filter *f, const char *name)
{
	assert(f != NULL);
	detach_reference(f);
	if (!name || !*name) return;

	obs_source_t *src = obs_get_source_by_name(name);
	if (!src) {
		LVQE_BLOG(LOG_WARNING,
			  "reference source '%s' not found", name);
		return;
	}
	f->weak_ref = obs_source_get_weak_source(src);
	if (!f->weak_ref) {
		LVQE_BLOG(LOG_WARNING,
			  "failed to weak-ref reference source '%s'", name);
		obs_source_release(src);
		return;
	}
	obs_source_add_audio_capture_callback(src, ref_audio_cb, f);
	obs_source_release(src);
}

/* ── obs_source_info callbacks ──────────────────────────────────────── */

static const char *lvqe_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	/* obs_module_text returns NULL if the locale lookup misses (e.g.
	 * locale file failed to load). Fall back to a literal so OBS's
	 * source picker still has a label. */
	const char *name = T_("LocalVQE.FilterName");
	return name ? name : "LocalVQE";
}

static void lvqe_update(void *data, obs_data_t *settings)
{
	struct lvqe_filter *f = data;
	assert(f != NULL);
	assert(settings != NULL);

	const char *path = obs_data_get_string(settings, S_MODEL_PATH);
	if (!path) path = "";
	int64_t threads_i64 = obs_data_get_int(settings, S_THREADS);
	if (threads_i64 < LVQE_THREADS_MIN) threads_i64 = LVQE_THREADS_MIN;
	if (threads_i64 > LVQE_THREADS_MAX) threads_i64 = LVQE_THREADS_MAX;
	int threads = (int)threads_i64;

	/* (Re)load the model when the path or thread count changes. Threads
	 * are baked into the backend at ggml_backend_set_n_threads call —
	 * the only way to change them is a fresh ctx. */
	bool need_reload = !f->model_path || strcmp(f->model_path, path) != 0 ||
			   f->n_threads != threads;
	if (need_reload) {
		if (f->ctx) { localvqe_free(f->ctx); f->ctx = 0; }
		bfree(f->model_path);
		f->model_path = bstrdup(path);
		f->n_threads = threads;
		if (*path) {
			localvqe_options_t opts = localvqe_options_new();
			if (!opts) {
				LVQE_BLOG(LOG_ERROR,
					  "localvqe_options_new returned NULL");
			} else {
				if (localvqe_options_set_model_path(opts, path) < 0)
					LVQE_BLOG(LOG_WARNING,
						  "options_set_model_path "
						  "rejected '%s'", path);
				if (localvqe_options_set_threads(opts, threads) < 0)
					LVQE_BLOG(LOG_WARNING,
						  "options_set_threads "
						  "rejected n=%d", threads);
				f->ctx = localvqe_new_with_options(opts);
				localvqe_options_free(opts);
			}
			if (!f->ctx) {
				LVQE_BLOG(LOG_ERROR,
					  "localvqe_new_with_options failed for "
					  "'%s' (threads=%d)", path, threads);
			} else {
				/* Contract check: confirm the model agrees with
				 * our compile-time framing assumptions. */
				int sr = localvqe_sample_rate(f->ctx);
				int hop = localvqe_hop_length(f->ctx);
				if (sr != (int)LVQE_RATE || hop != (int)LVQE_HOP) {
					LVQE_BLOG(LOG_ERROR,
						  "model framing mismatch: "
						  "sr=%d hop=%d (expected %u/%u)",
						  sr, hop, LVQE_RATE, LVQE_HOP);
					localvqe_free(f->ctx);
					f->ctx = 0;
				}
			}
		}
	}

	if (f->ctx) {
		int en = (int)obs_data_get_bool(settings, S_NOISE_GATE_EN);
		double db = obs_data_get_double(settings, S_NOISE_GATE_DB);
		/* Clamp to the same dBFS range the property slider exposes —
		 * keeps anything pathological out of the model. */
		if (db < -120.0) db = -120.0;
		if (db >    0.0) db =    0.0;
		if (localvqe_set_noise_gate(f->ctx, en, (float)db) < 0) {
			const char *err = localvqe_last_error(f->ctx);
			LVQE_BLOG(LOG_WARNING,
				  "localvqe_set_noise_gate(en=%d, db=%.1f) "
				  "failed: %s",
				  en, db, err ? err : "unknown");
		}
	}

	const char *ref_name = obs_data_get_string(settings, S_REF_SOURCE);
	attach_reference(f, ref_name ? ref_name : "");
}

static void *lvqe_create(obs_data_t *settings, obs_source_t *source)
{
	assert(settings != NULL);
	assert(source != NULL);

	struct lvqe_filter *f = bzalloc(sizeof(*f));
	if (!f) {
		/* libobs's bzalloc aborts on OOM in practice, but the contract
		 * doesn't guarantee that — never dereference an unchecked
		 * allocation. */
		LVQE_BLOG(LOG_ERROR, "bzalloc(%zu) failed", sizeof(*f));
		return NULL;
	}
	f->parent = source;
	if (pthread_mutex_init(&f->ref_lock, NULL) != 0) {
		LVQE_BLOG(LOG_ERROR, "pthread_mutex_init failed");
		bfree(f);
		return NULL;
	}

	audio_t *a = obs_get_audio();
	if (!a) {
		LVQE_BLOG(LOG_ERROR, "obs_get_audio() returned NULL");
		pthread_mutex_destroy(&f->ref_lock);
		bfree(f);
		return NULL;
	}
	f->obs_rate     = audio_output_get_sample_rate(a);
	f->obs_channels = audio_output_get_channels(a);

	if (f->obs_rate < LVQE_MIN_OBS_RATE ||
	    f->obs_channels == 0 ||
	    f->obs_channels > MAX_AV_PLANES) {
		LVQE_BLOG(LOG_ERROR,
			  "unsupported OBS audio config: rate=%u (need ≥%u) "
			  "ch=%u (need 1..%d); filter will pass audio through",
			  f->obs_rate, LVQE_MIN_OBS_RATE,
			  f->obs_channels, MAX_AV_PLANES);
		pthread_mutex_destroy(&f->ref_lock);
		bfree(f);
		return NULL;
	}
	/* OBS's filter_audio always delivers planar float at the mix; warn
	 * if the mix is configured otherwise — we'd silently mis-interpret
	 * the data otherwise. */
	const struct audio_output_info *ainfo = audio_output_get_info(a);
	if (ainfo && ainfo->format != AUDIO_FORMAT_FLOAT_PLANAR) {
		LVQE_BLOG(LOG_WARNING,
			  "OBS audio format is not FLOAT_PLANAR (%d) — "
			  "filter will pass audio through",
			  (int)ainfo->format);
	}

	build_io_resamplers(f);

	/* filter_audio must return one block for every block it receives.
	 * Seed the output rings with the algorithmic delay so the first model
	 * hop can accumulate without returning NULL (which means "drop this
	 * block" to libobs, not "hold and retry"). */
	static const float silence[LVQE_MAX_FRAMES_PER_CHUNK] = {0};
	uint64_t latency_frames_64 =
		((uint64_t)LVQE_HOP * f->obs_rate + LVQE_RATE - 1) / LVQE_RATE;
	uint32_t latency_frames = (uint32_t)latency_frames_64;
	assert(latency_frames <= LVQE_MAX_FRAMES_PER_CHUNK);
	for (size_t ch = 0; ch < f->obs_channels; ++ch)
		deque_push_back(&f->out_obs[ch], silence,
				(size_t)latency_frames * sizeof(float));

	lvqe_update(f, settings);
	return f;
}

static void lvqe_destroy(void *data)
{
	struct lvqe_filter *f = data;
	assert(f != NULL);
	detach_reference(f);
	release_resamplers(f);
	if (f->ctx) localvqe_free(f->ctx);
	bfree(f->model_path);
	deque_free(&f->in_16k);
	deque_free(&f->out_16k);
	deque_free(&f->ref_16k);
	for (size_t ch = 0; ch < MAX_AV_PLANES; ++ch)
		deque_free(&f->out_obs[ch]);
	pthread_mutex_destroy(&f->ref_lock);
	bfree(f);
}

static void lvqe_defaults(obs_data_t *s)
{
	/* Resolve the bundled GGUF as the default model path so a fresh
	 * install "just works". Keep ordered newest-first. */
	static const char *const candidates[] = {
		"localvqe-v1.3-4.8M-f32.gguf",
		"localvqe-v1.2-1.3M-f32.gguf",
		"localvqe-v1.1-1.3M-f32.gguf",
		"localvqe-v1-1.3M-f32.gguf",
		NULL,
	};
	char *resolved = NULL;
	for (size_t i = 0; candidates[i] && !resolved; ++i) {
		char *p = obs_module_file(candidates[i]);
		if (p && os_file_exists(p)) { resolved = p; break; }
		bfree(p);
	}
	obs_data_set_default_string(s, S_MODEL_PATH, resolved ? resolved : "");
	bfree(resolved);

	obs_data_set_default_bool(s, S_NOISE_GATE_EN, false);
	obs_data_set_default_double(s, S_NOISE_GATE_DB, -45.0);
	obs_data_set_default_string(s, S_REF_SOURCE, "");
	obs_data_set_default_int(s, S_THREADS, LVQE_THREADS_DEFAULT);
}

static bool enum_audio_sources(void *param, obs_source_t *src)
{
	obs_property_t *p = param;
	assert(p != NULL);
	if (!src) return true;
	uint32_t flags = obs_source_get_output_flags(src);
	if ((flags & OBS_SOURCE_AUDIO) == 0) return true;
	const char *name = obs_source_get_name(src);
	if (!name || !*name) return true;
	obs_property_list_add_string(p, name, name);
	return true;
}

static obs_properties_t *lvqe_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();
	if (!props) {
		LVQE_BLOG(LOG_ERROR, "obs_properties_create returned NULL");
		return NULL;
	}

	obs_properties_add_path(props, S_MODEL_PATH, T_("LocalVQE.ModelPath"),
				OBS_PATH_FILE, "GGUF (*.gguf)", NULL);

	/* Spinbox, not slider — model reloads on thread-count change, so a
	 * drag-to-scrub would re-allocate the ctx per tick. The spinbox
	 * fires once per Enter/spin click. */
	obs_properties_add_int(props, S_THREADS, T_("LocalVQE.Threads"),
			       LVQE_THREADS_MIN, LVQE_THREADS_MAX, 1);

	obs_properties_add_bool(props, S_NOISE_GATE_EN,
				T_("LocalVQE.NoiseGate.Enabled"));
	obs_properties_add_float_slider(props, S_NOISE_GATE_DB,
					T_("LocalVQE.NoiseGate.ThresholdDbFS"),
					-90.0, -10.0, 0.5);

	obs_property_t *ref = obs_properties_add_list(
		props, S_REF_SOURCE, T_("LocalVQE.ReferenceSource"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	if (ref) {
		obs_property_list_add_string(
			ref, T_("LocalVQE.ReferenceSource.None"), "");
		obs_enum_sources(enum_audio_sources, ref);
	} else {
		LVQE_BLOG(LOG_WARNING,
			  "failed to add reference-source list property");
	}

	return props;
}

/* ── filter_audio: the hot path ─────────────────────────────────────── */

static struct obs_audio_data *lvqe_filter_audio(void *data,
						struct obs_audio_data *audio)
{
	struct lvqe_filter *f = data;
	assert(f != NULL);
	if (!audio) return NULL;
	if (audio->frames == 0) return audio;
	if (audio->frames > LVQE_MAX_FRAMES_PER_CHUNK) {
		LVQE_BLOG(LOG_WARNING,
			  "mic chunk implausibly large (%u frames); "
			  "passing through unmodified",
			  audio->frames);
		return audio;
	}
	if (!audio->data[0]) {
		/* OBS contract violation — first plane must exist. */
		return audio;
	}
	if (!f->ctx || !f->down || !f->up)
		return audio; /* not configured — pass through */

	if (!obs_mix_config_ok(f, "passing audio through"))
		return audio;

	assert(f->obs_channels >= 1 && f->obs_channels <= MAX_AV_PLANES);
	assert(f->in_16k.size  % sizeof(float) == 0);
	assert(f->out_16k.size % sizeof(float) == 0);

	/* 1. Detect mic-side timestamp discontinuity (seek/loop/restart). */
	flush_on_ts_discontinuity(&f->in_16k, &f->in_16k_head_ts,
				  &f->in_16k_has_ts, audio->timestamp, "mic");

	/* 2. Downsample the mic chunk to 16 kHz mono → in_16k. */
	{
		uint8_t *out[MAX_AV_PLANES] = {0};
		uint32_t out_frames = 0;
		uint64_t ts_off = 0;
		const uint8_t *in[MAX_AV_PLANES];
		for (size_t i = 0; i < MAX_AV_PLANES; ++i)
			in[i] = audio->data[i];

		bool was_empty = (f->in_16k.size == 0);
		if (!audio_resampler_resample(f->down, out, &out_frames,
					      &ts_off, in, audio->frames)) {
			LVQE_BLOG(LOG_WARNING,
				  "mic downsample failed for %u frames",
				  audio->frames);
		} else if (out_frames > LVQE_MAX_FRAMES_PER_CHUNK || !out[0]) {
			LVQE_BLOG(LOG_WARNING,
				  "mic resampler returned implausible "
				  "output (frames=%u, out[0]=%p)",
				  out_frames, (void *)out[0]);
		} else if (out_frames > 0) {
			deque_push_back(&f->in_16k, out[0],
					out_frames * sizeof(float));
			if (was_empty) {
				f->in_16k_head_ts = audio->timestamp;
				f->in_16k_has_ts = true;
			}
		}
	}

	/* 3. Drain as many 256-frame hops as we can. */
	const uint64_t hop_ns = samples_to_ns(LVQE_HOP);
	uint32_t hops_done = 0;
	while (f->in_16k.size >= LVQE_HOP_BYTES &&
	       hops_done < LVQE_MAX_HOPS_PER_CALL) {
		++hops_done;
		assert(f->in_16k_has_ts);
		assert(f->in_16k.size % sizeof(float) == 0);

		uint64_t mic_hop_ts = f->in_16k_head_ts;
		deque_pop_front(&f->in_16k, f->hop_mic, LVQE_HOP_BYTES);
		f->in_16k_head_ts += hop_ns;
		if (f->in_16k.size == 0) f->in_16k_has_ts = false;

		/* Align ref to this mic hop's timestamp. AlignBlock soaks
		 * the residual ≤320 ms; we only need to keep the queues
		 * within that window. Lag = how far ref lags mic. */
		bool used_ref = false;
		pthread_mutex_lock(&f->ref_lock);
		if (f->ref_16k_has_ts) {
			int64_t lag_ns = (int64_t)mic_hop_ts -
					 (int64_t)f->ref_16k_head_ts;
			if (lag_ns > 0) {
				/* Ref is older than mic — drop the stale prefix. */
				uint64_t drop_samples =
					((uint64_t)lag_ns * LVQE_RATE) /
					1000000000ull;
				size_t drop_bytes = drop_samples * sizeof(float);
				if (drop_bytes > f->ref_16k.size)
					drop_bytes = f->ref_16k.size;
				if (drop_bytes) {
					deque_pop_front(&f->ref_16k, NULL,
							drop_bytes);
					f->ref_16k_head_ts += samples_to_ns(
						drop_bytes / sizeof(float));
				}
			}
			/* If ref is now in the past or roughly aligned (head_ts
			 * within one hop of mic) and we have a full hop, use it.
			 * Otherwise the ref is still in the future — feed zeros
			 * for this hop and let it accumulate. */
			int64_t lead_ns = (int64_t)f->ref_16k_head_ts -
					  (int64_t)mic_hop_ts;
			if (lead_ns < (int64_t)hop_ns &&
			    f->ref_16k.size >= LVQE_HOP_BYTES) {
				deque_pop_front(&f->ref_16k, f->hop_ref,
						LVQE_HOP_BYTES);
				f->ref_16k_head_ts += hop_ns;
				if (f->ref_16k.size == 0)
					f->ref_16k_has_ts = false;
				used_ref = true;
			}
		}
		pthread_mutex_unlock(&f->ref_lock);

		/* No reference (no sidechain configured, or it hasn't caught
		 * up yet) — feed zeros so the AEC head sees "nothing playing".
		 * Never feed mic here; mic == ref would cancel everything. */
		if (!used_ref) memset(f->hop_ref, 0, LVQE_HOP_BYTES);

		assert(f->hop_mic && f->hop_ref && f->hop_out);
		if (localvqe_process_frame_f32(f->ctx, f->hop_mic, f->hop_ref,
					       LVQE_HOP, f->hop_out) != 0) {
			const char *err = localvqe_last_error(f->ctx);
			LVQE_BLOG(LOG_WARNING,
				  "localvqe_process_frame_f32 failed: %s",
				  err ? err : "unknown");
			memcpy(f->hop_out, f->hop_mic, LVQE_HOP_BYTES);
		}

		deque_push_back(&f->out_16k, f->hop_out, LVQE_HOP_BYTES);
	}
	if (hops_done >= LVQE_MAX_HOPS_PER_CALL) {
		LVQE_BLOG(LOG_WARNING,
			  "hit hop drain cap (%u) in one filter_audio call — "
			  "queues may have desynced",
			  LVQE_MAX_HOPS_PER_CALL);
	}

	/* 4. Upsample every 16 kHz sample we have ready and push the result
	 *    into the per-channel OBS-rate output rings. The resampler
	 *    returns a variable frame count (±1 around the rate ratio);
	 *    fixing the OBS-side chunk size to what we then drain avoids
	 *    splicing fresh output with stale mic tail bytes. */
	while (f->out_16k.size >= sizeof(float)) {
		uint32_t to_up = (uint32_t)(f->out_16k.size / sizeof(float));
		if (to_up > LVQE_MAX_FRAMES_PER_CHUNK)
			to_up = LVQE_MAX_FRAMES_PER_CHUNK;

		deque_pop_front(&f->out_16k, f->scratch_up,
				(size_t)to_up * sizeof(float));

		const uint8_t *in_planes[MAX_AV_PLANES] = {
			(uint8_t *)f->scratch_up };
		uint8_t *out_planes[MAX_AV_PLANES] = {0};
		uint32_t out_frames = 0;
		uint64_t ts_off = 0;
		if (!audio_resampler_resample(f->up, out_planes, &out_frames,
					      &ts_off, in_planes, to_up)) {
			LVQE_BLOG(LOG_WARNING,
				  "up-resample failed for %u 16k frames", to_up);
			break;
		}
		if (out_frames == 0) break;
		if (out_frames > LVQE_MAX_FRAMES_PER_CHUNK) {
			LVQE_BLOG(LOG_WARNING,
				  "up-resampler returned implausible frame "
				  "count (%u); dropping",
				  out_frames);
			break;
		}
		const size_t bytes = (size_t)out_frames * sizeof(float);
		for (size_t ch = 0; ch < f->obs_channels &&
				    ch < MAX_AV_PLANES; ++ch) {
			if (out_planes[ch])
				deque_push_back(&f->out_obs[ch],
						out_planes[ch], bytes);
		}
	}

	/* 5. Drain exactly `audio->frames` per channel into OBS's buffer.
	 *    Returning NULL drops the current block in libobs; it does not ask
	 *    OBS to retry it.  The startup padding above should make underflow
	 *    impossible during normal operation, but zero-fill any anomalous
	 *    shortfall so timestamps and block cadence remain continuous. */
	const size_t need = (size_t)audio->frames * sizeof(float);
	if (f->out_obs[0].size < need) {
		++f->output_underflows;
		LVQE_BLOG(LOG_WARNING,
			  "output underflow #%llu: have=%zu need=%zu frames "
			  "(in16=%zu out16=%zu hops=%u)",
			  (unsigned long long)f->output_underflows,
			  f->out_obs[0].size / sizeof(float),
			  (size_t)audio->frames,
			  f->in_16k.size / sizeof(float),
			  f->out_16k.size / sizeof(float), hops_done);
	}
	for (size_t ch = 0; ch < f->obs_channels && ch < MAX_AV_PLANES; ++ch) {
		if (!audio->data[ch]) continue;
		size_t available = f->out_obs[ch].size;
		if (available > need) available = need;
		if (available)
			deque_pop_front(&f->out_obs[ch], audio->data[ch], available);
		if (available < need)
			memset(audio->data[ch] + available, 0, need - available);
	}

	/* Keep OBS's timestamp unchanged.  The seeded silence represents the
	 * real filter delay in the sample stream; back-dating each block makes
	 * OBS's dynamic audio buffering grow in an attempt to compensate. */
	return audio;
}

struct obs_source_info localvqe_filter_info = {
	.id             = "localvqe_filter",
	.type           = OBS_SOURCE_TYPE_FILTER,
	.output_flags   = OBS_SOURCE_AUDIO,
	.get_name       = lvqe_get_name,
	.create         = lvqe_create,
	.destroy        = lvqe_destroy,
	.update         = lvqe_update,
	.get_defaults   = lvqe_defaults,
	.get_properties = lvqe_properties,
	.filter_audio   = lvqe_filter_audio,
};
