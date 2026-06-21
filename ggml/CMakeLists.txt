cmake_minimum_required(VERSION 3.20)
project(localvqe-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LOCALVQE_BUILD_SHARED "Build shared library with C API" OFF)
option(LOCALVQE_CUDA "Enable GGML CUDA backend" OFF)
option(LOCALVQE_VULKAN "Enable GGML Vulkan backend" OFF)

# libFuzzer-based fuzz targets for the C API. Requires clang. Implies
# -fsanitize=fuzzer,address,undefined unless overridden via
# LOCALVQE_FUZZ_SANITIZERS (e.g. "fuzzer,address" to drop UBSan).
option(LOCALVQE_FUZZ "Build libFuzzer harnesses (clang only)" OFF)
set(LOCALVQE_FUZZ_SANITIZERS "fuzzer,address,undefined"
    CACHE STRING "Sanitizer/fuzzer instrumentation for LOCALVQE_FUZZ targets")

# Need PIC for everything when building shared lib (ggml static lib gets linked in)
if(LOCALVQE_BUILD_SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# ggml from git submodule (ggml/vendor/ggml)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
if(LOCALVQE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_STATIC OFF CACHE BOOL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    # x86: build all CPU backend variants (SSE4.2 through AVX-512/Zen4) as shared
    # libraries, selected at runtime based on CPU capabilities.
    # This avoids needing -march=native (which Nix strips).
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_STATIC OFF CACHE BOOL "" FORCE)
    # Put all binaries and .so's in the same directory so ggml_backend_load_all()
    # finds the CPU variant libraries next to the executable.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "" FORCE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "" FORCE)
else()
    # Non-x86 (e.g. aarch64 / Raspberry Pi 5 Cortex-A76): GGML_CPU_ALL_VARIANTS
    # is x86-only and FATAL-errors here, so build a single self-contained binary
    # targeting the requested -march. Pass -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
    # for a Pi 5 (see docker/Dockerfile.arm64). No runtime backend DL — the CPU
    # backend is statically registered, so ggml_backend_load_all() still finds it.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_STATIC ON CACHE BOOL "" FORCE)
endif()
if(LOCALVQE_VULKAN)
    # Composes with the CPU-DL path above: Vulkan becomes a separate
    # libggml-vulkan.so in build/bin/ that ggml_backend_load_all() picks up
    # at runtime, falling back to the CPU variants when no Vulkan ICD is found.
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(vendor/ggml)

#
# Detect libsndfile without going through pkg-config. libsndfile.pc lists
# libFLAC / libvorbis / libogg in Requires.private; pkg_check_modules
# refuses to declare libsndfile found when those .pc files aren't on
# disk, even though we only link the public libsndfile.so. find_library
# follows the actual runtime ABI.
#
find_library(SNDFILE_LIBRARY NAMES sndfile)
find_path(SNDFILE_INCLUDE_DIR NAMES sndfile.h)
if(SNDFILE_LIBRARY AND SNDFILE_INCLUDE_DIR)
    set(SNDFILE_FOUND TRUE)
    set(SNDFILE_LIBRARIES ${SNDFILE_LIBRARY})
    set(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIR})
endif()

# Common library (npy I/O, comparison utilities, GGUF tensor loading).
add_library(dvqe_common STATIC common.cpp)
target_include_directories(dvqe_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dvqe_common PUBLIC ggml)

# Audio file I/O (libsndfile-backed). Linked by binaries that open files.
if(SNDFILE_FOUND)
    add_library(dvqe_audio_io STATIC audio_io.cpp)
    target_include_directories(dvqe_audio_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                    ${SNDFILE_INCLUDE_DIRS})
    target_link_libraries(dvqe_audio_io PUBLIC ${SNDFILE_LIBRARIES})
endif()

# Hand-rolled model (streaming forward_frame only — kept for legacy tests)
add_library(dvqe_model STATIC localvqe_model.cpp)
target_link_libraries(dvqe_model PUBLIC dvqe_common)

# GGML graph model (streaming + batch inference)
add_library(dvqe_graph STATIC localvqe_graph.cpp model_hash.cpp)
target_link_libraries(dvqe_graph PUBLIC dvqe_common)

# Main inference binary (CLI). Needs WAV I/O.
if(SNDFILE_FOUND)
    add_executable(localvqe localvqe.cpp localvqe_api.cpp)
    target_link_libraries(localvqe PRIVATE dvqe_graph dvqe_audio_io)
endif()

# Shared library with C API (for purego / dlopen)
if(LOCALVQE_BUILD_SHARED)
    add_library(localvqe_shared SHARED localvqe_api.cpp)
    target_link_libraries(localvqe_shared PRIVATE dvqe_graph)
    target_compile_definitions(localvqe_shared PRIVATE LOCALVQE_BUILD)
    set_target_properties(localvqe_shared PROPERTIES
        OUTPUT_NAME "localvqe"
        VERSION 0.1.0
        SOVERSION 0
    )

    # Hide all symbols except those marked LOCALVQE_API
    if(NOT WIN32)
        target_compile_options(localvqe_shared PRIVATE -fvisibility=hidden)
    endif()

    install(TARGETS localvqe_shared LIBRARY DESTINATION lib)
    install(FILES localvqe_api.h DESTINATION include)
endif()

# Per-block test binaries (self-contained, dvqe_common only)
foreach(block fe encoder bottleneck decoder ccm align)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_${block}.cpp")
        add_executable(test_${block} test_${block}.cpp)
        target_link_libraries(test_${block} PRIVATE dvqe_common)
    endif()
endforeach()

# GTCRN-AEC backend (standalone arch; distinct from the LocalVQE graph).
# Weights from train/export_gtcrn_ggml.py; parity-tested + benched via test_gtcrn.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gtcrn/gtcrn.cpp")
    add_library(dvqe_gtcrn STATIC gtcrn/gtcrn.cpp gtcrn/gtcrn_ggml.cpp)
    target_link_libraries(dvqe_gtcrn PUBLIC dvqe_common)
    target_include_directories(dvqe_gtcrn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gtcrn)
    add_executable(test_gtcrn gtcrn/test_gtcrn.cpp)
    target_link_libraries(test_gtcrn PRIVATE dvqe_gtcrn)
endif()

# DCT matmul layout test (validates ggml_mul_mat vs analytic DCT-II)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_dct.cpp")
    add_executable(test_dct test_dct.cpp)
    target_link_libraries(test_dct PRIVATE dvqe_graph)
endif()

# Streaming/batch equivalence test (uses C API + WAV I/O)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_streaming.cpp" AND SNDFILE_FOUND)
    add_executable(test_streaming test_streaming.cpp localvqe_api.cpp)
    target_link_libraries(test_streaming PRIVATE dvqe_graph dvqe_audio_io)
endif()

# Noise-gate unit test (header-only utility, no model needed)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_noise_gate.cpp")
    add_executable(test_noise_gate test_noise_gate.cpp)
endif()

# Shared HuggingFace-asset download infrastructure. Used by bench (model
# + WAVs) and the regression test (model only). Lives outside the
# SNDFILE_FOUND block so the regression target works on minimal CI
# images that don't ship libsndfile.
set(_assets_dir ${CMAKE_BINARY_DIR}/bench_assets)
set(_dl_script  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_asset.cmake)
set(_gguf_v1    ${_assets_dir}/localvqe-v1-1.3M-f32.gguf)
set(_gguf_v11   ${_assets_dir}/localvqe-v1.1-1.3M-f32.gguf)
set(_gguf_v12   ${_assets_dir}/localvqe-v1.2-1.3M-f32.gguf)
set(_gguf_v13   ${_assets_dir}/localvqe-v1.3-4.8M-f32.gguf)
set(_gguf_v14a  ${_assets_dir}/localvqe-v1.4-aec-200K-f32.gguf)
set(_gguf_v14ab ${_assets_dir}/localvqe-v1.4-aec-200K-bf16.gguf)
set(_gguf_v14fe ${_assets_dir}/localvqe-v1.4-aec-2.7K-f32.gguf)
file(MAKE_DIRECTORY ${_assets_dir})

function(_dvqe_download OUTFILE URL SHA COMMENT_TEXT)
    add_custom_command(OUTPUT ${OUTFILE}
        COMMAND ${CMAKE_COMMAND}
            -DURL=${URL}
            -DOUTFILE=${OUTFILE}
            -DEXPECTED_SHA256=${SHA}
            -P ${_dl_script}
        COMMENT "${COMMENT_TEXT}"
        VERBATIM)
endfunction()

_dvqe_download(${_gguf_v1}
    https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1-1.3M-f32.gguf
    d5eaf577449d0f920d8ee5e1042b8ddc7b6627313a042c62e2ada1b42719ab30
    "Fetching LocalVQE v1 F32 model from HuggingFace")
_dvqe_download(${_gguf_v11}
    https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.1-1.3M-f32.gguf
    c118227c6b433d6aa36d9e4b993e0f31aa60787ea38d301d04db917a4a2b0a84
    "Fetching LocalVQE v1.1 F32 model from HuggingFace")
_dvqe_download(${_gguf_v12}
    https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.2-1.3M-f32.gguf
    4856ecf5f522b23fb2bc5caeac81f323c0ef1c4c156a9c7d40a6adbe092ba9ce
    "Fetching LocalVQE v1.2 F32 model from HuggingFace")
_dvqe_download(${_gguf_v13}
    https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.3-4.8M-f32.gguf
    c4f7912485c32cfc206c536f2f050b52513f2f613fdbc616391f6b26ab1d51ec
    "Fetching LocalVQE v1.3 F32 model from HuggingFace")
_dvqe_download(${_gguf_v14a}
    https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.4-aec-200K-f32.gguf
    b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731
    "Fetching LocalVQE v1.4-AEC F32 model from HuggingFace")
_dvqe_download(${_gguf_v14ab}
    https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.4-aec-200K-bf16.gguf
    c14899f4dec79e1bc3360ebb9845621bd32b81d7c9a6c1ee9e56fc1d7b1978a6
    "Fetching LocalVQE v1.4-AEC BF16 model from HuggingFace")
_dvqe_download(${_gguf_v14fe}
    https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.4-aec-2.7K-f32.gguf
    d79f824f6ee6f58b2a7108d1df90db51c42c15a7400a5addbd3be9972ba657a4
    "Fetching LocalVQE v1.4-AEC 2.7K front-end-only model from HuggingFace")

# Benchmark (uses C API + WAV I/O)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench.cpp" AND SNDFILE_FOUND)
    add_executable(bench bench.cpp localvqe_api.cpp)
    target_link_libraries(bench PRIVATE dvqe_graph dvqe_audio_io)

    # `bench-run` — build bench, fetch the released F32 model + a doubletalk
    # mic/ref WAV pair from HuggingFace, and run the benchmark. Intended for
    # turnkey use on a fresh machine.
    set(_bench_mic ${_assets_dir}/dt_mic.wav)
    set(_bench_ref ${_assets_dir}/dt_ref.wav)

    _dvqe_download(${_bench_mic}
        https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo/resolve/main/examples/dt_mic.wav
        8d0c15e56a4c7f387847451952123e8f23161520e81c1ba8878c922f11592a62
        "Fetching doubletalk mic WAV")
    _dvqe_download(${_bench_ref}
        https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo/resolve/main/examples/dt_ref.wav
        e931d49e2802c1d7b15750caff1455495cd1dc40d27cb2fe51a8c58cbebe60fe
        "Fetching doubletalk reference WAV")

    add_custom_target(bench-assets
        DEPENDS ${_gguf_v12} ${_bench_mic} ${_bench_ref})

    set(BENCH_ITERS 10 CACHE STRING "Iterations for the bench-run target")
    set(BENCH_BACKEND "CPU"
        CACHE STRING "Backend for bench-run (e.g. CPU, Vulkan, CUDA)")
    set(BENCH_DEVICE  "0"
        CACHE STRING "Device index for bench-run (per-backend, see bench-list-devices)")
    add_custom_target(bench-run
        COMMAND $<TARGET_FILE:bench>
            ${_gguf_v12}
            --backend ${BENCH_BACKEND}
            --device  ${BENCH_DEVICE}
            --in-wav ${_bench_mic} ${_bench_ref}
            --iters ${BENCH_ITERS}
            --profile
        DEPENDS bench bench-assets
        WORKING_DIRECTORY ${_assets_dir}
        USES_TERMINAL
        VERBATIM)

    add_custom_target(bench-list-devices
        COMMAND $<TARGET_FILE:bench> --list-devices
        DEPENDS bench
        WORKING_DIRECTORY ${_assets_dir}
        USES_TERMINAL
        VERBATIM)
endif()

# Minimal conv2d test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_conv2d_minimal.cpp")
    add_executable(test_conv2d_minimal test_conv2d_minimal.cpp)
    target_link_libraries(test_conv2d_minimal PRIVATE ggml)
endif()

# Quantize tool (F32 GGUF → Q4_K/Q8_0/etc.)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/quantize.cpp")
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PRIVATE ggml)
endif()

# GGML graph evaluation (uses C API + WAV I/O)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eval_graph.cpp" AND SNDFILE_FOUND)
    add_executable(eval_graph eval_graph.cpp localvqe_api.cpp)
    target_link_libraries(eval_graph PRIVATE dvqe_graph dvqe_audio_io)
endif()

# End-to-end regression test (no audio file deps — uses raw .f32 fixtures
# in tests/fixtures/). SKIPS at runtime when the .gguf isn't present, so
# CTest is happy even without assets downloaded. The `regression-assets`
# target fetches both GGUFs from HuggingFace; after that, ctest runs
# both entries.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_regression.cpp")
    enable_testing()
    add_executable(test_regression tests/test_regression.cpp localvqe_api.cpp)
    target_link_libraries(test_regression PRIVATE dvqe_graph)

    add_custom_target(regression-assets
        DEPENDS ${_gguf_v1} ${_gguf_v11} ${_gguf_v12} ${_gguf_v13}
                ${_gguf_v14a} ${_gguf_v14ab} ${_gguf_v14fe})

    set(_reg_fix    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
    set(_reg_input  ${_reg_fix}/regression_input.f32)

    # One entry per released GGUF, spelled as the full file stem — the
    # fixture is <stem>.out.f32. (v1.3 broke the "1.3M everywhere" size
    # assumption; v1.4-AEC adds a dtype variant, so the stem now carries
    # everything.)
    set(_reg_stems
        "localvqe-v1-1.3M-f32"
        "localvqe-v1.1-1.3M-f32"
        "localvqe-v1.2-1.3M-f32"
        "localvqe-v1.3-4.8M-f32"
        "localvqe-v1.4-aec-200K-f32"
        "localvqe-v1.4-aec-200K-bf16"
        "localvqe-v1.4-aec-2.7K-f32")
    foreach(_stem IN LISTS _reg_stems)
        add_test(NAME regression_${_stem}
                 COMMAND test_regression
                     --gguf-name ${_stem}.gguf
                     --input    ${_reg_input}
                     --expected ${_reg_fix}/${_stem}.out.f32)
        set_tests_properties(regression_${_stem} PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "LOCALVQE_GGUF_DIR=${_assets_dir}")
    endforeach()

    # Prime-coverage test. The standalone front-end's whole-signal delay
    # priming only acts on inputs longer than the 1.024 s GCC window, so the
    # 1 s fixtures above leave it a no-op. This uses a 3 s echo+delay fixture
    # (regression_input_prime.f32 — far-end ref with a known 3000-sample echo)
    # so daf_prime_delay locks the bulk delay and aligns the filter from
    # frame 0; the committed output pins that path.
    add_test(NAME regression_prime_localvqe-v1.4-aec-2.7K-f32
             COMMAND test_regression
                 --gguf-name localvqe-v1.4-aec-2.7K-f32.gguf
                 --input    ${_reg_fix}/regression_input_prime.f32
                 --expected ${_reg_fix}/localvqe-v1.4-aec-2.7K-f32.prime.out.f32)
    set_tests_properties(regression_prime_localvqe-v1.4-aec-2.7K-f32 PROPERTIES
        SKIP_RETURN_CODE 77
        ENVIRONMENT "LOCALVQE_GGUF_DIR=${_assets_dir}")
endif()

# Fuzz reproducer regression tests. Pin specific bugs found by the
# libFuzzer harnesses so they don't silently regress.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fuzz_repros.cpp")
    enable_testing()
    add_executable(test_fuzz_repros tests/test_fuzz_repros.cpp localvqe_api.cpp)
    target_link_libraries(test_fuzz_repros PRIVATE dvqe_graph)

    # Bug: process_frame_graph memcpy overlap. Exercises the streaming
    # graph for one frame; meaningful in any build, definitive under ASan.
    add_test(NAME fuzz_repro_frame_roundtrip
             COMMAND test_fuzz_repros frame-roundtrip
                 --gguf-name localvqe-v1.2-1.3M-f32.gguf)
    set_tests_properties(fuzz_repro_frame_roundtrip PROPERTIES
        SKIP_RETURN_CODE 77
        ENVIRONMENT "LOCALVQE_GGUF_DIR=${_assets_dir}")

    # SHA-256 allowlist: non-released GGUFs must be rejected before
    # reaching the (unhardened) GGUF parser. No assets / env needed.
    add_test(NAME fuzz_repro_reject_unhashed
             COMMAND test_fuzz_repros reject-unhashed)
endif()

# ── libFuzzer harnesses ──────────────────────────────────────────────────────
#
# Two targets:
#   fuzz_audio       — drives the audio-processing API with a real model loaded
#   fuzz_model_load  — feeds attacker-controlled bytes to the GGUF loader
#
# Both link statically against dvqe_graph and are instrumented with libFuzzer
# + sanitizers. See ggml/fuzz/README.md for the run recipe.
if(LOCALVQE_FUZZ)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR
            "LOCALVQE_FUZZ requires clang (got ${CMAKE_CXX_COMPILER_ID}). "
            "Configure with -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++.")
    endif()

    set(_fuzz_flags "-fsanitize=${LOCALVQE_FUZZ_SANITIZERS}")
    # Coverage instrumentation must reach the libraries the harness calls
    # into, not just the harness TU. We apply the flag to the whole build
    # tree by injecting it into CMAKE_*_FLAGS for this directory.
    add_compile_options(${_fuzz_flags} -fno-omit-frame-pointer -g)
    add_link_options(${_fuzz_flags})

    add_executable(fuzz_audio fuzz/fuzz_audio.cpp localvqe_api.cpp)
    target_link_libraries(fuzz_audio PRIVATE dvqe_graph)

    # fuzz_native — drives the hand-rolled DAF + native engine directly
    add_executable(fuzz_native fuzz/fuzz_native.cpp native_engine.cpp)
    target_link_libraries(fuzz_native PRIVATE dvqe_model dvqe_graph)
endif()

# Per-layer tap tool for PyTorch parity debugging (port verification)
add_executable(tap_debug tap_debug.cpp)
target_link_libraries(tap_debug PRIVATE dvqe_graph)

# DAF front-end (stage 1 of the v1.4-AEC cascade) + standalone CLI.
# The front-end is an adaptive filter — host-side float DSP whose
# trajectory amplifies any codegen-level FP difference — so pin FP
# contraction off to keep its output stable across optimization levels
# and reproducible against the committed regression fixture.
set_source_files_properties(daf_frontend.cpp PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off")
target_sources(dvqe_graph PRIVATE daf_frontend.cpp)
add_executable(daf_cli daf_cli.cpp)
target_link_libraries(daf_cli PRIVATE dvqe_graph)

# Developer-only: native-ISA codegen for the hand-rolled DAF front-end
# (the ggml libs keep their per-ISA runtime dispatch regardless). OFF by
# default — it makes every binary linking dvqe_graph Zen4-only and does
# not even compile on non-x86 hosts. Opt in for local benchmarking.
option(LOCALVQE_DAF_NATIVE "Compile the dvqe_graph lib with -march=znver4 (developer benchmarking only)" OFF)
if (LOCALVQE_DAF_NATIVE)
    # NOTE: per-source COMPILE_OPTIONS silently failed (and Nix strips
    # -march=native); target-level explicit arch.
    target_compile_options(dvqe_graph PRIVATE -march=znver4 -funroll-loops)
endif()

# Developer-only: hand-rolled single-CPU streaming engine + harness. Same
# caveat as above — Zen4-only binaries; OFF by default.
option(LOCALVQE_NATIVE_ENGINE "Build the -march=znver4 native streaming engine (developer benchmarking only)" OFF)
if (LOCALVQE_NATIVE_ENGINE)
    add_executable(native_test native_test.cpp native_engine.cpp)
    target_link_libraries(native_test PRIVATE dvqe_model dvqe_graph)
    # NOTE: set_source_files_properties COMPILE_OPTIONS silently failed to
    # apply here (zero ymm/zmm in the binary); use target-level options.
    target_compile_options(native_test PRIVATE -march=znver4 -funroll-loops)

    # Same harness with the AVX512BF16 conv kernels masked off, so the
    # f32 (AVX512F 4-chain) kernel path keeps its own regression fixture.
    add_executable(native_test_f32 native_test.cpp native_engine.cpp)
    target_link_libraries(native_test_f32 PRIVATE dvqe_model dvqe_graph)
    target_compile_options(native_test_f32 PRIVATE -march=znver4 -mno-avx512bf16 -funroll-loops)
endif()
