cmake_minimum_required(VERSION 3.20)
project(localvqe-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LOCALVQE_BUILD_SHARED "Build shared library with C API" OFF)
option(LOCALVQE_CUDA "Enable GGML CUDA backend" OFF)
option(LOCALVQE_VULKAN "Enable GGML Vulkan backend" OFF)

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
else()
    # Build all CPU backend variants (SSE4.2 through AVX-512/Zen4) as shared
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
endif()
if(LOCALVQE_VULKAN)
    # Composes with the CPU-DL path above: Vulkan becomes a separate
    # libggml-vulkan.so in build/bin/ that ggml_backend_load_all() picks up
    # at runtime, falling back to the CPU variants when no Vulkan ICD is found.
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(vendor/ggml)

find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(SNDFILE sndfile)
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
add_library(dvqe_graph STATIC localvqe_graph.cpp)
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

# Benchmark (uses C API + WAV I/O)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench.cpp" AND SNDFILE_FOUND)
    add_executable(bench bench.cpp localvqe_api.cpp)
    target_link_libraries(bench PRIVATE dvqe_graph dvqe_audio_io)

    # `bench-run` — build bench, fetch the released F32 model + a doubletalk
    # mic/ref WAV pair from HuggingFace, and run the benchmark. Intended for
    # turnkey use on a fresh machine.
    set(_bench_dir   ${CMAKE_BINARY_DIR}/bench_assets)
    set(_bench_model ${_bench_dir}/localvqe-v1-1.3M-f32.gguf)
    set(_bench_mic   ${_bench_dir}/dt_mic.wav)
    set(_bench_ref   ${_bench_dir}/dt_ref.wav)
    set(_dl_script   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_asset.cmake)

    file(MAKE_DIRECTORY ${_bench_dir})

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

    _dvqe_download(${_bench_model}
        https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1-1.3M-f32.gguf
        d5eaf577449d0f920d8ee5e1042b8ddc7b6627313a042c62e2ada1b42719ab30
        "Fetching LocalVQE F32 model from HuggingFace")
    _dvqe_download(${_bench_mic}
        https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo/resolve/main/examples/dt_mic.wav
        8d0c15e56a4c7f387847451952123e8f23161520e81c1ba8878c922f11592a62
        "Fetching doubletalk mic WAV")
    _dvqe_download(${_bench_ref}
        https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo/resolve/main/examples/dt_ref.wav
        e931d49e2802c1d7b15750caff1455495cd1dc40d27cb2fe51a8c58cbebe60fe
        "Fetching doubletalk reference WAV")

    add_custom_target(bench-assets
        DEPENDS ${_bench_model} ${_bench_mic} ${_bench_ref})

    set(BENCH_ITERS 10 CACHE STRING "Iterations for the bench-run target")
    set(BENCH_BACKEND "CPU"
        CACHE STRING "Backend for bench-run (e.g. CPU, Vulkan, CUDA)")
    set(BENCH_DEVICE  "0"
        CACHE STRING "Device index for bench-run (per-backend, see bench-list-devices)")
    add_custom_target(bench-run
        COMMAND $<TARGET_FILE:bench>
            ${_bench_model}
            --backend ${BENCH_BACKEND}
            --device  ${BENCH_DEVICE}
            --in-wav ${_bench_mic} ${_bench_ref}
            --iters ${BENCH_ITERS}
            --profile
        DEPENDS bench bench-assets
        WORKING_DIRECTORY ${_bench_dir}
        USES_TERMINAL
        VERBATIM)

    add_custom_target(bench-list-devices
        COMMAND $<TARGET_FILE:bench> --list-devices
        DEPENDS bench
        WORKING_DIRECTORY ${_bench_dir}
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
