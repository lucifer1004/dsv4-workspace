# https://just.systems

dsv4 := "deepseek-ai/DeepSeek-V4-Flash"
sm120 := "12.0"
sm120_cudaarchs := "120"

default:
    echo 'Hello, world!'

[script("bash")]
build:
    source .venv/bin/activate
    export DEEPGEMM_SRC_DIR="$PWD/DeepGEMM"
    # vllm's setup.py copytree of cmake's deep_gemm vendor tree into the editable
    # source tree raises SameFileError on re-builds: both src (build_lib) and dst
    # (source dir) have `cute`/`cutlass` symlinks pointing to the same DeepGEMM
    # cutlass include tree, and shutil.samefile() returns True for every header
    # reached through them. Pre-clean those two symlinks so copytree starts fresh
    # for those subtrees; cmake recreates them on each build.
    rm -f vllm/vllm/third_party/deep_gemm/include/cute \
          vllm/vllm/third_party/deep_gemm/include/cutlass
    uv pip install -e vllm

# Fast rebuild of just DeepGEMM's C++ extension (`_C.so`) and refresh the
# vendored copy inside vllm/. Use this when only files under DeepGEMM/csrc/
# have changed — much faster than `just build` (which also rebuilds vllm's
# own C++ extensions). The JIT-compiled kernel headers under
# DeepGEMM/deep_gemm/include/ don't go through this build path; they are
# copied verbatim into the vendored tree.
[script("bash")]
build-deepgemm:
    set -euo pipefail
    source .venv/bin/activate
    cd DeepGEMM
    # Ensure CUTLASS include symlinks exist (matches develop.sh)
    ln -sf "$PWD/third-party/cutlass/include/cutlass" deep_gemm/include/ 2>/dev/null || true
    ln -sf "$PWD/third-party/cutlass/include/cute"    deep_gemm/include/ 2>/dev/null || true
    # Force-clean build/: setup.py doesn't track header dependencies, so an
    # edit to runtime.hpp / sm120.hpp etc. won't trigger a recompile of
    # python_api.o without removing build/ first.
    rm -rf build
    # Build only the _C.so extension
    python setup.py build
    # Locate the freshly built _C.so
    so_file=$(find build -name "_C*.so" -type f | head -n 1)
    if [ -z "$so_file" ]; then
        echo "ERROR: _C.so not produced under DeepGEMM/build/" >&2
        exit 1
    fi
    # Refresh the vendored copy inside vllm so vllm.third_party.deep_gemm
    # picks up the new binary on next import.
    target="../vllm/vllm/third_party/deep_gemm/$(basename "$so_file")"
    cp -v "$so_file" "$target"
    # Sync the JIT-include headers too in case any .cuh was edited.
    # (cp -r is portable; rsync isn't always installed.)
    rm -rf ../vllm/vllm/third_party/deep_gemm/include
    cp -r deep_gemm/include ../vllm/vllm/third_party/deep_gemm/include
    echo "build-deepgemm: ok — $target updated"

[script("bash")]
run:
    source .venv/bin/activate
    vllm serve {{dsv4}} \
        --trust-remote-code \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.95 \
        --max-model-len 65536 \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
        --no-enable-flashinfer-autotune

[script("bash")]
run-bak:
    source .venv/bin/activate
    vllm serve {{dsv4}} \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --block-size 256 \
        --tensor-parallel-size 2 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.95 \
        --max-model-len 65536 \
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
        --tokenizer-mode deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --enable-auto-tool-choice \
        --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
        --no-enable-flashinfer-autotune
