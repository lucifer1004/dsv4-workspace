#!/usr/bin/env bash
# DSv4-Flash on SM120 — container entrypoint.
#
# Required env: MODEL_PATH (path to DeepSeek-V4-Flash directory)
# Required CLI: --tensor-parallel-size N (must be passed; no default)
#
# Anything passed on the command line is forwarded verbatim to vllm serve,
# so callers can override block-size, kv-cache-dtype, max-model-len, etc.

set -uo pipefail

if [[ -z "${MODEL_PATH:-}" ]]; then
  cat >&2 <<EOF
ERROR: MODEL_PATH environment variable is not set.

The container does not bundle the DeepSeek-V4-Flash model. Mount the
model on the host and point MODEL_PATH at it, e.g.:

  docker run --rm --gpus all \\
    -v /path/to/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash \\
    -e MODEL_PATH=/models/DeepSeek-V4-Flash \\
    -p 8000:8000 \\
    dsv4-flash-sm120 \\
    --tensor-parallel-size 2

EOF
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH=${MODEL_PATH} does not exist or is not a directory." >&2
  exit 2
fi

# Require --tensor-parallel-size to be in CLI (matches the design decision
# to never silently guess GPU count).
if ! printf '%s\n' "$@" | grep -q -- '--tensor-parallel-size'; then
  cat >&2 <<EOF
ERROR: --tensor-parallel-size is required.

Pick a value that divides your visible GPU count (use \`docker run --gpus
'device=0,1'\` to constrain). Typical values: 2, 4, 8.

Example:
  docker run --rm --gpus all \\
    -e MODEL_PATH=/models/DeepSeek-V4-Flash \\
    dsv4-flash-sm120 \\
    --tensor-parallel-size 2

EOF
  exit 2
fi

# Unset PYTHONPATH defensively — host-style leaks have caused real
# issues with this stack (see project_dsv4_sm120_correctness.md).
unset PYTHONPATH

# NCCL P2P: we do NOT default NCCL_P2P_USE_CUDA_MEMCPY here. It is a
# targeted workaround for the open-driver 595.58.03+ AllReduce hang and
# is NOT harmless on other drivers: on driver 590.48.01 (proprietary) it
# caused a deep_gemm tf32_hc_prenorm_gemm kernel livelock during
# vllm profile_run (19+ min stuck, ~100W per GPU vs expected ~500W).
#
# If your host is on driver 595.58.03+ open kernel module and you see a
# hang immediately after the NCCL init line, pass:
#   docker run ... -e NCCL_P2P_USE_CUDA_MEMCPY=1 ...
# See docker/README.md → Troubleshooting for the symptom matrix.

# Defaults that match the dsv4-sm120 reference config. Each can be
# overridden by passing the same flag on the CLI (vllm respects last
# occurrence).
exec vllm serve "${MODEL_PATH}" \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
    --no-enable-flashinfer-autotune \
    "$@"
