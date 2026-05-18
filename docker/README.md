# DSv4-Flash on SM120 — Docker image

vLLM serve image for **DeepSeek-V4-Flash** on **RTX PRO 6000 Blackwell** (SM120).
Bundles the dsv4-sm120 vllm fork + DeepGEMM sm120 + sparse_mla_sm120 with all
CUDA extensions pre-built against NGC PyTorch 26.04.

## Run

The model is **not** baked in — mount it from the host:

```sh
docker run --rm --gpus all --shm-size=8g \
    -v /path/to/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash \
    -e MODEL_PATH=/models/DeepSeek-V4-Flash \
    -p 8000:8000 \
    lucifer1004/dsv4-flash-sm120:latest \
    --tensor-parallel-size 2
```

Required:
- `--gpus all` (or `--gpus '"device=0,1"'`) with NVIDIA Container Toolkit
- `--shm-size=8g` when `--tensor-parallel-size > 1` (default 64 MB causes NCCL init to fail)
- `MODEL_PATH` env pointing at the DSv4-Flash directory inside the container
- `--tensor-parallel-size N` — must divide visible GPU count (no default)

Anything after the image name is forwarded to `vllm serve`. The entrypoint sets
the dsv4-sm120 reference defaults (FP8 KV, block-size 256, expert parallel,
FULL_AND_PIECEWISE cudagraph, `--max-model-len 65536`, etc.); override any by
re-passing the flag on the CLI.

### Persist JIT caches across restarts

```sh
-v $(pwd)/.dsv4-cache:/cache
```

`XDG_CACHE_HOME=/cache` — DeepGEMM, Triton and torch.compile write under it.

## Host requirements

- 1+ RTX PRO 6000 Blackwell GPU (SM120)
- NVIDIA driver compatible with CUDA 13 / NGC PyTorch 26.04 (R570+)
- NVIDIA Container Toolkit

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NCCL error: unhandled system error` at init (TP > 1) | Add `--shm-size=8g` |
| Boot hangs after `vLLM is using nccl==...`, GPUs at 100% util but ~949 MiB used | Open-driver 595.58.03+ regression. Add `-e NCCL_P2P_USE_CUDA_MEMCPY=1` |
| Boot hangs ≥10 min in `profile_run`, GPUs at 100% util but only ~100 W power | The above workaround was set on a driver that doesn't need it. Unset `NCCL_P2P_USE_CUDA_MEMCPY` |
| `Engine core initialization failed` | Likely OOM — lower `--gpu-memory-utilization` (e.g. 0.85) or `--max-model-len` |
| First request slow (~30 s extra) | DeepGEMM autotune. Persist `/cache` (see above) |

## Sources

- vLLM fork: dsv4-sm120 branch
- DeepGEMM: sm120 branch
- sparse_mla_sm120: master

Build the image yourself from
`https://github.com/lucifer1004/dsv4-workspace` (Dockerfile under `docker/`).
