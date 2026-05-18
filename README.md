# dsv4-workspace

Workspace for building and serving **DeepSeek-V4-Flash** on **RTX PRO 6000
Blackwell** (SM120) GPUs. Wraps three component repos behind a single
Docker image.

- **Pre-built image:** [`lucifer1004/dsv4-flash-sm120`](https://hub.docker.com/r/lucifer1004/dsv4-flash-sm120) on Docker Hub
- **Validated on:** RTX PRO 6000 Blackwell (SM120), driver 590.48.01, NGC PyTorch 26.04
- **GSM8K 5-shot, 200q:** 0.915

## Components

| Repo | Branch | Commit | Role |
|---|---|---|---|
| [lucifer1004/vllm](https://github.com/lucifer1004/vllm) | `dsv4-sm120` | `300332deb` | vLLM fork with DSv4 patches + prefill-metadata IMA clamp |
| [lucifer1004/DeepGEMM](https://github.com/lucifer1004/DeepGEMM) | `sm120` | `aa960cd` | SM120-specific FP8/MXFP4 GEMM kernels (AB-swap path) |
| [lucifer1004/sparse_mla_sm120](https://github.com/lucifer1004/sparse_mla_sm120) | `master` | `a003afc` | KV-cache padded-stride fix + sparse-MLA kernels |

## Use the pre-built image (fastest)

```sh
docker run --rm --gpus all --shm-size=8g \
    -v /path/to/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash \
    -e MODEL_PATH=/models/DeepSeek-V4-Flash \
    -p 8000:8000 \
    lucifer1004/dsv4-flash-sm120:latest \
    --tensor-parallel-size 2
```

See [`docker/README.md`](docker/README.md) for the full run reference,
troubleshooting, and the open-driver 595.58.03+ NCCL workaround.

## Build the image yourself

```sh
git clone https://github.com/lucifer1004/dsv4-workspace
cd dsv4-workspace
./scripts/clone.sh                          # clones the 3 component repos at pinned commits
docker build -f docker/Dockerfile -t dsv4-flash-sm120:latest .
```

Build takes ~30 min the first time (vLLM's CUDA extension build is the
long pole). The `clone.sh` script accepts `--ssh` for SSH clones.

## Licensing

Each component repo carries its own license — see those repos. The
Dockerfile, entrypoint, README, and clone script in this meta-repo are
released under Apache 2.0 (see [`LICENSE`](LICENSE)).
