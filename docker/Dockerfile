# DSv4-Flash on SM120 (RTX PRO 6000 Blackwell) — vLLM serve image.
#
# This is a multi-stage image. The builder stage compiles all CUDA / C++
# extensions in editable mode against the NGC PyTorch base, and the
# runtime stage copies the venv + source trees into a fresh layer to
# keep the published image smaller than the builder's intermediate
# build trees (sccache, .deps build dirs, ninja state, etc.).
#
# Build context (relative to this Dockerfile, i.e. workspace root):
#   docker build -f docker/Dockerfile -t dsv4-flash-sm120 .
#
# Run (host must have RTX PRO 6000 Blackwell GPUs visible):
#   docker run --rm --gpus all \
#     -v /path/to/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash \
#     -e MODEL_PATH=/models/DeepSeek-V4-Flash \
#     -p 8000:8000 \
#     dsv4-flash-sm120 \
#     --tensor-parallel-size 2
#
# See docker/README.md for the host clock-lock requirement.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
ARG WORKSPACE=/workspace/dsv4-workspace
ARG VENV_PATH=/workspace/dsv4-workspace/.venv

# ============================================================================
# Stage 1: builder
# ============================================================================
FROM ${BASE_IMAGE} AS builder

ARG WORKSPACE
ARG VENV_PATH

# NGC PyTorch 26.04 already ships with `uv` on PATH (verified at build
# time). If a future base image doesn't, install it; otherwise no-op.
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh \
        && ln -sf /root/.local/bin/uv /usr/local/bin/uv ; \
    fi \
    && uv --version

WORKDIR ${WORKSPACE}

# Copy the three repos. .dockerignore strips out .git/objects (we use the
# committed source state, not the history), build/, .venv, JIT caches.
COPY vllm/             ${WORKSPACE}/vllm/
COPY DeepGEMM/         ${WORKSPACE}/DeepGEMM/
COPY sparse_mla_sm120/ ${WORKSPACE}/sparse_mla_sm120/

# Pre-create venv from the python in the base image.
RUN uv venv ${VENV_PATH} --python 3.12

# Build vllm editable. Same recipe as `just build` PLUS extra cleanup
# for cold docker builds: the host's vllm/vllm/third_party/deep_gemm/
# directory was populated by a previous host-side install and contains
# `cute` / `cutlass` symlinks pointing into the host filesystem (not
# valid inside the container). It can also contain a populated
# `deep_gemm/` subtree from an earlier copytree. CMake's INSTALL step
# fails with "File exists" if any of these are present when the build
# tries to recreate them. Remove the entire `include/` tree (other
# than what cmake will recreate) before running pip install.
ENV DEEPGEMM_SRC_DIR=${WORKSPACE}/DeepGEMM
RUN . ${VENV_PATH}/bin/activate \
    && rm -rf vllm/vllm/third_party/deep_gemm/include \
    && uv pip install -e vllm

# Build DeepGEMM's _C.so against the just-installed torch, then refresh
# the vendored copy under vllm/vllm/third_party/deep_gemm/. Same recipe
# as `just build-deepgemm`.
RUN . ${VENV_PATH}/bin/activate \
    && cd ${WORKSPACE}/DeepGEMM \
    && ln -sf "$PWD/third-party/cutlass/include/cutlass" deep_gemm/include/ \
    && ln -sf "$PWD/third-party/cutlass/include/cute"    deep_gemm/include/ \
    && rm -rf build \
    && python setup.py build \
    && so_file=$(find build -name "_C*.so" -type f | head -n 1) \
    && cp "$so_file" "../vllm/vllm/third_party/deep_gemm/$(basename $so_file)" \
    && rm -rf ../vllm/vllm/third_party/deep_gemm/include \
    && cp -r deep_gemm/include ../vllm/vllm/third_party/deep_gemm/include

# Sync DeepGEMM Python wrappers (the vllm setup.py copytree only runs at
# initial install; the build-deepgemm step above only refreshes .so +
# include/. The Python wrappers — __init__.py, utils/, legacy/, testing/,
# mega/ — must be re-synced so symbols added on the sm120 branch
# (e.g. set_mk_alignment_for_contiguous_layout) are reachable from
# `vllm.third_party.deep_gemm`).
RUN cd ${WORKSPACE} \
    && cp DeepGEMM/deep_gemm/__init__.py vllm/vllm/third_party/deep_gemm/__init__.py \
    && for d in utils legacy testing mega; do \
         if [ -d DeepGEMM/deep_gemm/$d ]; then \
           rm -rf vllm/vllm/third_party/deep_gemm/$d \
           && cp -r DeepGEMM/deep_gemm/$d vllm/vllm/third_party/deep_gemm/$d ; \
         fi ; \
       done

# Build sparse_mla_sm120 (out-of-tree CUDA extension; editable install).
RUN . ${VENV_PATH}/bin/activate \
    && uv pip install -e sparse_mla_sm120/ --no-deps --no-build-isolation

# Sanity check: imports work.
RUN . ${VENV_PATH}/bin/activate && cd /tmp \
    && python -c "import vllm, flash_mla_sm120 ; \
                   import vllm.third_party.deep_gemm as dg ; \
                   assert hasattr(dg, 'set_mk_alignment_for_contiguous_layout'), \
                          'deep_gemm: missing set_mk_alignment_for_contiguous_layout'; \
                   print('vllm', vllm.__version__) ; \
                   print('flash_mla_sm120 ok') ; \
                   print('deep_gemm ok')"

# ============================================================================
# Stage 2: runtime
# ============================================================================
FROM ${BASE_IMAGE} AS runtime

ARG WORKSPACE
ARG VENV_PATH

# uv is part of the NGC PyTorch 26.04 base image; no copy needed.

# Copy the venv and source repos from the builder. We need both because
# vllm + sparse_mla_sm120 + DeepGEMM are editable installs (the .pth
# files in site-packages point back to the source trees).
COPY --from=builder ${VENV_PATH} ${VENV_PATH}
COPY --from=builder ${WORKSPACE}/vllm ${WORKSPACE}/vllm
COPY --from=builder ${WORKSPACE}/DeepGEMM ${WORKSPACE}/DeepGEMM
COPY --from=builder ${WORKSPACE}/sparse_mla_sm120 ${WORKSPACE}/sparse_mla_sm120

# Entrypoint script.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Activate the venv on every container shell (PATH first, then PYTHONHOME
# left untouched per uv venv layout). We do NOT set PYTHONPATH because
# host-style leaks bit us on bare-metal; venv .pth files handle imports.
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV VIRTUAL_ENV="${VENV_PATH}"
ENV WORKSPACE="${WORKSPACE}"

# Default JIT cache locations inside the container (overridable). Persist
# these via -v on the host if you want to amortize JIT cost across runs.
ENV XDG_CACHE_HOME="/cache"

# vllm's --serve port.
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Default CMD is empty; user provides at least --tensor-parallel-size.
CMD []
