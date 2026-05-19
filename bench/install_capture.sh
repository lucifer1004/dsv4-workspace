#!/usr/bin/env bash
# Install the auto-capture .pth file into the active venv's site-packages.
#
# After installation, `VLLM_CAPTURE_KERNEL_SHAPES=1` enables capture for any
# Python process started from the venv; default (unset / 0) is zero overhead.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
# .pth runs at Python startup. Inserting WORKSPACE permanently into sys.path
# would let `vllm/` (a subdir of WORKSPACE) shadow the real installed vllm
# package as a PEP-420 namespace — that's the namespace-package trap that
# bites every `from vllm import X` in serve_dsv4. Insert WORKSPACE only long
# enough to import `bench.capture._auto`, then remove it. `bench.__path__`
# is captured at import time, so later `import bench.kernels` still works
# without WORKSPACE on sys.path.
PTH_CONTENT="import sys; sys.path.insert(0, '${WORKSPACE}'); import bench.capture._auto; sys.path.remove('${WORKSPACE}')
"

# Resolve venv site-packages.
SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
if [[ -z "$SITE" || ! -d "$SITE" ]]; then
  echo "ERROR: could not locate site-packages (got: $SITE)" >&2
  exit 1
fi

PTH_PATH="$SITE/dsv4_bench_capture.pth"
printf '%s\n' "$PTH_CONTENT" > "$PTH_PATH"
echo "Wrote $PTH_PATH:"
sed 's/^/  /' "$PTH_PATH"
echo ""
echo "Enable with: export VLLM_CAPTURE_KERNEL_SHAPES=1"
