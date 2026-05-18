#!/usr/bin/env bash
# Clone the three component repos at the exact commits shipped in the
# latest lucifer1004/dsv4-flash-sm120 image. Run from the workspace root —
# afterwards the layout is:
#
#   ./vllm/             (lucifer1004/vllm @ 300332deb, dsv4-sm120 branch)
#   ./DeepGEMM/         (lucifer1004/DeepGEMM @ 243a8a1, sm120 branch)
#   ./sparse_mla_sm120/ (lucifer1004/sparse_mla_sm120 @ a003afc, master)
#
# Pass --ssh to clone via git@github.com instead of https.

set -euo pipefail

PROTO=https
for arg in "$@"; do
    case "$arg" in
        --ssh) PROTO=ssh ;;
        --https) PROTO=https ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

clone_at() {
    local name=$1 commit=$2
    local url
    if [[ "$PROTO" == "ssh" ]]; then
        url="git@github.com:lucifer1004/${name}.git"
    else
        url="https://github.com/lucifer1004/${name}.git"
    fi
    if [[ -d "$name" ]]; then
        echo "==> $name/ already exists — skipping clone, just checking out $commit"
        git -C "$name" fetch origin "$commit"
        git -C "$name" checkout "$commit"
    else
        echo "==> cloning $url"
        git clone "$url" "$name"
        git -C "$name" checkout "$commit"
    fi
    # vllm needs its own submodules initialised
    git -C "$name" submodule update --init --recursive
}

clone_at vllm              300332deb7d7dcffb662246bb9638f993601e2d4
clone_at DeepGEMM          243a8a16a59eb143aea100acba0ace3a9280dc2c
clone_at sparse_mla_sm120  a003afc186b471890d393df9f3e3ba6d665d1678

echo
echo "All three repos ready at the pinned commits."
echo "Next: build the image with"
echo "    docker build -f docker/Dockerfile -t dsv4-flash-sm120:latest ."
