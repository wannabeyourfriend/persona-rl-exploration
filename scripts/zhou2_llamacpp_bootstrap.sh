#!/usr/bin/env bash
# zhou-2 (glibc 2.27): bootstrap CUDA llama.cpp via micromamba (static binary,
# avoids the Miniconda installer's glibc>=2.28 requirement). conda-forge pkgs
# target glibc 2.17 so they run here. Everything under $HOME.
# NOTE: glibc 2.27 caps the usable build at 6800 (cuda120); newer builds need
# libcublas 12.9 -> glibc 2.28. Build 6800 predates qwen35 arch, so use Qwen3-32B.
set -uo pipefail
cd "$HOME"
export TMPDIR="$HOME/.tmp"; mkdir -p "$TMPDIR" "$HOME/.local/bin"
MM="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export CONDA_OVERRIDE_CUDA=12   # tell solver the (driver) CUDA is 12.x

if [ ! -x "$MM" ]; then
  echo "[$(date)] === download micromamba (micro.mamba.pm, slow but reachable) ==="
  curl -fL --retry 6 --retry-delay 3 -m 300 -o "$HOME/mm.tar.bz2" \
    https://micro.mamba.pm/api/micromamba/linux-64/latest || { echo "MM_DL_FAILED"; exit 1; }
  mkdir -p "$HOME/.local"
  tar -xjf "$HOME/mm.tar.bz2" -C "$HOME/.local" bin/micromamba || { echo "MM_EXTRACT_FAILED"; exit 1; }
  chmod +x "$MM"
fi
echo "[$(date)] === micromamba version ==="
"$MM" --version 2>&1 | head -2 || { echo "MM_RUN_FAILED_GLIBC"; exit 1; }

echo "[$(date)] === create env: conda-forge CUDA llama.cpp (tuna mirror) ==="
"$MM" create -y -p "$HOME/llamaenv" \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
  "llama.cpp=*=*cuda*" 2>&1 | tail -25 || echo "LLAMA_CUDA_CREATE_FAILED"

SRV="$HOME/llamaenv/bin/llama-server"
echo "[$(date)] === verify ==="
if [ -x "$SRV" ]; then
  echo "FOUND_LLAMA_SERVER"
  "$SRV" --version 2>&1 | head -6
  echo "--- ldd cuda ---"; ldd "$SRV" 2>/dev/null | grep -iE "cuda|cublas|ggml|not found" | head
  echo "BOOTSTRAP_DONE_CUDA"
else
  echo "NO_CUDA_LLAMA_SERVER"; ls "$HOME/llamaenv/bin" 2>/dev/null | grep -i llama | head
  echo "BOOTSTRAP_DONE_NOCUDA"
fi
