#!/usr/bin/env bash
# Run modern vLLM on zhou-2 (Ubuntu 18.04 / glibc 2.27) WITHOUT root.
#
# The trick: bundle glibc 2.31 and patchelf the env python's interpreter onto it.
#  - 2.31 >= 2.28  -> satisfies manylinux_2_28 wheels (torch/xformers).
#  - 2.31 <  2.34  -> libdl.so.2 / libpthread.so.0 still carry their symbols
#                     (glibc>=2.34 stubs them; torch wheels then segfault on
#                     dlopen@GLIBC_2.2.5). DO NOT use glibc 2.35.
#  - pip run under the patched python reports glibc 2.31, so it ACCEPTS the
#    manylinux_2_28 wheels (under the host's 2.27 it rejects them -> source
#    builds -> fail on the old gcc).
#  - Use DT_RPATH (--force-rpath), NOT LD_LIBRARY_PATH at runtime (see _env.sh).
#
# Prereqs: micromamba at ~/.local/bin/micromamba (see zhou2_llamacpp_bootstrap.sh)
#   plus: ~/.local/bin/micromamba install -n base -c conda-forge patchelf zstd
set -uo pipefail
cd "$HOME"
MM="$HOME/.local/bin/micromamba"
ENV="$HOME/micromamba/envs/vllm"
GLIBC="$HOME/glibc231"
PY="$ENV/bin/python3.11"
TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "== 1. env + recent libstdc++ (env ships none; torch needs GLIBCXX>=3.4.30) =="
"$MM" create -n vllm -y -c conda-forge python=3.11 pip setuptools wheel
"$MM" install -n vllm -y -c conda-forge libstdcxx-ng libgcc-ng

echo "== 2. bundle glibc 2.31 (Ubuntu 20.04 libc6; ar+xz, host dpkg can't read newer debs) =="
rm -rf "$GLIBC" "$HOME/glibc231_debs"; mkdir -p "$GLIBC" "$HOME/glibc231_debs"; cd "$HOME/glibc231_debs"
curl -fL -m 180 -o libc6.deb \
  "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/pool/main/g/glibc/libc6_2.31-0ubuntu9.18_amd64.deb"
rm -rf t; mkdir t; ( cd t; ar x ../libc6.deb
  if [ -f data.tar.zst ]; then "$HOME/micromamba/bin/zstd" -dq data.tar.zst -o data.tar
  elif [ -f data.tar.xz ]; then xz -dk data.tar.xz; fi
  tar xf data.tar -C "$GLIBC" )
cd "$HOME"
"$GLIBC/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2" --version | head -1

echo "== 3. patchelf env python onto the 2.31 loader (DT_RPATH, not LD_LIBRARY_PATH) =="
LD="$GLIBC/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
RP="$GLIBC/lib/x86_64-linux-gnu:$GLIBC/usr/lib/x86_64-linux-gnu:$ENV/lib"
cp -n "$PY" "$PY.orig"
"$HOME/micromamba/bin/patchelf" --set-interpreter "$LD" --force-rpath --set-rpath "$RP" "$PY"
"$PY" -c "import platform; print('glibc', platform.libc_ver()); assert platform.libc_ver()[1].startswith('2.31')"

echo "== 4. pip install vLLM (patched python -> manylinux_2_28 wheels accepted) =="
# vllm 0.8.5 = newest that resolves purely from Tuna (pins torch 2.6.0, mirrored;
# 0.9+ wants torch 2.7/2.8 absent from Tuna). Supports Qwen3.
"$PY" -m pip install -i "$TUNA" "vllm==0.8.5"
# vllm pulls transformers 5.x which breaks its tokenizer API -> pin to 4.51.x.
"$PY" -m pip install -i "$TUNA" "transformers==4.51.3" "tokenizers==0.21.1"

echo "== 5. patch vLLM's NVML device-name probe (crashes when CUDA_VISIBLE_DEVICES set) =="
VLLM_ENV="$ENV" "$PY" "$(dirname "$0")/zhou2_patch_vllm_cuda.py"

echo "== 6. verify (needs healthy GPUs; will report cuda False if a GPU is wedged) =="
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())"
"$PY" -c "import vllm; print('vllm', vllm.__version__)"
echo "DONE. Serve e.g.:"
echo "  CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \\"
echo "    scripts/zhou2_vllm_env.sh vllm serve <model> --tensor-parallel-size 2 \\"
echo "    --port 8000 --served-model-name qwen-judge --gpu-memory-utilization 0.9"
