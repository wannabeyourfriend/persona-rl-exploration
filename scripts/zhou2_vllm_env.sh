#!/usr/bin/env bash
# No-root vLLM launcher for zhou-2 (glibc 2.31 bundled, patchelf'd python).
# DO NOT set LD_LIBRARY_PATH: the python's DT_RPATH already points at the bundled
# glibc; adding LD_LIBRARY_PATH re-introduces a bad lib mix and segfaults.
#   scripts/zhou2_vllm_env.sh vllm serve <model> --port 8000 --served-model-name qwen-judge ...
export PATH="$HOME/micromamba/envs/vllm/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset LD_LIBRARY_PATH
exec "$@"
