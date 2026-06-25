#!/usr/bin/env python3
# Patch vLLM's CUDA platform probe so `vllm serve` doesn't crash when
# CUDA_VISIBLE_DEVICES is set. vllm/platforms/cuda.py log_warnings() ->
# _get_physical_device_name() calls nvmlDeviceGetHandleByIndex over the PHYSICAL
# device count while the process only sees the visible subset -> NVMLError_Unknown.
# It's a cosmetic warning; wrap it so a bad index returns "unknown" instead of
# raising. (The P2P/NVLink probe in the same file is already try/except-guarded.)
#
# Usage:  VLLM_ENV=~/micromamba/envs/vllm python zhou2_patch_vllm_cuda.py
import os
import sys

env = os.environ.get("VLLM_ENV", os.path.expanduser("~/micromamba/envs/vllm"))
f = os.path.join(env, "lib/python3.11/site-packages/vllm/platforms/cuda.py")
s = open(f).read()
old = ('        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)\n'
       '        return pynvml.nvmlDeviceGetName(handle)')
new = ('        try:\n'
       '            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)\n'
       '            return pynvml.nvmlDeviceGetName(handle)\n'
       '        except Exception:\n'
       '            return "unknown"')
if 'return "unknown"' in s:
    print("ALREADY_PATCHED")
elif old not in s:
    print("PATTERN_NOT_FOUND")
    sys.exit(1)
else:
    open(f, "w").write(s.replace(old, new, 1))
    print("PATCHED_OK")
