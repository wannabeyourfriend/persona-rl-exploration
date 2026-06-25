#!/usr/bin/env bash
# zhou-2: serve Qwen3-32B-Q4 (GGUF) as an OpenAI-compatible judge via llama.cpp (CUDA).
# One instance per call; PORT/GPUS let you run one-per-GPU(-pair) for data-parallel.
#   PORT=8000 GPUS=0 bash zhou2_serve_judge.sh         # validation (1 GPU)
set -uo pipefail
SRV="$HOME/llamaenv/bin/llama-server"
MODEL="${MODEL:-$HOME/models/Qwen_Qwen3-32B-Q4_K_M.gguf}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"
CTX="${CTX:-32768}"
PAR="${PAR:-4}"
export CUDA_VISIBLE_DEVICES="$GPUS"
echo "[$(date)] serve port=$PORT gpus=$GPUS ctx=$CTX parallel=$PAR model=$(basename "$MODEL")"
# --reasoning-budget 0 disables Qwen3 thinking (clean partner dialogue + direct JSON judging)
nohup "$SRV" -m "$MODEL" --host 0.0.0.0 --port "$PORT" \
  -ngl 99 --ctx-size "$CTX" --parallel "$PAR" --jinja --reasoning-budget 0 --alias qwen-judge \
  </dev/null >"$HOME/llama_server_${PORT}.log" 2>&1 &
echo "SERVER_PID=$!"
