#!/usr/bin/env bash
# zhou-1 sequential real runs: GRPO baseline (default) then Ditto (copy),
# sotopia, Qwen3-4B-Instruct-2507. Judge + partner served LOCALLY by llama.cpp
# (Qwen3-32B) on zhou-2's 8x3090 (nginx LB :8000), reached over the tailnet.
# Run detached:  nohup bash scripts/zhou1_chain_sotopia_4b.sh </dev/null >logs/chain.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

set -a; source .env 2>/dev/null; set +a   # HF_TOKEN, WANDB_API_KEY
# NOTE: .env also exports OPENAI_BASE_URL=https://api.openai.com/v1 (blocked from
# China) and a real OPENAI_API_KEY. Judge/partner are the LOCAL zhou-2 llama.cpp,
# so FORCE the local endpoint here (override .env). Use JUDGE_URL to point elsewhere.
export OPENAI_BASE_URL="${JUDGE_URL:-http://100.94.99.25:8000/v1}"
export OPENAI_API_KEY="EMPTY"
export SIM_LLM_MODEL="${SIM_LLM_MODEL:-qwen-judge}"

MODEL="${MODEL:-/home/2025user/zhou/hf_models/Qwen/Qwen3-4B-Instruct-2507}"
GPU=actor_rollout_ref.rollout.gpu_memory_utilization=0.3   # heavy other-user load (~20GB/GPU); keep peak ~48GB

run() {  # $1=agent_version  $2=experiment_name
  echo "=== [$(date)] START $2 (agent_version=$1) ==="
  MODEL_PATH="$MODEL" AGENT_VERSION="$1" EXPERIMENT_NAME="$2" \
    TOTAL_TRAINING_STEPS="${STEPS:-200}" SAVE_FREQ=50 TEST_FREQ=25 EXTRA_ARGS="$GPU" \
    bash scripts/zhou1_run_rl.sh sotopia
  echo "=== [$(date)] DONE $2 exit=$? ==="
}

run default grpo-baseline-sotopia-4b
run copy    ditto-copy-sotopia-4b
echo "=== [$(date)] CHAIN COMPLETE ==="
