#!/usr/bin/env bash
# zhou-1 sequential real runs: GRPO baseline (default) then Ditto (copy),
# sotopia, Qwen3-4B-Instruct-2507. Judge + partner via DashScope (qwen-plus),
# because api.openai.com is unreachable from this host.
# Run detached:  nohup bash scripts/zhou1_chain_sotopia_4b.sh </dev/null >logs/chain.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

set -a; source .env; set +a
export OPENAI_API_KEY="${QWEN_LLM_API_KEY:?missing in .env}"
export OPENAI_BASE_URL="${QWEN_LLM_BASE_URL:?missing in .env}"
export SIM_LLM_MODEL="${SIM_LLM_MODEL:-qwen-plus}"

MODEL="${MODEL:-/home/2025user/zhou/hf_models/Qwen/Qwen3-4B-Instruct-2507}"
GPU=actor_rollout_ref.rollout.gpu_memory_utilization=0.5   # other users share the A100s

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
