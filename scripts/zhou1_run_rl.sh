#!/usr/bin/env bash
# Real (full-scale) Ditto RL run on the zhou-1 container.
# Analog of scripts/zhou1_container_smoke.sh, but full recipe defaults instead
# of the 1-step smoke overrides.
#
# First real run (per decisions 2026-06-23): GRPO baseline, sotopia, Qwen3-8B, 4 GPUs.
#   - AGENT_VERSION=default  → vanilla GRPO (no copy-mode hint/teacher step)
#   - sotopia STILL needs an external OpenAI-compatible endpoint for the
#     partner + 7-dim judge + hack-judge (default mode does NOT remove these).
#
# Usage (on zhou-1):
#   OPENAI_API_KEY=sk-... bash scripts/zhou1_run_rl.sh sotopia
#   # quick end-to-end validation before committing to 200 steps:
#   OPENAI_API_KEY=sk-... TOTAL_TRAINING_STEPS=5 SAVE_FREQ=-1 bash scripts/zhou1_run_rl.sh sotopia
set -euo pipefail

# ── Task ──────────────────────────────────────────────────────────────────────
TASK="${1:-${TASK:-sotopia}}"

# ── Host paths / image ────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/home/2025user/zhou/persona-rl-exploration}"
IMAGE="${IMAGE:-dockerhub.zjusct.io/verlai/verl:vllm011.latest}"
MODEL_PATH="${MODEL_PATH:-/home/2025user/zhou/hf_models/Qwen/Qwen3-8B-Instruct}"
NVIDIA_BIND_LIB="${NVIDIA_BIND_LIB:-/home/2025user/zhou/nvidia-bind/lib}"

# ── Run identity / scale ──────────────────────────────────────────────────────
EXPERIMENT_NAME="${EXPERIMENT_NAME:-baseline-grpo-${TASK}}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/real}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
AGENT_VERSION="${AGENT_VERSION:-default}"      # default = GRPO baseline; copy = full Ditto
N_GPUS="${N_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-5}"

# ── External judge/partner endpoint (sotopia & other interactive tasks) ───────
# sotopia (default mode too) calls real OpenAI for the partner + 7-dim judge +
# hack-judge. OPENAI_API_KEY / OPENAI_BASE_URL come from .env (source it first).
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt-5.4-mini}"      # 7-dim judge + hack-judge
JUDGE_MODEL_REASONING="${JUDGE_MODEL_REASONING:-low}"
# Partner (interlocutor) is hard-coded to gpt-5-nano in agents/sotopia/agent.py;
# real OpenAI serves it, so no patch needed (edit that file to change it).

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/outputs/real" "${REPO_ROOT}/tmp/ray"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}"

ray stop --force >/dev/null 2>&1 || true

set +e
docker run --rm --runtime=runc --entrypoint /bin/bash --network=host --ipc=host --shm-size=64g \
  --device=/dev/nvidia0 --device=/dev/nvidia1 --device=/dev/nvidia2 --device=/dev/nvidia3 \
  --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools \
  -v "${NVIDIA_BIND_LIB}:/usr/local/nvidia/lib64:ro" \
  -v "${REPO_ROOT}:/workspace/persona-rl" \
  -v /home/2025user/zhou/hf_models:/home/2025user/zhou/hf_models:ro \
  -e "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}" \
  -e PYTHONPATH=/workspace/persona-rl/OdysSim \
  -e "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}" \
  -e WANDB_MODE=offline \
  -e HYDRA_FULL_ERROR=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e RAY_TMPDIR=/workspace/persona-rl/tmp/ray \
  -e TMPDIR=/tmp \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e OPENAI_API_KEY -e OPENAI_BASE_URL \
  -e GPT54_OPENAI_API_KEY -e GPT54_OPENAI_BASE_URL \
  -e HF_TOKEN -e WANDB_API_KEY \
  -e "JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME}" \
  -e "JUDGE_MODEL_REASONING=${JUDGE_MODEL_REASONING}" \
  "${IMAGE}" \
  -lc "cd /workspace/persona-rl/OdysSim && \
    ACTOR_MODEL_PATH='${MODEL_PATH}' \
    DATA_DIR=../data \
    OUTPUT_DIR='${OUTPUT_DIR}' \
    EXPERIMENT_NAME='${EXPERIMENT_NAME}' \
    AGENT_VERSION='${AGENT_VERSION}' \
    N_GPUS='${N_GPUS}' \
    TOTAL_TRAINING_STEPS='${TOTAL_TRAINING_STEPS}' \
    SAVE_FREQ='${SAVE_FREQ}' \
    TEST_FREQ='${TEST_FREQ}' \
    bash recipe/ditto/run_rl.sh '${TASK}'" 2>&1 | tee "${LOG_FILE}"
    # ── If you hit CUDA OOM on 4 GPUs, append these proven (smoke) fallbacks to
    #    the run_rl.sh line above, easing the heaviest knobs first:
    #      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    #      actor_rollout_ref.rollout.max_model_len=12288 \
    #      actor_rollout_ref.rollout.enforce_eager=True \
    #      +actor_rollout_ref.rollout.enable_sleep_mode=False \
    #      actor_rollout_ref.rollout.tensor_model_parallel_size=2   # shard 8B across 2 GPUs
status=${PIPESTATUS[0]}
set -e

ray stop --force >/dev/null 2>&1 || true
echo "RUN_STATUS=${status}"
echo "RUN_LOG=${LOG_FILE}"
exit "${status}"
