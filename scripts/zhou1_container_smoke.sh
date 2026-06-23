#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/2025user/zhou/persona-rl-exploration}"
IMAGE="${IMAGE:-dockerhub.zjusct.io/verlai/verl:vllm011.latest}"
MODEL_PATH="${MODEL_PATH:-/home/2025user/zhou/hf_models/Qwen/Qwen2.5-7B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-container-smoke-mistakes}"
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/container-smoke}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
NVIDIA_BIND_LIB="${NVIDIA_BIND_LIB:-/home/2025user/zhou/nvidia-bind/lib}"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/outputs/container-smoke" "${REPO_ROOT}/tmp/ray"
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
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e WANDB_MODE=offline \
  -e HYDRA_FULL_ERROR=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e RAY_TMPDIR=/workspace/persona-rl/tmp/ray \
  -e TMPDIR=/tmp \
  -e VLLM_LOGGING_LEVEL=INFO \
  "${IMAGE}" \
  -lc "cd /workspace/persona-rl/OdysSim && \
    ACTOR_MODEL_PATH='${MODEL_PATH}' \
    DATA_DIR=../data \
    OUTPUT_DIR='${OUTPUT_DIR}' \
    EXPERIMENT_NAME='${EXPERIMENT_NAME}' \
    AGENT_VERSION=default \
    N_GPUS=4 \
    TOTAL_TRAINING_STEPS=1 \
    SAVE_FREQ=-1 \
    TEST_FREQ=0 \
    TRAIN_BATCH_SIZE=2 \
    PPO_MINI_BATCH_SIZE=4 \
    N_RESP_PER_PROMPT=2 \
    N_RESP_PER_PROMPT_VAL=1 \
    AGENT_NUM_WORKERS=1 \
    TRAIN_FILES=../data/smoke/mistakes_train_2.parquet \
    VAL_FILES=../data/smoke/mistakes_val_2.parquet \
    bash recipe/ditto/run_rl.sh mistakes \
      trainer.logger='[\"console\"]' \
      trainer.val_before_train=False \
      data.max_prompt_length=1024 \
      data.max_response_length=512 \
      actor_rollout_ref.rollout.max_model_len=2048 \
      actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
      actor_rollout_ref.rollout.max_num_seqs=8 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
      actor_rollout_ref.rollout.enforce_eager=True \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.rollout.enable_sleep_mode=False \
      actor_rollout_ref.model.use_fused_kernels=False \
      +actor_rollout_ref.model.override_config.attn_implementation=eager \
      actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
      actor_rollout_ref.ref.fsdp_config.use_torch_compile=False" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

ray stop --force >/dev/null 2>&1 || true
echo "SMOKE_STATUS=${status}"
echo "SMOKE_LOG=${LOG_FILE}"
exit "${status}"
