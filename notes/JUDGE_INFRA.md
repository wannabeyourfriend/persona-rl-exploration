# Judge / training infrastructure (zhou-1, zhou-2)

How the Ditto reproduction runs across two remote GPU hosts, and how the
LLM-as-judge (which also supplies the verbal feedback in `copy` mode) is served.

## Hosts & access

| Host | tailnet IP | GPUs | Role |
|------|-----------|------|------|
| `zhou-remote-1` | 100.74.145.47 | 4× A100 80GB | training (Docker + verl image), user `2025user`/`zhou`, repo at `/home/2025user/zhou/persona-rl-exploration` |
| `zhou-remote-2` | 100.94.99.25  | 8× RTX 3090 24GB, **no NVLink** | judge serving, user `normal_user_2`, `$HOME=/home/normal_user_2/zhou` |

Tailscale is **off by default** on these hosts — wrap every call:

```bash
cd /Users/admin/Codebase/ssrm-workspace
scripts/with-tailscale -- ssh zhou-remote-1 '<cmd>'
```

Gotchas: macOS has no `timeout`; `scp` through the wrapper is flaky (exit 255) —
deploy files via base64 (`B64=$(base64 < f|tr -d '\n'); ssh "echo $B64|base64 -d > dest"`);
the wrapper does **not** forward stdin. zhou-1↔zhou-2 talk directly over the
tailnet (fast, 0.85ms) even when the Mac→host path flaps.

## THE judge endpoint bug (fixed)

`.env` exports `OPENAI_BASE_URL=https://api.openai.com/v1` (blocked from China).
A launcher that does `set -a; source .env; set +a` then
`OPENAI_BASE_URL="${OPENAI_BASE_URL:-<judge>}"` keeps the **blocked** value (the
`:-` default never fires). Symptom: run looks alive but **every** judge call
fails (`[CALL OPENAI] Error after 3 attempts`), reward collapses to a constant
fallback ≈ `0.0714` (=1/14) → `advantages=0`, `grad_norm=0` → **no learning**.

Fix (`scripts/zhou1_chain_sotopia_4b.sh`): after sourcing `.env`, **force** the
local judge — `OPENAI_BASE_URL="${JUDGE_URL:-http://<judge-host>:8000/v1}"`,
`OPENAI_API_KEY="EMPTY"`. `scripts/zhou1_run_rl.sh` forwards these into the
training container via `-e`; nothing re-clobbers them inside.

The endpoint-portability code change (route partner/judge/hack/hint through one
model via `SIM_LLM_MODEL`, chat.completions + json_schema instead of the
OpenAI-only Responses API) lives in the **OdysSim submodule**
(`agents/utils.py`, commit `f03d1070`).

## Judge option A — llama.cpp on zhou-2 (works today; slow)

glibc 2.27 (Ubuntu 18.04) blocks modern vLLM/torch, so the first judge is
llama.cpp (conda-forge CUDA build 6800), serving **Qwen3-32B-Q4_K_M (GGUF)**.

- `zhou2_llamacpp_bootstrap.sh` — micromamba + conda-forge CUDA llama.cpp.
- `zhou2_serve_judge.sh` — one llama-server (PORT/GPUS/CTX/PAR).
- `zhou2_serve_all.sh` — 4 instances (GPU pairs 0,1/2,3/4,5/6,7, ports 8001-4)
  behind nginx least-conn LB on `:8000`. Current: `PAR=8 CTX=65536` (32 slots).
- `zhou2_judge_test.sh` — /v1/models, plain chat, nested AgentEval 7-dim json_schema.

**Throughput ceiling:** no NVLink → llama.cpp **layer-splits** each instance
across its 2 GPUs (sequential per token; one GPU of each pair idle, ~40 tok/s
single-stream). Under `num_workers=64` the fleet saturates and judge calls take
60–120s (right at the `OPENAI_TIMEOUT=120` edge in `OdysSim/agents/utils.py`).
Validated working (val ran with 0 errors, real per-dim scores) but step time is
judge-bound.

## Judge option B — no-root vLLM on zhou-2 (validated; see caveats)

`zhou2_vllm_noroot_setup.sh` runs modern vLLM on the glibc-2.27 host with no root:
bundle **glibc 2.31** (NOT 2.35 — ≥2.34 stubs libdl/libpthread → torch segfaults),
patchelf the env python's interpreter onto it (DT_RPATH, **never** LD_LIBRARY_PATH),
then `pip install vllm==0.8.5` (newest that resolves from the Tuna mirror; pins
torch 2.6.0+cu124). Pin `transformers==4.51.3 tokenizers==0.21.1` (vLLM pulls
transformers 5.x which breaks its tokenizer API). Apply `zhou2_patch_vllm_cuda.py`
(NVML device-name probe crashes when CUDA_VISIBLE_DEVICES is set).

Driver is **575.64.05 (CUDA 12.9)** so cu124 is fine. **Verified:** patched
python imports `torch 2.6.0+cu124` (CUDA available, 8 GPUs) and `vllm 0.8.5`.

**Caveats learned the hard way:**
- The multiprocess `vllm serve` is fragile on this old OS; a bad worker crash can
  **wedge a GPU** (NVML "Unknown Error"). This happened to **GPU2** — and a wedged
  GPU **poisons CUDA/NVML enumeration host-wide** (every new CUDA process sees 0
  devices, even when pinned to a healthy GPU). Recovery needs root:
  `nvidia-smi --gpu-reset -i 2` or a reboot. There is **no no-root** GPU reset.
- For real tensor-parallel throughput across no-NVLink 3090s: 4 replicas × TP=2
  (GPU pairs) behind the nginx LB; do NOT do one TP=8 (PCIe all-reduce scales badly).

## Judge option C — vLLM on a zhou-1 A100 (recommended)

Sidesteps zhou-2 entirely (no glibc hacks, no wedged GPU, no root). zhou-1 has
working Docker, the `verl:vllm011` image (vLLM 0.11), and **`Qwen3-32B` /
`Qwen3-32B-FP8` already on disk** at `/home/2025user/zhou/hf_models/Qwen/`.
Plan: serve `Qwen3-32B-FP8` via vLLM in the verl container on 1 A100 (FP8 ≈ 32GB
→ ~40GB KV → huge batches), train Qwen3-4B on the other 3 A100s
(`trainer.n_gpus_per_node=3`), point training at `http://127.0.0.1:8000/v1`.
An A100 + real vLLM continuous batching far outperforms the whole 3090 box.

## Training launchers (zhou-1)

- `scripts/zhou1_run_rl.sh` — wraps `OdysSim/recipe/ditto/run_rl.sh` in `docker run`;
  forwards `OPENAI_*`, `SIM_LLM_*`, `HF_TOKEN`, `WANDB_*`, `JUDGE_MODEL_NAME` via `-e`.
- `scripts/zhou1_chain_sotopia_4b.sh` — GRPO baseline (`default`) then Ditto (`copy`),
  sotopia, Qwen3-4B-Instruct-2507; forces the judge endpoint (see bug above).
- `scripts/zhou1_container_smoke.sh` — 1-step smoke (task `mistakes`, programmatic
  reward, no judge API).
