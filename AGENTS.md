# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> `CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`.

## What this repo is

Reproducing **Ditto** — a Qwen3-8B human-behavior simulator trained with RL from
**verbal feedback** (reward is a natural-language judge critique, not a scalar).
Code runs on **OdysSim** (submodule): a fork of **verl 0.7.0** + a **FoldGRPO** patch.

**Read `notes/DITTO_REPRODUCTION.md` first** — the full verified guide (method,
hyperparams, task↔data map, plan).

## Layout

| Path | Contents |
|------|----------|
| `OdysSim/` | Runnable code — git submodule (`wannabeyourfriend/OdysSim`). |
| `data/sim_rl_data/`, `data/sim_eval_data/` | Parquets, **not in git** — on HF `wannabeyourfriend-hf/persona-rl-data`. |
| `notes/` | Reproduction guide + extracted paper text. |

## Setup

```bash
git submodule update --init --recursive
hf download wannabeyourfriend-hf/persona-rl-data --repo-type dataset --local-dir data
cp .env.example .env                  # fill HF_TOKEN + judge API keys
git config core.hooksPath .githooks   # pre-commit blocks secrets
# Install OdysSim (verl fork) per OdysSim/README.md + OdysSim/docker/.
```

## Commands

Run from **inside `OdysSim/`**, and pass **`DATA_DIR=../data`** (parquets live one
level up, not in `OdysSim/data`):

```bash
cd OdysSim
DATA_DIR=../data bash run_rl.sh sotopia   # RL, one SOUL task (default: sotopia)
DATA_DIR=../data bash run_sft.sh          # SFT (full-param)
pytest tests/<path>/test_x.py::test_name  # single test
```

`run_*.sh` assemble Hydra configs and launch `train_ppo.py` / `train_sft.py`
(thin wrappers over `verl.trainer.main_ppo`). Task→parquet map is the case-statement
in `run_rl.sh`. Env overrides: `TASK`, `ACTOR_MODEL_PATH`, `OUTPUT_DIR`, `EXPERIMENT_NAME`.

## Reproduction gotchas

- **`run_rl.sh` ships `agent_version="default"` (plain GRPO). Ditto needs
  `agent_version="copy"`** — edit the script; it's not an env var.
- **Parquet `prompt` column is a dummy `"x"`** — real prompt is rebuilt from
  `extra_info` per task in `agents/<task>/`; reward is computed programmatically there.
- **Judge/partner/coach models are placeholders** (`gpt-5.4`, …) on an OpenAI-compatible
  API, separate from the policy. Substitute real models. Paper numbers aren't reproducible,
  only orderings (DITTO > GRPO > Base).
- **Absent despite README mentions:** `recipe/ditto/`, `run_opd.sh`, `eval.sh`.
- **SFT defaults to `Qwen3-VL-8B-Instruct`** (vision variant) and needs a
  `sft_train.parquet` that doesn't ship — build the corpus first.

## Architecture

- **OdysSim = verl (FSDP actor + colocated vLLM rollout + Ray/Hydra) + FoldGRPO**
  (`adv_estimator=foldgrpo`, groups by `(uid, agent_role)`, dedups by `gen_uid`).
- **Method:** policy emits a **student** `y0`; judge returns `(r, h)`; the *same* policy
  emits a **teacher** `y1` conditioned on feedback `h`; advantage is group-relative over
  all samples. `agent_version="copy"` produces 3 sequences/prompt (student, teacher-copy,
  hint_agent).
- **Per-task rollout logic** lives in `OdysSim/agents/<task>/` (`agent.py`, `hint.py`,
  `hint_agent.py`, `judge_agent.py`); registry in `agents/agents.yaml`.
- **Eval = SOUL suite** (10 tasks / 6 categories), val parquets in `data/sim_eval_data/`.

For heavy exploration of OdysSim, delegate to subagents rather than reading raw files in the main context.
