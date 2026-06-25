# persona-rl-exploration

Working repo for reproducing **Ditto** — a Qwen3-8B human-behavior simulator
trained with RL from *verbal feedback* — built on the **OdysSim** codebase
(a fork of verl 0.7.0 + a FoldGRPO multi-agent patch).

This repo also tracks **UserRL** as a related reference codebase for multi-turn,
user-centric RL gyms. Use OdysSim for the Ditto reproduction path; use UserRL
for comparison against a broader GRPO-based interactive-agent training
framework.

## Layout

| Path | Contents |
|------|----------|
| `OdysSim/` | The training/eval codebase, as a **git submodule** → [`wannabeyourfriend/OdysSim`](https://github.com/wannabeyourfriend/OdysSim) |
| `UserRL/` | Related user-centric RL framework, as a **git submodule** → [`wannabeyourfriend/UserRL`](https://github.com/wannabeyourfriend/UserRL) |
| `data/` | `.gitkeep` placeholders only — real parquets live on [HF: `wannabeyourfriend-hf/persona-rl-data`](https://huggingface.co/datasets/wannabeyourfriend-hf/persona-rl-data) (see `data/README.md`) |
| `notes/` | Reproduction guide + extracted paper notes |

## Clone

```bash
git clone --recurse-submodules https://github.com/wannabeyourfriend/persona-rl-exploration.git
# or, if already cloned:
git submodule update --init --recursive

# fetch the datasets
hf download wannabeyourfriend-hf/persona-rl-data --repo-type dataset --local-dir data
```

## Submodules

This repository intentionally keeps the runnable frameworks as submodules:

- `OdysSim/` is the implementation target for Ditto reproduction. Run Ditto
  experiments from inside `OdysSim/`, with `DATA_DIR=../data`.
- `UserRL/` is a related reference repo for multi-turn user-centric RL. It is
  useful for comparing gym abstractions, GRPO multi-turn credit assignment,
  SFT cold-start setup, and simulated-user evaluation, but it is not the primary
  Ditto training path.

To refresh both submodules after pulling:

```bash
git submodule update --init --recursive
```

To inspect their pinned commits:

```bash
git submodule status --recursive
```

## Secrets

Secrets live in `.env` (gitignored). Copy `.env.example` → `.env` and fill in.
A `pre-commit` hook in `.githooks/` blocks committing `.env` or credential
patterns; activate it with `git config core.hooksPath .githooks`.
