# persona-rl-exploration

Working repo for reproducing **Ditto** — a Qwen3-8B human-behavior simulator
trained with RL from *verbal feedback* — built on the **OdysSim** codebase
(a fork of verl 0.7.0 + a FoldGRPO multi-agent patch).

## Layout

| Path | Contents |
|------|----------|
| `OdysSim/` | The training/eval codebase, as a **git submodule** → [`wannabeyourfriend/OdysSim`](https://github.com/wannabeyourfriend/OdysSim) |
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

## Secrets

Secrets live in `.env` (gitignored). Copy `.env.example` → `.env` and fill in.
A `pre-commit` hook in `.githooks/` blocks committing `.env` or credential
patterns; activate it with `git config core.hooksPath .githooks`.
