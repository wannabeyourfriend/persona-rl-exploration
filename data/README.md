# Data

The training / evaluation parquets are **not** stored in git (only `.gitkeep`
placeholders are). They are hosted on the Hugging Face Hub:

**https://huggingface.co/datasets/wannabeyourfriend-hf/persona-rl-data**

- `sim_rl_data/` — RL-training parquets (mirror of `sunweiwei/sim-rl-data`)
- `sim_eval_data/` — evaluation parquets (mirror of `sunweiwei/sim-eval-data`)

## Download

```bash
pip install huggingface_hub
hf download wannabeyourfriend-hf/persona-rl-data --repo-type dataset --local-dir data
```

This restores `data/sim_rl_data/` and `data/sim_eval_data/` as expected by
`OdysSim/run_rl.sh` (which references the underscore directory names).
