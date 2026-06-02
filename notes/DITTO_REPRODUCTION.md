# Ditto Reproduction Guide

A from-scratch reproduction guide for **Ditto**, a human-behavior simulator trained with verbal-feedback reinforcement learning, built on the **OdysSim** codebase (a fork of `verl 0.7.0`). Synthesized from a deep read of the Ditto paper, the OdysSim repo, and its HuggingFace parquet datasets.

> Naming note: model/judge identifiers in the paper and repo (`Qwen3-VL-8B-Instruct`, `gpt-5-nano`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, dates like 2026) are anachronistic relative to a Jan-2026 knowledge cutoff. Treat them as **nominal/placeholder identifiers**; reproduction must substitute real, available models and will not numerically match the paper.

---

## 1. Overview & Research Goal

Ditto is a single 8B model trained to **be a believable human** — to stand in as a user, patient, student, character, or persona inside interactive/evaluative systems — rather than to answer the human. The core research claim is that human-likeness is **subjective, multi-faceted, and multi-dimensional**, so the standard RL practice of collapsing all judge feedback into a **scalar reward** (RLHF/DPO/GRPO) discards the actionable information needed to know *why* a behavior was un-human and *how* to fix it. Ditto instead treats the LLM judge's **natural-language critique ("verbal feedback")** as a first-class RL signal: a draft rollout is critiqued, a feedback-conditioned "teacher" rollout is generated, and both are jointly optimized via GRPO so the improvement is internalized into the base policy and **no feedback is needed at test time**. The accompanying benchmark/training suite **SOUL** ("Simulation gym Of hUman-Like behavior") covers **10 tasks across 6 categories**. Reported headline: Qwen3-(VL-)8B + Ditto reaches average normalized score **0.726** (Base 0.235, GRPO 0.533, GPT-5.4 0.663), matching/exceeding GPT-5.4 on **6/10** tasks.

---

## 2. Conceptual Framework

### 2.1 Problem formulation (§3.1, paper L275–285)

- **Input** `x`: an instruction, scenario, dialogue history, or user profile.
- **Output**: a trajectory `y = (y_1, …, y_T) ~ πθ(·|x)`. The `1..T` notation explicitly accommodates **multi-turn** rollouts, not just single responses.
- **What is modeled**: the conditional distribution `πθ(y|x)` over human-like behavior trajectories.
- **Supervision**: a task-specific judge/simulator scores each trajectory. Standard RL reduces this to `r(x,y) ∈ ℝ`.

### 2.2 The verbal-feedback idea (§3.2, paper L287–311)

Ditto's judge returns a **pair**, not a scalar:

```
(Eq. 1)   (r, h) = J(x, y)
```

- `r ∈ ℝ`: scalar reward.
- `h`: **structured verbal feedback** — per-dimension scores + textual critiques + task-specific suggestions ("what was done well, what went wrong, how to improve"). Example Sotopia dimensions: believability, relationship maintenance, knowledge consistency, goal achievement, adherence to social norms.

`h` is framed as **privileged information** (Vapnik & Vashist 2009; Vapnik & Izmailov 2015): available **at training time, NOT at test time**. The whole design goal is to internalize `h`-driven improvement into `πθ(·|x)` while keeping inference unchanged (`x → πθ → y`, no `h`).

**Why scalar fails / why verbal is proposed**: a scalar 0.6 "conveys almost nothing about *why* a simulated user was too polite or too predictable, let alone how to fix it." Humans learn social behavior "not from scores but from verbal feedback, through explanations, corrections, and dialogue." Verbal feedback recovers signal on minor/auxiliary dimensions (e.g. the safety-critical `secret` dimension) that scalar RL ignores.

---

## 3. Method — Verbal-Feedback RL Recipe

### 3.1 Two-rollout, on-policy, FORWARD self-distillation as GRPO

The same shared-weight policy `πθ` plays two roles via different prompting context (no separate teacher network, no SFT inside the loop, no explicit KL-to-reference in the written objective — only PPO/GRPO clipping):

| Role | Rollout | Context | Score |
|---|---|---|---|
| **Student / "draft"** | `y0` | `x` alone | `r0` |
| **Teacher / "refined"** | `y1` | `x` **+ feedback `h`** | `r1` |

Because both come from the same policy, "distillation" is **self-distillation**: the policy distills the improvement it makes *given feedback* back into its *unconditioned* behavior.

### 3.2 Exact training procedure (§3.3, paper L313–377)

Per prompt `x`, sample `G` drafts:

```
(Eq. 2)   y0 ~ πθ(·|x)                        # draft / student
(Eq. 3)   (r0, h) = J(x, y0)                  # judge → scalar r0 + reflection h
(Eq. 4)   y1 ~ πθ(·|x, h),   r1 = R(x, y1)    # refined / teacher (h appended to x)
(Eq. 5)   G(x) = { y_{i,0}, y_{i,1} }_{i=1}^G # joint group, size 2G
```

**Joint group-relative advantage** (pooled over all 2G rewards):

```
(Eq. 6)   A_{i,j} = (r_{i,j} − μ_r) / σ_r
          μ_r = (1/2G) Σ_{i=1}^G Σ_{j=0}^1 r_{i,j}      # mean over all 2G
          σ_r = std({ r_{i,j} }_{i=1..G, j=0,1})          # std over all 2G
```

**Joint clipped GRPO loss** (students j=0 and teachers j=1):

```
(Eq. 7)   L_group = −E_x[ Σ_i Σ_{j∈{0,1}} Σ_t min( ρ_{i,j,t} A_{i,j},
                                                    clip(ρ_{i,j,t}, 1−ε, 1+ε) A_{i,j} ) ]
(Eq. 8)   ρ_{i,j,t} = πθ(y_{i,j,t} | x, y_{i,j,<t}) / π_old(y_{i,j,t} | x, y_{i,j,<t})
```

**Extra feedback-conditioned loss** (teacher-only group; *different* normalization pool — only the G teacher rewards):

```
(Eq. 9)   L_fb = −E_x[ Σ_i Σ_t min( ρ̂_{i,t} Â_{i,1}, clip(ρ̂_{i,t}, 1−ε, 1+ε) Â_{i,1} ) ]
          Â_{i,1} normalized within { r_{i,1} }_{i=1}^G   # teacher-only
```

**Final objective** (equal weight):

```
(Eq. 10)  L = L_group + L_fb
```

The two roles of feedback: (1) `L_group` — teacher rollouts serve as high-advantage on-policy **distillation targets** for the base policy; (2) `L_fb` — a direct RL signal for being good at the **teacher role itself** (responding to feedback). Ablation: without `L_fb`, the teacher–student gap rises early then declines.

### 3.3 Forward vs reverse (on-policy self-)distillation (§3.4)

| | Forward (Ditto) | Reverse (SDPO, what Ditto is NOT) |
|---|---|---|
| KL direction | `KL(teacher ‖ student)` (inferred) | `KL(student ‖ teacher)` |
| Sample from | **teacher** (`y1`), include in GRPO group | student |
| Behavior | mode-**covering** | mode-**seeking** |
| Teacher def. | live policy conditioned on `h` + trained via `L_fb` | frozen/EMA copy |
| Empirical | best | **collapses** on most metrics except `secret` |

### 3.4 Connections to other RL approaches (App D, Table 14)

| Method | Feedback | Granularity | Training | vs Ditto |
|---|---|---|---|---|
| **Ditto (Ours)** | Verbal (LLM judge) | Token (distill) | **GRPO + distillation** | — |
| GRPO (vanilla) | Scalar | Sequence | GRPO | collapses multi-dim → 1 scalar; loses `secret` |
| RLHF / DPO | Scalar / binary pref | Sequence | RLHF / DPO | no verbal signal |
| Self-Rewarding LM | Verbal (self) | Sequence | DPO | verbal → only preference pairs |
| **RLTF / RLTF-SD** (concurrent) | Verbal (LLM) | Token (distill) | RL + distill (**AWR**) | same structure, but AWR (no importance correction) vs Ditto's GRPO+clipping; raw feedback vs reflection `h`; verifiable domains |
| **ERL** | Verbal (self) | Episode | RL + consolidation (**SFT on reject-sampled y1**) | no reward normalization → unstable under noisy feedback |

### 3.5 Re-implementation checklist (symbols)

`x` input; `y=(y_1..y_T)` trajectory; `πθ` policy (single shared model); `π_old` behavior policy for ratio; `J` judge → `(r,h)`; `R` reward fn; `r` scalar reward; `h` verbal reflection; `y0~πθ(·|x)` student; `y1~πθ(·|x,h)` teacher; `G` group size; `G(x)` 2G joint group; `A_{i,j}` joint advantage (norm over 2G); `ε` clip; `ρ` token importance ratio; `L_group` joint loss; `Â_{i,1}` teacher-only advantage; `L_fb` teacher-only loss; `L = L_group + L_fb`.

> **Method→code reconciliation (critical).** In OdysSim the verbal-feedback recipe is realized **not** by re-deriving Eqs. 6–10 verbatim but by the **`agent_version='copy'` reward path** over a custom **FoldGRPO** advantage estimator (see §4). The shipped `run_rl.sh` uses *that* path (`adv_estimator=foldgrpo`, NOT `use_opd`). A separate, fully-implemented but **un-shipped** OPD (on-policy distillation) path exists (`algorithm.use_opd=True`) that maps more literally to a token-level KL distillation. The paper's reported numbers most plausibly correspond to the `copy`/FoldGRPO path; confirm against any frozen recipe before assuming OPD.

---

## 4. System Architecture

### 4.1 The integration seam (verl `experimental.agent_loop`)

`agents/agents.yaml` registers exactly **two** AgentLoop classes (both in `agents/base_agent.py`):

| `agent_name` | class | policy backend | use |
|---|---|---|---|
| `agent_hub` | `AgentHubLoop` | `CallLLM` → verl-managed vLLM/SGLang server → real `token_ids`+`log_probs` | **training** (on-policy, trainable) |
| `openai_agent` | `OpenAIAgentLoop` | `CallAPI` → OpenAI-compatible endpoint, logprobs zeroed | **eval / data-gen** (NOT trainable) |

verl loads this via `actor_rollout_ref.rollout.agent.agent_loop_config_path`. Each dataset row's `agent_name` (default `default_agent_loop=agent_hub`) selects the loop. Both loops are **thin dispatchers**: they build an LLM client + `TaskContext`, then route by **`item["data_source"]`** through `_get_agent_loop(data_source)` (`base_agent.py:16-84`) to a per-task `async def agent_loop(data, context)` coroutine in `agents/<task>/`. **Unknown `data_source` silently falls back to `math_agent_loop`** — both `agent_name` AND a routable `data_source` are required.

### 4.2 Episode machinery (`agents/utils.py`)

- **`TaskContext`** (dataclass): `config, global_step, is_train, tokenizer, llm_client`. Env agents branch on `context.is_train` and read `context.config.algorithm.use_opd` / `.agent_version`.
- **`AgentContext`** — token-level conversation bookkeeping. Parallel per-turn lists (`chat`, `chat_ids`, `log_probs`, `token_mask`, `additional_info`, `chat_completions`). `prompt_turn=2` (system+first user) marks the ungraded prompt; `append(turn, completion)` sets `token_mask=True` **only on the policy's own generated tokens** (user/tool/observation turns are mask-False context). `get_data()` → `prompt_ids = sum(chat_ids[:2])`, `response_ids = sum(chat_ids[2:])`, `response_mask` (1 for policy tokens) — the interleaved `1…1 0…0 1…1` multi-turn layout verl expects. `fork()` deep-copies but **zeros** logprobs/masks (context reuse with no training signal).
- **`Agent(AgentContext)`** — `.step()` (one vLLM generation, skips if <10 tokens left), `.react(...)` (generic multi-turn tool/ReAct loop, `max_turn=64`), `think_format_correct()` (0/1 format gate).
- **`get_agent_output(reward, extra_info, teacher_prompt, gen_uid, agent_role)`** → a verl `AgentLoopOutput`. `extra_fields` carries `reward_extra_info`, optional `gen_uid`, `agent_role`, and (when `teacher_prompt` given) `teacher_prompt_ids` (tokenized hint-augmented chat, `add_generation_prompt=True`).
- **Judge/aux calls**: `call_openai` / `call_openai_parse` (structured Pydantic via Responses API `.parse()`) use a *separate* `AsyncOpenAI` client (`OPENAI_API_KEY`/`OPENAI_BASE_URL`), with Azure fallback and `JUDGE_MODEL_NAME`/`JUDGE_MODEL_REASONING` overrides. **Entirely off the trainable policy path.**

### 4.3 The verbal-feedback mechanism for Sotopia (canonical task)

Files: `agents/sotopia/agent.py` (rollout + reward + hack judge), `agents/sotopia/hint.py` (critique→coaching brief), `agents/sotopia/hint_agent.py` (teacher rollout). **`agents/sotopia/judge_agent.py` is an EMPTY 0-byte placeholder** — judge logic lives inline in `agent.py`.

End-to-end (`copy` path):
1. Policy student rollout (2-agent sim, `max_turns=15`; partner is a fixed `SimpleAgent` via `gpt-5-nano`).
2. `evaluate_episode_single()` — LLM judge scores **7 dimensions** (ranges + weights below); each dim's `reasoning` = the verbal critique. Reward = Σ(normalized·weight)/Σweight. `hack_judge()` (gpt-5.4-mini) audits reward-hacking risk (concurrent via `asyncio.gather`).
3. Penalties (train only): hack HIGH → reward/4, MEDIUM → reward/2; `think_format_correct()==0` → reward/2.
4. **Gate** (`agent.py:747`): if `(use_opd OR agent_version=='copy') AND actor_avg ≤ 4 AND hack=='low'` → `generate_hint()` (an "expert social coach" LLM, default `gpt-5.4-nano`) reads scores+critiques+transcript and writes a ~1000-word brief with literal `say: "..."` example lines (without revealing answers).
5. `get_teacher_character_prompt()` injects the hint into a private `## Coaching Notes` system-prompt section → **teacher chat**.
6. `hint_agent.agent_loop` re-runs the episode under the teacher prompt, re-judged → `new_reward`.
7. Returns a **list of 3 outputs**: `[student, copy_agent_output, hint_agent_output]`.

```
DIMENSION_RANGES:  believability(0,10) relationship(-5,5) knowledge(0,10)
                   secret(-10,0) social_rules(-10,0) financial_and_material_benefits(-5,5) goal(0,10)
DIMENSION_WEIGHTS: believability 0.5, relationship 1.0, knowledge 1.0,
                   secret 0.5, social_rules 0.5, financial 0.5, goal 2.0
```

### 4.4 The FoldGRPO multi-agent patch (the "copy" trick)

`copy_agent_output` is the key construction (`agent.py:795-805`): a deepcopy of the teacher (`hint_agent`) output whose **`prompt_ids` are overwritten to the STUDENT prompt_ids** and given a **new `gen_uid`**. So the teacher's improved response is presented *as if produced from the plain student prompt* — within the student's GRPO group, the higher-reward teacher-derived sequence gets positive advantage and the student gets negative advantage, **pushing the policy toward the hint-conditioned behavior**. This is the practical realization of Eqs. 6–7. `hint_agent_output` is tagged `agent_role='hint_agent'` so it normalizes in its **own** group (the practical realization of `L_fb`'s separate pool).

**FoldGRPO** (`verl/trainer/ppo/core_algos.py:332-379`): group key = `(uid, agent_role)` if `agent_role` else `uid`; **de-dups sub-sequences by `gen_uid`** (so variable sub-seq counts don't bias the group mean/std); `adv = (score − mean_g)/(std_g+1e-6)` broadcast over response tokens. `ray_trainer.py:248` asserts `gen_uid` present for FoldGRPO; `uid` set per prompt (`ray_trainer.py:1520`), `gen_uid` per rollout slot (`:1532`), `n_resp_per_prompt=8`. Metric utils use the first sub-seq per `gen_uid` and skip `agent_role` rows (metrics reflect the primary agent only).

### 4.5 `agent_version`: `copy` vs `default` (mapped to method)

| `agent_version` | hint? | outputs | method | run_rl.sh |
|---|---|---|---|---|
| `default` (shipped) | no (gate requires copy/opd) | 1 (student) | **vanilla GRPO** on scalar judge reward | line 65 default |
| `copy` | yes (low-score, non-hack) | 3 `[student, copy(teacher@student-prompt, new gen_uid), teacher(role=hint_agent)]` | **Ditto verbal-feedback RL** (forward distillation via FoldGRPO) | **set manually** |

### 4.6 The (un-shipped) OPD path — for completeness

`algorithm.use_opd=True` + worker class `OPDActorRolloutRefWorker` adds a token-level distillation term: `teacher_prompt_ids` (hint-augmented context) → EMA teacher computes `log π_teacher(a_t) − log π_student(a_t)` over the student's own actions. Single-token mode: per-token log-ratio advantage `advantages = grpo_adv + opd_alpha·opd_adv` (`opd_alpha=0.1, opd_clip=5.0, estimator=local`). Top-k mode (`opd_topk>1 AND use_fused_kernels AND use_topk_kernel`): "Hinton-style" forward KL `Σ p_teacher(log p_teacher − log p_student)` added to `policy_loss` (`dp_actor.py:883-903`). EMA update rate 0.05. **Not enabled by any shipped script; `run_opd.sh` is referenced in README but missing.**

### 4.7 Environment HTTP contract (for tool/env tasks like tau_usi)

Some tasks step an external **runtime service** (`BaseEnv`, `RUNTIME_SERVICE_URL=http://$BASE_DOMAIN:8005`): `POST /create` (reset), `/step`, `/reward`, `/ping`; tool calls emitted as `<function=NAME><parameter=KEY>value</parameter></function>`. Not needed for the core 10 SOUL tasks but required for tau_usi/taubench-style envs.

---

## 5. SOUL Evaluation Suite

10 tasks / 6 categories. Eval protocol: **sample 100 instances per task** (not full set); for multi-metric tasks report the **average** as the primary metric.

| Category / Task | What it simulates | Output format | Metric | Train data source | Train file → Eval file |
|---|---|---|---|---|---|
| **ToM** · FanToM | ToM in multi-party convos (info asymmetry) | MCQ+QA | Accuracy (binary/list 1/0; fact = token F1) | official train split (18 convs held out) | `fantom_rl_train.parquet` → `fantom_val.parquet` |
| **ToM** · HiToM | higher-order nested beliefs, deception | MCQ | Accuracy | official + 200 gen stories → 6,000 ex | `hitom_rl_train.parquet` → `hitom_val.parquet` |
| **ToM** · ToMi (ParaToMi) | paraphrased false-belief test | QA | Accuracy | ParaToMi 6,004 paraphrased ex | `paratomi_rl_train.parquet` → `paratomi_val.parquet` |
| **Role Play** · CoSER | literary character role-play (GCA), multi-turn | Multi-turn gen (≤20 rounds) | LLM-judge flaw detection, 4 dims (Storyline Consistency, Anthropomorphism, Character Fidelity, Storyline Quality) → mean/100; +BLEU/ROUGE-L | 26.5k-dialogue train set (771 books) | `coser_rl_train.parquet` → `coser_val.parquet` |
| **Role Play** · LifeChoices | persona-driven decisions (pick char's actual choice) | MCQ (4-way) | exact-match Accuracy | train split (1,462 pts / 388 novels) | `lifechoices_hard_rl.parquet` → `lifechoices_val.parquet` |
| **Social Skill** · Sotopia | dyadic social interaction w/ private goals | Multi-agent interaction (≤15–20 turns) | LLM-judge **7 dims** (believability, relationship, knowledge, secret, social_rules, financial, goal) → weighted-normalized | Sotopia-π 2,310 scenarios (462×5) | `sotopia_clean_rl.parquet` → `sotopia_hard_val.parquet` |
| **Learner Sim** · Mistakes | select WRONG answer matching a misconception | MCQ (4-way) | Accuracy | Eedi (NeurIPS 2024 Kaggle), per (Q, wrong-opt) | `mistakes_rl_train.parquet` → `mistakes_val.parquet` |
| **User Sim** · MirrorBench | multi-turn human-like user turns | Multi-turn gen | 6 metrics: MATTR(w=50)/HD-D/Yule's K (z-scored) + GTEval/Pairwise-Indistinguishability/Rubric-and-Reason | ~3,400 resampled convos (Arena/ClariQ/OASST1/QuLAC) | `mirrorbench_rl_train.parquet` → `mirrorbench_val.parquet` |
| **User Sim** · UserLLM | single user turn (3 subtasks) | Single-turn gen | CSQA role-adherence (substring rule); NQ intent-adherence (LLM REFUSED→1); PRISM 4 metrics (diversity, intent-decomp, termination-F1, AI-detector) | PRISM + NaturalQuestions + CommonsenseQA | `userllm_rl_train.parquet` → `userllm_val.parquet` |
| **Persona Sim** · TwinVoice | pick reply matching user's style | MCQ (4-way) | Accuracy | Bluesky 471 + Telegram 384 + Gutenberg 480 = 1,335 (GPT-5.4-mini drafts, GPT-5.4 filters) | `twinvoice_rl_train.parquet` → `twinvoice_val.parquet` |

**Target numbers (Table 1/3)** — primary metric, higher better, 100 instances/task:

| Task | GPT-5.4 | GPT-5-nano | Base | GRPO | **DITTO** |
|---|---|---|---|---|---|
| FanToM | 0.900 | 0.720 | 0.780 | 0.780 | **0.950** |
| HiToM | 0.700 | 0.370 | 0.580 | 0.580 | **0.780** |
| ToMi | 0.880 | 0.850 | 0.680 | 0.680 | **0.930** |
| CoSER | **0.659** | 0.352 | 0.435 | 0.435 | 0.512 |
| LifeChoices | **0.870** | 0.600 | 0.670 | 0.670 | 0.800 |
| Sotopia | 0.300 | 0.310 | 0.277 | 0.277 | **0.470** |
| Mistakes | 0.570 | **0.580** | 0.460 | 0.460 | 0.560 |
| MirrorBench | 0.536 | 0.358 | 0.547 | 0.547 | **0.713** |
| UserLLM | 0.575 | 0.324 | 0.469 | 0.469 | **0.930** |
| TwinVoice | 0.640 | 0.230 | 0.430 | 0.430 | 0.610 |
| **Average** | 0.663 | 0.469 | 0.533 | 0.533 | **0.726** |

> The "Base" column varies between the two table transcriptions (Table 1 reports a stronger base ≈0.533; Table 3 reports a near-zero raw base ≈0.235). Treat the **per-task DITTO/GRPO/GPT values** as ground truth; the "+36%" prose claim matches **0.726/0.533 − 1 = +36% over GRPO**, mislabeled "over base" in the abstract. DITTO beats GPT-5.4 on 6/10 (FanToM, HiToM, ToMi, Sotopia, MirrorBench, UserLLM) and GPT-5-nano on all 10. Largest GRPO→DITTO deltas: TwinVoice +14, ToMi +11, LifeChoices +11.

---

## 6. Datasets

### 6.1 The single unifying parquet schema (RL train AND eval — all 50 files)

```
prompt:      list<struct<content: string, role: string>>   # ALWAYS the placeholder [{"content":"x","role":"user"}]
data_source: string                                          # task key → reward routing + agent dispatch
extra_info:  struct<...>                                     # per-task: holds the REAL prompt/scenario/persona/gold
```

**No `reward_model`, no `ground_truth`, no `ability` columns.** Reward is computed **programmatically per task** (exact-match for MCQ; LLM-judge for open-ended). `extra_info.index` (int64) is the row id.

> **Critical reproduction fact:** the top-level `prompt` is a non-functional dummy `"x"`. The real prompt/scenario/persona is reconstructed at rollout time from `extra_info` by `agents/<task>/agent.py`. A naive verl run that generates from `prompt` produces garbage. **You must port both the parquet builder AND the matching agent code per `data_source`.**

`extra_info` structural families: (1) self-contained JSON blob `{index, raw}` (`sim_arena_math/doc`, `behavior_chain`); (2) role-play scenario struct (sotopia explicit fields; coser `circumstance` JSON ~10KB); (3) MCQ with explicit gold (mistakes, hitom, fantom, paratomi, lifechoices nested `Multiple Choice Question{Correct Answer,...}`, social_r1); (4) conversation/persona-prediction with embedded chat (humanllm, socsci210, mirrorbench, twinvoice unicode-escaped Chinese, userllm); (5) text continuation (humanual ×6); (6) preference (alignx `{chosen, rejected, demographic, icl_pairs, ugc}`).

### 6.2 Full TASK → train/val file mapping (`run_rl.sh:30-55`)

Files under `$DATA_DIR/sim_rl_data/` (train) and `$DATA_DIR/sim_eval_data/` (val); override via `TRAIN_FILES`/`VAL_FILES`.

| TASK | train parquet (rows, data_source) | val parquet |
|---|---|---|
| sotopia | `sotopia_clean_rl` (405, `sotopia`) | `sotopia_hard_val` (+`hard`) |
| coser | `coser_rl_train` (1024, `coser`) | `coser_val` |
| lifechoices | `lifechoices_hard_rl` (1150, `lifechoices`) | `lifechoices_val` |
| userllm | `userllm_rl_train` (1024, `userllm`) | `userllm_val` (richer: +`answer_key`,`related_metrics`,`choices`) |
| mirrorbench | `mirrorbench_rl_train` (1024, `mirrorbench`) | `mirrorbench_val` |
| fantom | `fantom_rl_train` (1024, `fantom`) | `fantom_val` |
| hitom | `hitom_rl_train` (1024, `hitom`) | `hitom_val` |
| paratomi | `paratomi_rl_train` (1024, `paratomi`) | `paratomi_val` |
| mistakes | `mistakes_rl_train` (1024, `mistakes`) | `mistakes_val` |
| twinvoice | `twinvoice_rl_train` (1024, `twinvoice`) | `twinvoice_val` |
| social_r1 | `social_r1_rl` (687, `social_r1`) | `social_r1_val` |
| behaviorchain | `behaviorchain_rl_train` (1024, `behavior_chain`) | `behaviorchain_val` |
| sim_math | `sim_math_rl` (1043, `sim_arena_math`) | `sim_math_val` |
| sim_doc | `sim_doc_rl` (815, `sim_arena_doc`) | `sim_doc_val` |
| humanual_{book,chat,email,news,opinion,politics} | `humanual_rl_<type>` (1000 ea, `humanual-<type>`) | `humanual_<type>_val` (+`split_type`) |
| alignx | `alignx_rl_8k` (8191, `alignx`) | `alignx_demo_val` (+ 4 more variants: arbitrary/history16/pair/ugc) |
| socsci210 | `socsci210_rl_2k` (2000, `socsci210`) | `socsci210_val` |
| humanllm | `humanllm_rl_train` (1024, `humanllm`) | `humanllm_val` (500 rows) |

> **Naming mismatches that break naive reproduction**: TASK `behaviorchain` (file) vs `data_source` `behavior_chain` (router); TASK `sim_math/sim_doc` vs `data_source` `sim_arena_math/sim_arena_doc`; TASK `humanual_book` vs `data_source` `humanual-book`. The mapping only works because the `data_source` value lives **inside** the parquet rows. There is **no in-repo `prepare_dataset.py`** for these tasks; the parquets ship from HF (`sunweiwei/sim-rl-data`, `sim-eval-data`).

> The README lists **23 supported RL tasks** (matching `run_rl.sh` cases). The paper's canonical eval is **10 tasks** — repo subtasks/subsets (UserLLM×3, MirrorBench×4, CoSER ID+OOD, TwinVoice×3, alignx×5) expand the count. Eval-only dirs (`reward_bench`, `rewardbench2`, `rm_r1`, `rolermbench`, `tau_usi`, `taubench`, `tic_tac_toe`, `tombench`, `mmtom`, `instruct`) are not RL tasks; several are not even in the router.

### 6.3 SFT data schema (different — and NOT provided)

The SFT stage consumes a **different** schema: rows with `data_source: str` + a `messages: list<{role, content}>` list (or `extra_info.messages`). **No such files ship** — `run_sft.sh` expects `data/sft_train.parquet`, `data/sft_val.parquet`, `data/val.parquet` that do not exist. The HF parquets are the RL/eval schema (`prompt`/`data_source`/`extra_info`), NOT the SFT `messages` schema. This is the single biggest SFT reproduction blocker (see §8).

---

## 7. From-Scratch Reproduction Plan

> Convention: success criteria are stated per phase. Numbers from a placeholder-judge run will **not** match the paper exactly; treat relative orderings (DITTO > GRPO > Base) and curve shapes as the verification targets.

### Phase A — Environment / Docker setup

OdysSim is a `verl 0.7.0` fork. Use the stable vLLM Docker base.

```bash
# (a1) Clone into the working dir
git clone <odyssim-repo-url> /Users/admin/codebase/persona-rl/OdysSim   # already present locally
cd /Users/admin/codebase/persona-rl/OdysSim

# (a2) Container — use the shipped stable base (see docker/)
#   docker/Dockerfile.stable.vllm   OR   docker/verl0.5-cu126-torch2.7.1-fa2.8.0
# (a3) Inside the container / a fresh GPU env:
uv pip install -e .                 # verl + extras
uv pip install -r requirements-cuda.txt
uv pip install vllm flash-attn      # rollout engine + FA2/3
uv pip install pandas pyarrow       # for parquet inspection (pyarrow >= 23, pandas >= 3 OK)
```

Hardware target: **8× GPUs** (`n_gpus_per_node=8`), bf16, FSDP param+optimizer offload, vLLM `gpu_memory_utilization=0.7`.

**Judge / aux LLM endpoint** (load-bearing — substitute real models):
```bash
export OPENAI_API_KEY=...           # judge/coach/partner client (call_openai*)
export OPENAI_BASE_URL=...          # OpenAI-compatible endpoint
export JUDGE_MODEL_NAME=...         # replaces gpt-5-nano (e.g. gpt-4o / a strong local model)
export JUDGE_MODEL_REASONING=low
# Partner/coach default to gpt-5-nano/gpt-5.4-nano/gpt-5.4-mini → all overridable via the same client
```

**✅ Verification A**: `python -c "import verl, vllm; print('ok')"`; `python train_ppo.py --help` (4-line wrapper around `verl.trainer.main_ppo.main`) imports cleanly; a 1-prompt smoke rollout against your judge endpoint returns a parsed `(r, h)` from `call_openai_parse`.

### Phase B — Data prep

```bash
# (b1) Place the RL/eval parquets (from HF sunweiwei/sim-rl-data & sim-eval-data) under:
#   data/sim_rl_data/*.parquet   data/sim_eval_data/*.parquet     (already present locally)
ls /Users/admin/Codebase/persona-rl/data/sim_rl_data | wc -l   # expect 23
ls /Users/admin/Codebase/persona-rl/data/sim_eval_data | wc -l # expect 27
```

**✅ Verification B**: for `sotopia_clean_rl.parquet`, confirm 405 rows, columns `[prompt, data_source, extra_info]`, `prompt == [{"content":"x","role":"user"}]`, and `extra_info` has `agent1_background, agent2_goal, eval_position, scenario`. For `sim_math_rl.parquet`, confirm `data_source == "sim_arena_math"` and `json.loads(extra_info["raw"])` yields `{problem, correct_answer, user_profile_text}`.

### Phase C — SFT stage (warm-start; OPTIONAL/UNVERIFIED)

> The SFT stage is **plain cross-entropy behavior-cloning**, NOT distillation (`policy_loss.loss_mode=sft` → `compute_sft_loss = agg_loss(-log_prob, response_mask)`; `old_log_probs`/`advantages` are dummies). Its targets are human turns; chat processors **swap user↔assistant** so the human utterance becomes the loss-bearing "assistant" output. **Input data does not ship** — you must reconstruct the `messages`-format corpus (wildchat/lmsys/oasst/coser/humanual/tom_*/soc_*/convokit_* etc. with per-dataset system prompts baked in). There is **no scripted SFT→RL handoff**; RL defaults to the base model. Skip this phase unless reproducing the full pipeline.

```bash
# run_sft.sh (demo) — full-param FSDP, vLLM kept for optional gen-eval
ACTOR_MODEL_PATH=Qwen3-VL-8B-Instruct \   # NOTE: VL variant in the demo script
EXPERIMENT_NAME=sft-demo \
bash run_sft.sh
# Key hypers: lr=1e-5 cosine, warmup=50; maxlen 8192/8192; train_batch_size=1024,
#   ppo_mini_batch_size=256; total_training_steps=500; test_freq/save_freq=50;
#   loss_mode=sft; full-param (NO LoRA); TURNOFF_THINK=1
```

**✅ Verification C**: `val/sft_loss` (and per-source `val/sft_loss/<data_source>`) decreases over steps; a checkpoint lands in `outputs/sft-demo/global_step_500`. To chain into RL, manually set `ACTOR_MODEL_PATH` to that checkpoint dir.

### Phase D — RL stage (the core)

The **critical edit**: `run_rl.sh:65` ships `agent_version="default"` (vanilla GRPO). For **Ditto verbal-feedback RL, set it to `copy`**.

```bash
cd /Users/admin/Codebase/persona-rl/OdysSim
# Edit run_rl.sh line 65:  agent_version="copy"     (default = vanilla-GRPO baseline)

ACTOR_MODEL_PATH=Qwen/Qwen3-8B-Instruct \   # base for Ditto-8B (RL script default)
TASK=sotopia \
DATA_DIR=/Users/admin/Codebase/persona-rl/data \
OPENAI_API_KEY=... OPENAI_BASE_URL=... JUDGE_MODEL_NAME=... \
TURNOFF_THINK=1 \
bash run_rl.sh sotopia
```

**Exact RL hyperparameters (`run_rl.sh:60-160`):**

| Param | Value | Param | Value |
|---|---|---|---|
| base model | `Qwen3-8B-Instruct` | `adv_estimator` | `foldgrpo` |
| `agent_version` | `copy` (Ditto) / `default` (GRPO) | LoRA rank / alpha | 32 / 64 (target all-linear) |
| `optim.lr` | `5e-6` (constant) | `train_batch_size` | 64 |
| `ppo_mini_batch_size` | 16 | `rollout.n` (group G) | 8 |
| `max_prompt_length` | 8192 | `max_response_length` | 8192 |
| `clip_ratio_low/high/c` | 0.2 / 0.28 / 10.0 | `use_kl_loss` | False |
| `entropy_coeff` | 0 | `loss_mode` | vanilla |
| `n_gpus_per_node` | 8 | `total_training_steps` | 200 |
| `save_freq / test_freq` | 50 / 5 | `val_before_train` | True |
| rollout | vllm, `gpu_mem_util=0.7`, `max_model_len=17408` | `default_agent_loop` | `agent_hub` |
| `use_fused_kernels` | True | `TURNOFF_THINK` | 1 |
| `project_name` | `ditto` | `experiment_name` | `ditto-rl-${TASK}` |

Train one model per task (10 SOUL tasks); the paper subsamples **1,024 instances/task → 10,240 total** for a single multi-task model, but the shipped script is **per-task** (`TASK` selects one train/val pair). To reproduce the single Ditto-8B, either run multi-task by concatenating the 10 train parquets via `TRAIN_FILES`, or train per-task and report per-task.

**✅ Verification D**: (1) startup asserts `gen_uid` present (FoldGRPO); (2) in `copy` mode, low-scoring non-hack rollouts emit **3 sub-sequences** (`student`, `copy`, `hint_agent`) — check `__num_turns__`/`agent_role` in the batch; (3) on Sotopia, DITTO's reward curve exceeds GRPO **from the earliest steps**, the teacher–student gap is **positive and grows**, and the `secret` dimension does **not** degrade (vs GRPO which does). Expected Sotopia DITTO ≈ **0.470** vs GRPO ≈ 0.277.

### Phase E — Evaluation

> `eval.sh` is **referenced in README but MISSING** from this checkout. Reconstruct it from the `OpenAIAgentLoop` (api) / `AgentHubLoop` (local) paths and the val parquets.

```bash
# Intended (per README) — does not exist locally; reconstruct:
#   bash eval.sh local   # use the trained checkpoint as policy (agent_hub)
#   bash eval.sh api     # use an external model as policy (openai_agent)
ACTOR_MODEL_PATH=outputs/ditto-rl-sotopia/global_step_200 bash eval.sh local
```
Eval = run each task's `agent_loop` over its `*_val.parquet` (100 rows/task), with the trained model as policy (`agent_hub`) and judges via your endpoint; report per-task primary metric (`all/score` average per task; multi-metric tasks → mean).

**✅ Verification E**: per-task scores ordered DITTO ≥ GRPO ≥ Base, biggest deltas on Sotopia/UserLLM/TwinVoice; average in the ~0.7 range *if* judge quality approximates the paper's. Exact Table-1 match is not expected (100-instance sampling variance + judge drift).

---

## 8. Risks, Gaps & Discrepancies

**Missing files (README vs actual repo).** All three are advertised in README but **absent** in this checkout:
- `recipe/ditto/` ("Frozen recipe for the Ditto paper") — **no frozen config** to anchor the exact data mix, file ratios, multi-task setup, or final hyperparameters.
- `run_opd.sh` ("RL entry — on-policy distillation") — the OPD path is fully implemented in `verl/` but **no script enables it**.
- `eval.sh` ("Eval-only across the full eval suite") — **no eval driver**; must be reconstructed.

**Which mechanism = the paper's numbers?** Two distinct verbal-feedback implementations coexist: the **shipped `copy`/FoldGRPO reward path** (extra teacher sequence with `prompt_ids` swapped to the student's) and the **un-shipped OPD path** (EMA-teacher token-level KL). The paper's §3 math (Eqs. 6–10) most closely matches `copy`/FoldGRPO; confirm before assuming OPD. `run_rl.sh` defaults to `agent_version="default"` (vanilla GRPO) — **you MUST edit it to `copy`** for Ditto.

**SFT corpus does not exist.** `run_sft.sh` is an explicit *demo* with placeholder values, defaults to `Qwen3-VL-8B-Instruct` (a vision model, inconsistent with the text-only Ditto-8B), and expects `messages`-schema parquets that nobody ships. No scripted SFT→RL handoff. Whether the released model warm-starts from SFT at all is unconfirmed.

**Placeholder judge/model identities.** `gpt-5-nano`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `Qwen3-VL-8B-Instruct`, and 2026 dates are anachronistic/internal. The judge **directly produces the reward and the feedback `h`**, so judge choice is the single largest source of numerical divergence. Base-model identity is itself ambiguous: paper says `Qwen3-VL-8B-Instruct`; `run_rl.sh` defaults to `Qwen3-8B-Instruct` (LoRA targets/tokenizer differ).

**The parquet `prompt` is a dummy `"x"`.** Reward is computed in per-task Python, not from a `ground_truth` column. Reproduction requires porting each `agents/<task>/agent.py` and matching its exact `extra_info` key reads; unknown/typo'd `data_source` **silently** routes to `math_agent_loop` (mis-scored, not an error). Nested/stringified JSON (`coser.circumstance`, `sim_*.raw`, `behaviorchain.raw` ~183KB, `twinvoice.*` unicode-escaped Chinese, `lifechoices` real nested struct, numpy-array-repr `conversations`) must be parsed carefully; some gold/judge fields are empty in train parquets.

**Top 5 risks/gaps (ranked):**
1. **Missing `recipe/ditto/`, `run_opd.sh`, `eval.sh`** — no frozen recipe, no OPD script, no eval driver; eval must be hand-built.
2. **Placeholder judge models** drive the reward — exact Table-1 numbers are unreproducible; only relative orderings (DITTO > GRPO > Base) are verifiable.
3. **`run_rl.sh` defaults to vanilla GRPO** — Ditto requires manually setting `agent_version="copy"` (not exposed as an env var).
4. **SFT data + SFT→RL chaining absent** — the `messages`-schema corpus and per-dataset system prompts must be reconstructed from scratch.
5. **`prompt` placeholder + programmatic reward + data_source naming mismatches + silent math fallback** — the parquet alone is insufficient; each task's agent + reward function must be ported exactly.

**Lesser gaps:** notation ambiguities in the paper (`R(x,y1)` vs `J`'s `(r,h)`; whether teacher tokens in `L_group` are scored under `x` or `x,h`; Eq. 6 vs Eq. 9 distinct normalization pools — easy to get wrong); `hint_agent` returns absolute `new_reward` not the delta despite docstring; hard-coded magic numbers (hint gate `actor_avg≤4`, hack penalties /2,/4) with no config flags; external deps (Pangram AI-detector, EditLens — both gated off; runtime service at `:8005` for env tasks); per-task eval uses only 100 sampled instances (sampling variance); ablation figures (4–8) are images with no recoverable numeric targets.
