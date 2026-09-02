# TARPO Reproduction — Handover

**Repo:** `ZachUOACS/TARPO-dissertation` (fork of `NKU-LITI/TARPO-master`)
**Author:** Zach Taylor (UPI `ztay632`), University of Auckland
**Status as of 2026-08-31:** 10 training runs complete, 6 evals complete, 4 evals running.
**Sibling study:** a separate HRPO (Yue et al., 2026) reproduction by the same author — see §12, its data is NOT in this repo.

This file is the single source of truth for the TARPO half of the project. It is written
for an agent that has not seen the work. Read §3 and §4 before running anything: the repo
contains vendored, modified libraries and several traps that will silently produce wrong
results.

---

## 1. What is being tested

TARPO (Zhang et al., 2026, arXiv 2606.05859) adds a **binary routing head** to an LLM that
decides, per token, whether the next reasoning step is a discrete token (`hard`) or a
continuous top-k embedding mixture (`soft`/latent). Backbone and router are trained jointly
with GRPO using one shared trajectory-level advantage.

**Paper's central claims** (the ones this study probes):
1. TARPO beats GRPO and HRPO on reasoning benchmarks.
2. The router *learns adaptive token-wise switching* — it is not a fixed heuristic.
3. TARPO improves token efficiency (fewer generated tokens).

**Research question here:** do these hold at the smallest scale the paper reports,
Qwen2.5-1.5B-Instruct on GSM8K, and do they survive a proper variance estimate?

---

## 2. Infrastructure

University of Auckland ML cluster, Slurm, 1×GPU per job, ~20 h per training run.
Full setup instructions: **`cluster/README.md`** (read it before touching the cluster).

```
Cluster host   foscsmlprd01.its.auckland.ac.nz   (password = <upi_password>:<2FA_code>)
Repo on cluster /data/ztay632/TARPO-dissertation
Venv            /data/ztay632/envs/tarpo
Models          /data/ztay632/models/Qwen2.5-1.5B-Instruct
W&B project     latent-reasoning  (entity ztay632-university-of-auckland), ONLINE works via proxy
```

Everything must live under `/data/<UPI>` — files elsewhere are deleted without notice.
Compute nodes have no direct internet; HF assets are pre-cached on the login node and jobs
run `HF_HUB_OFFLINE=1`. W&B reaches wandb.ai through `squid.auckland.ac.nz:3128` (verified,
job 15067), and all sbatch scripts now default to `WANDB_MODE=online`.

**Core commands**

```bash
# from the local machine
bash cluster/sync_to_cluster.sh push          # rsync up (excludes experiments/, data/, models/)
bash cluster/sync_to_cluster.sh pull          # metrics + logs back

# on the cluster
sbatch cluster/train_gsm8k.sbatch                       # defaults = paper config below
ACTION_BIAS="2.2 0.0" sbatch cluster/train_gsm8k.sbatch # override bias
CKPT=<path>/checkpoint-935 K=8 BS=4 sbatch cluster/eval.sbatch
python3 cluster/check_env.py                            # verifies vendored imports + cache staleness
python3 cluster/inspect_router.py <checkpoint>          # did the router move? (no GPU needed)
```

---

## 3. Codebase traps — READ THIS

### 3.1 The repo vendors modified copies of transformers, trl and unsloth
`./transformers`, `./trl`, `./unsloth` are **not** the upstream packages. The TARPO
`ActionHead` lives in `transformers/models/qwen2/modeling_qwen2.py:769`; the routing
generate loop is in `transformers/generation/utils.py` (~line 3300–3560).

Python puts the **script's own directory** on `sys.path`, not the cwd. The training and eval
scripts live in the repo root so they pick up the vendored copies automatically. Anything
under `cluster/` does not — `cluster/env.sh` exports `PYTHONPATH=$TARPO_DIR` to cover it.
Always run from the repo root. `cluster/check_env.py` prints `VENDORED-OK` when correct.

### 3.2 unsloth caches a generated trainer and never overwrites it
`PatchFastRL("GRPO", ...)` generates `./unsloth_compiled_cache/UnslothGRPOTrainer.py` from
the vendored trl source with `overwrite=False` ([unsloth/models/rl.py:560](unsloth/models/rl.py)).
**Any edit to `trl/` or `unsloth/` requires clearing that cache or it is silently ignored:**

```bash
RECOMPILE=1 sbatch cluster/train_gsm8k.sbatch      # clears it
```
`check_env.py` flags the cache as STALE when the vendored source is newer.

### 3.3 The live loss is NOT the one in trl
`trl/trainer/grpo_trainer.py:772` `compute_loss` is **dead code** — it calls
`_get_per_token_logps` with 4 args against a 5-arg signature. The executed objective is
unsloth's replacement, `grpo_compute_loss` in
**`unsloth/models/rl_replacements.py:256`**. Read that function, not the trl one.

### 3.4 Upstream bugs found
| # | Location | Issue | Status |
|---|---|---|---|
| 1 | `trl/trainer/grpo_trainer.py:570` | Unpacks 9 values from `generate()` but never passed `return_soft_metrics=True`; the default branch returns 4. **Training could not run as shipped** (`ValueError: not enough values to unpack`). | **FIXED** — kwarg added |
| 2 | `unsloth/models/rl_replacements.py:268` | Paper Eq. 7 defines the token loss with `𝟙[dₜ = Hard]`. `is_hard` is computed and never applied — the token loss runs at every step. Immaterial at 1% routing, material above 20%. | OPEN, documented |
| 3 | `unsloth/models/rl_replacements.py:305` | Action-KL coefficient α was hardcoded (as `beta * kl_action`, i.e. α=1, the paper default). The α=0 ablation was unreachable. | **FIXED** — see §4 |
| 4 | `trl/trainer/grpo_trainer.py:772` | Dead `compute_loss` (arity mismatch). | OPEN, harmless |
| 5 | `tarpo_math.py` | Labelled its experiment dir `-gsm8k-` even for MATH runs. | **FIXED** → `-math-` |

---

## 4. Modifications made to this fork (non-upstream)

All are deliberate and documented. An agent must not assume upstream behaviour.

**Training scripts** (`tarpo_gsm8k.py`, `tarpo_math.py`, `tarpo_dapo_math.py`):
- **Resume support.** Auto-resumes from the newest `checkpoint-*`; `--resume {auto,never,must}`.
  A wall-clock kill is recovered by resubmitting the identical command.
- **`--action_kl_alpha`** (default 1.0). Coefficient α on the action-head KL (paper Eq. 6);
  effective weight is `beta * alpha`. `0.0` = the paper's *w/o Action KL* ablation.
  Threaded through the whole autograd chain in `rl_replacements.py`
  (`grpo_compute_loss` → nested `compute_loss` → `accumulate_chunk` →
  `UnslothEfficientGRPO.forward`/`backward` → `grpo_accumulated_loss`).
  Verified: forward takes 15 args after ctx / backward returns 15; `argnums=(0,7)` still
  resolves to `new_hidden_states` / `new_action_logits` (the new param was appended, not inserted).
- **`--freeze_router`.** Control: sets `requires_grad_(False)` on the action head after the
  PEFT wrap so the router stays at initialisation. Routing still fires at the fixed rate;
  only the learning is removed. `patch_trainer_optimizer` filters on `requires_grad`, so the
  params drop out of the optimizer cleanly.
- **Experiment dir name** now encodes `seed`, and (when non-default) `alpha`, `lr_action_head`
  and `frozenrouter`, so variants cannot collide or silently resume each other.

**`trl/trainer/grpo_config.py`:** added `action_kl_alpha` field.

**`cluster/`** — all new, not upstream. See `cluster/README.md`.
A backup of the pre-patch `rl_replacements.py` is in the session scratchpad.

---

## 5. Experimental configuration

Paper settings (TARPO Appendix A, Tables 6–7), GSM8K / Qwen2.5-1.5B column:

```
group size g            4              soft token top-k     30
total train batch       32  (32 × 1)   action temperature   1.0
lr backbone             5e-6           temperature          0.5
lr action head          1e-4           LoRA                 r32 / α64
token KL β              0.005          max grad norm        0.1
action KL α             1.0            optimizer            AdamW-8bit, cosine + 0.1 warmup
routing weight λ        0.1            epochs               1  = 935 optimizer steps
action bias b₀          [4.6, 0]       precision            bf16
```

**Declared deviations** (state these in any write-up):

| Setting | Paper | Here | Why |
|---|---|---|---|
| Train lengths | 1024 / 1024 (TARPO Table 7) | **512 / 512** | HRPO Table 5 value — matches the HRPO baseline. Observed completions are 210–233 tokens, so the cap does not bind. |
| Accumulation | 8 × 4 (HRPO Table 4) | **32 × 1** | Mathematically identical optimizer step (advantages are standardised within each 4-completion group, not per micro-batch); ~2× faster. Per-step wall clock is therefore not comparable across studies. |
| Eval k | 32 | **8** | `avg@k` is k-invariant (it estimates single-sample accuracy), so avg@8 ≡ the paper's P@1. `pass@k` / `maj@k` are NOT comparable across k. |

**`b₀` is a logit-odds encoding of the initial routing rate:** `p(soft) = 1/(1+e^(b_hard−b_soft))`,
so `b₀ = ln((1−p)/p)`. The paper's values are round numbers: `[4.6,0]`→1%, `[2.2,0]`→10%,
`[0.85,0]`→30%. `[20,0]`→0% is our GRPO control.

---

## 6. Results — training (10 runs, all 935 steps, Qwen2.5-1.5B / GSM8K)

Values are means over the final 100 steps. `soft@0` is the mean over the first 20 steps.

| job | b₀ | seed | variant | soft@0 | soft@end | reward | compl. len | action entropy | KL median | KL spikes >10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 15307 | 20.0 | 42 | **GRPO control** | 0.00% | 0.00% | 0.775 | 232.5 | 0.0000 | 0.152 | 7 |
| 15067 | 4.6 | 42 | paper default | 0.91% | 1.01% | 0.795 | 231.7 | 0.052 | 0.125 | 11 |
| 15309 | 4.6 | 43 | replicate | 0.72% | 0.81% | 0.787 | 220.5 | 0.045 | 0.131 | 9 |
| 15271 | 2.2 | 42 | — | 8.83% | 9.39% | 0.765 | 225.3 | 0.313 | 0.208 | 25 |
| 15308 | 2.2 | 43 | replicate | 7.09% | 8.09% | 0.790 | 227.7 | 0.272 | 0.181 | 14 |
| 15306 | 0.85 | 42 | — | 20.09% | 21.78% | 0.770 | 217.2 | 0.528 | 0.303 | 38 |
| 15693 | 0.85 | 43 | replicate | 29.66% | **32.23%** | 0.769 | 210.0 | 0.627 | 0.377 | 32 |
| 15690 | 2.2 | 42 | **α = 0** | 9.38% | **0.09%** | **0.713** | 215.1 | 0.003 | 0.213 | 16 |
| 15691 | 2.2 | 42 | **head lr 1e-3** | 7.81% | 9.33% | 0.793 | 221.8 | 0.320 | 0.197 | 25 |
| 15692 | 2.2 | 42 | **frozen router** | 6.68% | 7.02% | 0.759 | 233.0 | 0.256 | 0.183 | 15 |

**α = 0 routing collapse, by decile** (soft %): 4.45 → 0.56 → 0.32 → 0.16 → 0.12 → 0.09 → …
First drops below 1% at **step 82** and never recovers.

---

## 7. Results — evaluation (GSM8K test, k=8, n=1319, identical protocol)

| b₀ | seed | variant | soft % | **avg@8** | 95% CI | pass@8 | maj@8 | #Tok | discrete |
|---|---|---|---|---|---|---|---|---|---|
| 20.0 | 42 | GRPO control | 0.00 | **69.11** | 67.24 – 70.97 | 91.28 | 80.82 | 251.6 | 251.6 |
| 4.6 | 42 | paper default | 0.99 | 69.20 | 67.34 – 71.06 | 92.12 | 80.67 | 257.2 | 254.6 |
| 4.6 | 43 | replicate | 0.82 | 68.05 | 66.16 – 69.94 | 91.36 | 79.15 | 241.4 | 239.5 |
| 2.2 | 42 | — | 9.69 | **69.48** | 67.62 – 71.35 | 91.51 | 79.61 | 244.7 | 221.0 |
| 2.2 | 43 | replicate | 7.97 | 68.93 | 67.07 – 70.80 | 91.58 | 79.38 | 250.2 | 230.2 |
| 0.85 | 42 | — | 22.58 | **67.28** | 65.38 – 69.17 | 90.75 | 79.08 | 235.8 | 182.5 |

**PENDING** (running as of handover): α=0, head-lr 1e-3, frozen router, b₀=0.85 seed 43.
Commands are in §11.

---

## 8. Findings

### F1 — The initial bias sets the routing rate; RL does not move it (under α=1)
Observed rate tracks `softmax(b₀)` and flatlines within ~300 steps:
1.00%→1.01%, 9.98%→9.39%, 29.94%→32.23%. `cluster/inspect_router.py` on checkpoint-935 of
the b₀=4.6 run: bias gap moved **4.6000 → 4.5938** (drift −0.0062) over 935 steps; weight std
grew 0.001 → 0.001724. With Adam at lr 1e-4, a perfectly consistent gradient would move the
bias by up to 0.09 — it achieved 7% of that, i.e. ~93% of the gradient cancels.
**Under α=1, `b₀` is effectively a fixed hyperparameter, not an initialisation.**

### F2 — Removing the action KL makes the router collapse, not migrate
α=0 drives routing from 9.98% to **below 1% by step 82** and 0.09% at convergence. So the
router **can** learn — decisively — and what it learns is *"stop using latent reasoning."*
This is the single most important result.
- It **contradicts the paper's Figure 3a**, where α=0 at 3B/MATH shows the soft ratio *rising*
  to 0.6–0.75.
- It **agrees with the paper's own Appendix B text**: "removing action KL leads to more
  unstable entropy and token-usage trajectories" and action KL "mitigates premature collapse
  to degenerate routing behaviors." Collapse to 0.09% is exactly a degenerate routing behaviour.
- Implication: at α=1 the KL anchor was *preventing the router from expressing a learned
  preference against latent reasoning*. The reported routing rates in the α=1 setting are an
  artifact of the anchor, not evidence of adaptive switching.
- Caveat: different scale and dataset from Figure 3a (1.5B/GSM8K vs 3B/MATH). Not yet tested at 3B.

### F3 — Routing is content-blind
Character composition inside vs outside soft spans, 10,066 spans from the b₀=0.85 run
(last 300 steps): digits **5.5% vs 5.3%** (enrichment 1.04×), operators **1.7% vs 1.6%**
(1.03×), letters 70.6% vs 66.3% (1.06×). Soft steps fire at the base rate on every token class.
The paper's Figure 4 claims TARPO "assigns higher soft probability to key mathematical tokens,
such as equations and operators" — **not reproduced**; enrichment is ~1.0×.
Mechanistically expected: `W_r` never leaves its `N(0, 0.001)` init, so
`ρ(·|hₜ) ≈ Softmax(b_r)` — independent of `hₜ`, i.e. a per-token coin flip.
Qualitatively, soft spans split words mid-token (`m[ow]`, `f[ertilizing]`, `1[2]0`) and
occasionally corrupt grammar (`[ we takes him twice]`).

### F4 — No accuracy benefit; seed noise exceeds every config effect
GRPO control **69.11** vs the mean of the four TARPO runs at b₀ 4.6/2.2 = **68.92**.
The b₀=4.6 seed gap is **1.15 pts**, larger than every TARPO-vs-control difference.
Excluding b₀=0.85, all five runs span 68.05–69.48 (1.43 pts) — the whole spread across three
routing rates is the size of one seed swap. The only possibly-real effect is b₀=0.85 at
**−1.83 vs control** (heavy routing hurts); its replicate eval is pending.

### F5 — The token-efficiency claim does not hold here
Two independent reasons:
1. The *discrete*-token reduction is **mechanical**: `hard = total × (1 − soft_ratio)`
   reproduces the reported figure to within 0.1 tokens in all six evals. Routing 22.6% of
   steps removes 22.6% of discrete tokens by definition. Each soft step still costs a
   forward pass, so **compute is unchanged**.
2. *Total* tokens (`#Tok`, the paper's metric) do not track routing: the b₀=4.6 seed gap is
   **15.8 tokens (6.1%)**, larger than the 12.4-token difference between bias settings.
   Range across all six evals is 235.8–257.2 with correlation −0.71 to routing rate, driven
   almost entirely by the single b₀=0.85 point.

### F6 — Training instability scales with routing rate
Steps with token KL > 10 (median KL is 0.13–0.38): 7 at 0% soft → 9/11 at ~1% → 14/25 at ~8–9%
→ 32/38 at 22–32%. Peaks reach 2.9e3–2.3e5. Clipped by `max_grad_norm=0.1`; all runs completed.

### F7 — 10× router LR does not unpin the router (under α=1)
`--lr_action_head 1e-3` (HRPO's value for its gating parameter Λ) leaves the rate at 9.33%,
essentially the 9.98% initialisation. Confirms F1's cause is the KL anchor, not the LR budget.

### F8 — Training the router contributes nothing
Frozen-router control reward **0.759** vs trained TARPO at the same bias (0.765 / 0.790) —
within noise. Eval pending, but training reward already suggests the learned router is surplus.

### F9 — Routing rate itself is seed-dependent at low bias
b₀=0.85: seed 42 converged to **21.78%**, seed 43 to **32.23%** — a 10-point spread at
identical configuration. (An earlier hypothesis that the bias→rate predictor systematically
under-delivers at high rates was **wrong**: seed 43 started at 29.66% vs the 29.94% prediction.
The seed-42 shortfall was a seed artifact.)

---

## 9. Mechanistic account

The router must learn *when* a latent step beats a discrete one from **one scalar reward per
trajectory** covering ~230 binary decisions. That credit assignment is close to hopeless, and
the data shows it failing exactly that way:

- α=1: the KL anchor pins `ρ` at `softmax(b₀)`; weights stay at init; routing is a
  content-blind coin flip at the initialised rate (F1, F3, F7).
- α=0: freed, the policy gradient finds that latent steps do not help and drives the rate to
  ~0 within 82 steps (F2).
- Either way the mechanism contributes nothing to accuracy (F4, F8), and its apparent token
  saving is definitional rather than computational (F5).

For **small models specifically**: latent reasoning's promise is more information per step
than a token ID can carry. A top-k probability-weighted embedding mixture at a *randomly
chosen* position carries a blurrier version of the same distribution, off the manifold the
pretrained backbone expects. At 1.5B, GSM8K competence comes from discrete CoT the model
already has; the binding constraints are arithmetic reliability and output format, neither of
which extra latent capacity addresses.

**Defensible thesis statement:** *the method's learning signal cannot train its own router at
this scale, so TARPO reduces to its initialisation — and that initialisation is a
hyperparameter, not a learned policy.*

---

## 10. Paper reference numbers (Qwen2.5-1.5B-Instruct, GSM8K, P@1 = avg over 32)

| Source | P@1 | 95% CI |
|---|---|---|
| TARPO, Table 1 | 70.76 | 69.01 – 72.52 |
| TARPO, Table 9, b₀=[4.6,0] + action KL *(matches our config)* | 69.96 | not published |
| TARPO, Table 9, b₀=[2.2,0] + action KL | 68.65 | not published |
| HRPO, Table 1 | 69.71 | 67.93 – 71.50 |
| GRPO, Table 1 | 69.04 | 67.24 – 70.83 |
| CoT, Table 1 | 60.89 | 59.03 – 62.74 |

Our b₀=4.6 run (**69.20**) reproduces Table 9's 69.96 to within 0.76 pts — inside our CI and
inside the paper's own 0.80-pt spread between its two reports of the same configuration
(69.96 vs 70.76). **Every RL method's interval overlaps every other one.** The paper's own
averaged improvement over GRPO is 0.52% P@1.

**On "is the paper wrong":** the defensible claim is *not reproduced at this cell, and the
paper's own statistics do not establish it either*. Untested here: 3B/7B backbones, MATH,
DAPO-MATH-17k, and Figure 3a's 3B/MATH α=0 cell. Do not over-claim.

---

## 11. What's left

### Immediate — evals for the four completed runs
```bash
cd /data/ztay632/TARPO-dissertation
P=experiments/Qwen2.5-1.5B-Instruct-gsm8k-tarpo-group4-lora32-temp0.5-len512-512
for D in $P-bias2.2-actemp1.0-topk30-weight0.1-seed42-alpha0.0 \
         $P-bias2.2-actemp1.0-topk30-weight0.1-seed42-headlr0.001 \
         $P-bias2.2-actemp1.0-topk30-weight0.1-seed42-frozenrouter \
         $P-bias0.85-actemp1.0-topk30-weight0.1-seed43; do
  CKPT=$D/checkpoint-935 K=8 BS=4 sbatch cluster/eval.sbatch
done
```
Questions they settle: does the α=0 collapse cost accuracy (training reward 0.713 says yes)?
Does the frozen router match trained TARPO (F8)? Is b₀=0.85's −1.83 real (F4)?

### High value, not yet run
1. **α=0 at b₀=0.85 and b₀=4.6** — is collapse universal, or specific to the 10% start?
2. **Re-run the F3 selectivity analysis on the α=0 run** before collapse (steps 0–82). If
   enrichment is still ~1.0× while the rate is actively moving, routing is content-blind
   under *every* setting the paper offers — a much stronger claim.
3. **Replicates for the GRPO control and b₀=20.0** — currently n=1 at both ends of the curve.
4. **k=32 eval** on the main runs, for `pass@32` / `maj@32` comparability with the paper's tables.
5. **Qwen2.5-3B / MATH with b₀=[2.2,0], α=0** — the paper's own Figure 3a cell. This is the
   only way to test whether F2's collapse is scale/dataset-specific. Needs `data/MATH`
   (`python3 cluster/prepare_data.py`) and the 3B download.

### Analysis owed
- Fold the trajectory comparison (§8 F3) and the selectivity table into the slide deck.
- Decide whether to patch upstream bug #2 (`is_hard` unused) and re-run — it only matters
  above ~20% routing, i.e. for the b₀=0.85 and α=0-early regimes.

---

## 12. The HRPO study (sibling — data NOT in this repo)

The author has a separate HRPO reproduction. Known facts:
- 934–935 optimizer steps for one GSM8K epoch (consistent with g=4/batch 32 **or** g=8/batch 64
  — both give 8 prompts per step, so the step count cannot disambiguate).
- ~50 s/iteration, consistent with HRPO Table 4's `8 × accumulation 4` layout.

**To make the comparison controlled, the agent must confirm from the HRPO run's own config:**
group size, total train batch, train lengths, and eval `k`. Then evaluate the HRPO checkpoint
through **this** repo's protocol (`K=8 BS=4`, same 1319 questions) — comparing against the
HRPO paper's published numbers instead is not a controlled comparison.

Shared knobs that must match for a fair TARPO-vs-HRPO claim: g, total batch, lengths, lr 5e-6,
β 0.005, temperature 0.5, LoRA r32/α64, cosine + 0.1 warmup, AdamW-8bit, 1 epoch.
Note HRPO's "hidden ratio" (`√(1−aₜ²)`, a continuous gate value) is **not** the same quantity
as TARPO's `soft_ratio` (a sampled binary rate) — do not plot them on one axis.

---

## 13. File map

```
HANDOVER.md                     this file
cluster/README.md               full cluster setup + run instructions
cluster/inspect_router.py       did the action head move? reads checkpoint safetensors, no GPU
cluster/check_env.py            vendored-import + compiled-cache-staleness check
cluster/train_gsm8k.sbatch      main training job (paper config in comments)
cluster/eval.sbatch             evaluation, any benchmark
cluster/sync_to_cluster.sh      rsync up/down

tarpo_gsm8k.py                  GSM8K training entry point (+ our resume/alpha/freeze flags)
utils.py                        prompts, answer extraction, reward function
unsloth/models/rl_replacements.py   THE LIVE LOSS (grpo_compute_loss, line 256)
transformers/generation/utils.py    routing generate loop (~3300-3560)
transformers/models/qwen2/modeling_qwen2.py:769   ActionHead

logs/tarpo-gsm8k-<jobid>.out.log    per-step metrics dicts + bracketed trajectory samples
experiments/<config>/checkpoint-935/eval_gsm8k_k8_batch4/eval_metrics_final.json
```

**Parsing the logs:** per-step metrics are Python dicts printable with
`re.finditer(r"\{'loss'.*?\}", text)` + `ast.literal_eval`. Keys include `reward`,
`soft_ratio`, `hard_ratio`, `action_entropy`, `token_entropy`, `completion_length`, `kl`,
`grad_action_norm`, `grad_hidden_norm`, `topk_entropy`, `embed_dist`.
Blocks matching `=== Step N Trajectory Sample ===` contain the generated text with
**soft-routed spans wrapped in `[square brackets]`** — this is the only place per-token
routing decisions are recoverable (the eval jsonl stores hard-token placeholders only).

---

## 14. Interpreting results — statistical guardrails

1. **n=1 proves nothing here.** Seed-to-seed variation is 1.15 pts avg@8 and up to 10 pts of
   routing rate. Any claimed effect must exceed that.
2. **`avg@k` is k-invariant** — avg@8 is directly comparable to the paper's P@1.
   **`pass@k` and `maj@k` are not** — never compare pass@8 to their pass@32.
3. **Per-step wall clock is not comparable** to the HRPO study (different accumulation layout).
   Compare `completion_length` and reward instead.
4. The CIs reported by the eval script are over question-level avg@k and capture sampling
   noise only — **not** training-run variance. Seed replicates are the only estimate of that.
5. Prefer TARPO Table 9 (69.96) over Table 1 (70.76) as the reproduction target: Table 9 is
   the b₀=[4.6,0] + action-KL row that matches our configuration.
