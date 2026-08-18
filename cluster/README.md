# Running TARPO on the UoA ML cluster (foscsmlprd01/02/03)

Everything here follows the March 2026 cluster guidelines: all files under
`/data/<UPI>`, `HOME`/`TMPDIR` redirected there, internet only through
`squid.auckland.ac.nz:3128`, and **all** GPU work submitted through Slurm.

```
cluster/
  env.sh                  sourced by everything: paths, proxy, HF/W&B env
  setup_env.sh            one-time venv build              (login node)
  requirements-cluster.txt pinned dependency set
  download_models.sh      base model -> /data/<UPI>/models (login node)
  prepare_data.py         MATH + eval dataset cache        (login node)
  check_env.py            verifies the vendored packages, CUDA, ActionHead
  smoke_test.sbatch       30-min GPU proof-of-life         (Slurm)
  train_gsm8k.sbatch      GSM8K training  <- start here    (Slurm)
  train_math.sbatch       MATH training                    (Slurm)
  eval.sbatch             evaluation, any benchmark        (Slurm)
  sync_wandb.sh           push offline W&B runs            (login node)
```

## Quickstart: GSM8K on Qwen2.5-1.5B

The cheapest way into this repo - smallest backbone in the paper, and GSM8K
needs no local dataset build (it comes from the HF cache).

```bash
# --- login node ---
bash cluster/setup_env.sh                                   # ~15 min
bash cluster/download_models.sh Qwen/Qwen2.5-1.5B-Instruct
python3 cluster/prepare_data.py --skip-math                 # caches gsm8k only
echo <wandb-key> > /data/$USER/.wandb_key

# --- slurm ---
mkdir -p logs
sbatch cluster/smoke_test.sbatch     # 30 min, proves the stack works
sbatch cluster/train_gsm8k.sbatch    # defaults: 1.5B, action_bias 4.6 0.0
squeue -u $USER && tail -f logs/tarpo-gsm8k-*.out.log
```

Everything below is the same walkthrough in detail, plus MATH.

## 0. Log in

```bash
ssh <UPI>@foscsmlprd01.its.auckland.ac.nz      # password = <upi_password>:<2FA>
```
Off campus, connect the university VPN first. New accounts take 1-2 hours to
sync before the first login works.

## 1. Clone into /data/<UPI>

```bash
export UPI=$USER
mkdir -p /data/$UPI && cd /data/$UPI
export https_proxy=http://squid.auckland.ac.nz:3128 http_proxy=$https_proxy
git clone https://github.com/ZachUOACS/TARPO-dissertation.git
cd TARPO-dissertation
```

Nothing may live in `/home` - files outside `/data/<UPI>` are deleted without
notice, and you have a 300 GB quota. Add the redirects to your shell profile so
interactive work also stays inside `/data`:

```bash
echo 'source /data/'"$UPI"'/TARPO-dissertation/cluster/env.sh' >> /data/$UPI/.bashrc
```

## 2. Build the environment (login node, ~15 min)

```bash
bash cluster/setup_env.sh
```

This creates `/data/<UPI>/envs/tarpo` and installs, through the proxy:
torch 2.6.0+cu124, xformers 0.0.29.post3, then everything in
`requirements-cluster.txt`.

> **Why these versions.** `./transformers`, `./trl` and `./unsloth` in the repo
> root are *modified* copies - the TARPO `ActionHead` lives in
> `transformers/models/qwen2/modeling_qwen2.py`, and `GRPOConfig` gains
> `action_temperature`, `soft_top_k`, `action_loss_weight`. Those directories
> shadow the pip-installed packages **only when python is launched from the repo
> root**, which is why every job does `cd $TARPO_DIR` first. The pip versions are
> pinned to the same releases the vendored code came from (transformers 4.50.3,
> trl 0.15.2, unsloth_zoo 2025.3.x to match vendored unsloth 2025.3.19) so the
> version checks inside unsloth agree with the code actually being imported.
> The repo's own `requirements.txt` is a full-system `pip freeze` (it contains
> `cloud-init`, `ufw`, `certbot`, ...) and will not install on the cluster - use
> `cluster/requirements-cluster.txt` instead.

Check it:

```bash
cd /data/$UPI/TARPO-dissertation
source cluster/env.sh && source $VENV_DIR/bin/activate
python3 cluster/check_env.py --no-gpu     # login node has no GPU
```
Every vendored package must report `VENDORED-OK`.

## 3. Models and data (login node - compute nodes have no internet)

```bash
bash cluster/download_models.sh Qwen/Qwen2.5-1.5B-Instruct
python3 cluster/prepare_data.py --skip-math   # gsm8k + MATH-500 cache only

# later, when you move on to MATH / the 3B backbone:
bash cluster/download_models.sh Qwen/Qwen2.5-3B-Instruct
python3 cluster/prepare_data.py               # also builds data/MATH
```

* `download_models.sh` saves to `/data/<UPI>/models/<name>` **and** symlinks
  `<name>` into the repo root. The symlink is not optional: the eval scripts
  infer the base model from the checkpoint path and then call
  `from_pretrained("Qwen2.5-3B-Instruct")` with no `Qwen/` prefix, so that bare
  name has to resolve as a local directory.
* `prepare_data.py` pre-caches `openai/gsm8k` and `HuggingFaceH4/MATH-500` so
  GPU jobs can run with `HF_HUB_OFFLINE=1`, and (without `--skip-math`) builds
  `data/MATH/train/<subject>/<n>.json`, the layout `tarpo_math.py` walks. It
  tries several Hub mirrors of Hendrycks MATH in turn and says so if none work.
  **GSM8K training needs no local data at all** - `tarpo_gsm8k.py` calls
  `load_dataset('openai/gsm8k')` and reads it from the cache.
* Gated repos (Llama): `echo <hf_token> > /data/$UPI/.hf_token` first.

## 4. Weights & Biases

The training scripts hardcode `report_to="wandb"` and
`os.environ["WANDB_PROJECT"] = "latent-reasoning"`, so W&B must work.

```bash
echo <key from https://wandb.ai/authorize> > /data/$UPI/.wandb_key
chmod 600 /data/$UPI/.wandb_key
echo <your wandb username or team> > /data/$UPI/.wandb_entity   # optional
```

Jobs default to `WANDB_MODE=offline` (compute nodes cannot reach wandb.ai), so
runs are written to `/data/<UPI>/wandb/` and uploaded afterwards from the login
node:

```bash
bash cluster/sync_wandb.sh
```

Want live logging instead? Try `WANDB_MODE=online sbatch ...` - `env.sh` exports
`HTTPS_PROXY`, which the wandb client honours. If the job stalls at
`wandb: Currently logged in as ...` the compute node cannot reach the proxy;
go back to offline + sync.

* Project name: change line 14 of `tarpo_math.py` / `tarpo_gsm8k.py` (the
  hardcoded `os.environ["WANDB_PROJECT"]` overrides any env var you export).
* Run name: `export WANDB_NAME=tarpo-3b-math-bias2.2` before `sbatch`.

## 5. Submit jobs

```bash
mkdir -p logs                      # Slurm will not create the log directory
sbatch cluster/smoke_test.sbatch   # do this one first
squeue -u $USER
tail -f logs/tarpo-smoke-*.out.log
```

The smoke test loads the 1.5B model through unsloth, asserts the `ActionHead` is
present, and runs a few real GRPO steps. When it passes:

```bash
sbatch cluster/train_gsm8k.sbatch                                 # 1.5B on GSM8K
MODEL_NAME=Qwen2.5-3B-Instruct ACTION_BIAS="2.2 0.0" \
  sbatch cluster/train_gsm8k.sbatch                               # 3B on GSM8K
sbatch cluster/train_math.sbatch                                  # 3B on MATH
```

Defaults per script: `train_gsm8k.sbatch` = Qwen2.5-1.5B-Instruct, bias
`4.6 0.0`, lengths 512/512; `train_math.sbatch` = Qwen2.5-3B-Instruct, bias
`2.2 0.0`, lengths 1024/1024. Extra flags pass straight through:
`sbatch cluster/train_gsm8k.sbatch --lora_rank 64 --max_completion_length 1024`.

The GSM8K run writes to
`experiments/Qwen2.5-1.5B-Instruct-gsm8k-tarpo-group8-lora32-temp0.5-len512-512-bias4.6-actemp1.0-topk10-weight0.1/`,
checkpointing every 250 optimizer steps.

Evaluation (`--checkpoint_path` must be a `checkpoint-*` dir, or the experiment
dir if you saved a final model):

```bash
CKPT=experiments/Qwen2.5-1.5B-Instruct-gsm8k-tarpo-group8-lora32-temp0.5-len512-512-bias4.6-actemp1.0-topk10-weight0.1/checkpoint-250 \
K=8 sbatch cluster/eval.sbatch                    # GSM8K is the default script

EVAL_SCRIPT=eval_tarpo_math_avg.py K=32 BS=2 CKPT=... sbatch cluster/eval.sbatch
```
Results land inside the checkpoint directory. `K` is the Pass@k sample count -
the paper uses 32, but 4-8 is enough to see whether a run is learning.

* `eval_tarpo_gsm8k_avg.py` declares `--data_path` as required and then never
  reads it (it pulls gsm8k from the HF cache); `eval.sbatch` passes a dummy
  value for you.
* The AMC / ARC-C / GPQA / OlympiadBench / HumanEval scripts really do need a
  local benchmark JSON: `EVAL_SCRIPT=eval_tarpo_amc_avg.py DATA_PATH=data/amc23.json ...`.
  They download nothing, so you supply those files yourself.

## 6. Resuming after a time-out or pre-emption

The training scripts resume automatically. `tarpo_math.py`, `tarpo_gsm8k.py` and
`tarpo_dapo_math.py` look for the newest `checkpoint-*` in the experiment
directory (`transformers.trainer_utils.get_last_checkpoint`) and hand it to
`trainer.train(resume_from_checkpoint=...)`, so a job killed by the wall clock
is continued by **resubmitting the identical command**:

```bash
sbatch cluster/train_gsm8k.sbatch        # first submission: fresh start
sbatch cluster/train_gsm8k.sbatch        # after a time-out: continues from checkpoint-N
```

Checkpoints are written every 250 optimizer steps, keeping the last 3
(`save_steps` / `save_total_limit` in the training script). Optimizer, LR
schedule and step count are all restored; the rollouts of the interrupted step
are resampled.

A `--resume` flag controls it:

| value | behaviour |
|---|---|
| `auto` (default) | continue from the newest checkpoint if one exists; if the directory exists but holds no checkpoint, refuse to touch it |
| `never` | the original behaviour - refuse to start if the experiment directory is non-empty |
| `must` | fail unless there is a checkpoint to resume, useful in a requeue script that must never silently restart from scratch |

```bash
sbatch cluster/train_gsm8k.sbatch --resume never
```

Because the experiment directory name is derived from the hyperparameters, only
the ones encoded in that name (model, group size, LoRA rank, temperatures,
lengths, bias, top-k, loss weight) protect you from resuming into a differently
configured run. Change `--lr`, `--beta` or `--seed` and the path stays the same,
so the resumed run keeps the *old* optimizer and schedule state - move the old
directory aside if you meant to start over.

Two related notes:

* **W&B**: each resubmission starts a new W&B run, so a long training shows up
  as several segments. To stitch them into one, set a fixed id before
  submitting: `WANDB_RUN_ID=tarpo-gsm8k-1p5b WANDB_RESUME=allow sbatch ...`.
* `tarpo_math.py` used to name its output directory `-gsm8k-` even for MATH
  (upstream copy-paste). That is now `-math-`, so a MATH run and a GSM8K run
  with identical hyperparameters can no longer resume into each other.

## Sizing the run

* **One GPU is the right request.** This unsloth build states plainly that it
  does not support multi-GPU, so `--gres=gpu:1` (no Slack approval needed) is
  all you can use.
* TRL requires `per_device_train_batch_size` to be divisible by `group_size`.
  The sbatch defaults use `8 x 8 accumulation`, i.e. one prompt group per step
  and the paper's effective batch of 64 completions. The README's
  `--per_device_train_batch_size 64` means 64 sequences of up to 2048 tokens
  resident at once - raise the per-device value only while watching
  `nvidia-smi`, and drop `--max_completion_length` before dropping the group
  size (group size is what GRPO's advantage estimate depends on).
* A full epoch does not fit in 24 h on one GPU: GSM8K is 7.47k prompts and MATH
  ~7.5k, each needing 8 sampled completions with plain HF generate
  (`use_vllm=False`). Expect a day or more of wall clock per epoch even for the
  1.5B. Just resubmit after each time-out (step 6 - resume is automatic) and
  quote the step count you reached rather than assuming the epoch finished.

## Gotchas

| Symptom | Cause |
|---|---|
| `AttributeError: 'Qwen2ForCausalLM' object has no attribute 'action_head'` | Python imported site-packages `transformers`, not the vendored one. You are not in the repo root. |
| `Experiment ... already exists. Exiting...` | The experiment directory is non-empty but has no `checkpoint-*` to resume from (e.g. a run that died before step 250). Move it aside and resubmit. |
| Job dies instantly, empty log | `logs/` did not exist when you submitted. |
| Hangs / `ConnectionError` inside a job | Compute nodes have no internet. Everything must be pre-downloaded on the login node; jobs run with `HF_HUB_OFFLINE=1`. |
| `ImportError: cannot import name ... from unsloth_zoo` | Wrong unsloth_zoo month. Pin inside `2025.3.17`-`2025.3.x`; a newer one expects a newer transformers than the vendored 4.50.3. |
| CUDA OOM | Lower `--per_device_train_batch_size` (keeping it a multiple of `group_size`), then `--max_completion_length`. |
| Slow/failed pip | Pass `--proxy http://squid.auckland.ac.nz:3128`, which `setup_env.sh` already does. |
| `sbatch: error: Memory specification can not be satisfied` | Drop or lower the `#SBATCH --mem=96G` line - the partition may not do memory accounting the way these scripts assume. |
