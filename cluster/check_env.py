#!/usr/bin/env python3
"""Sanity-check the TARPO environment.

    python3 cluster/check_env.py --no-gpu                 # login node
    python3 cluster/check_env.py --model models/Qwen...   # inside a GPU job

Verifies that the *vendored* transformers/trl/unsloth (the modified copies in
the repo root, which contain the ActionHead) are the ones being imported.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []


def check(label, fn, fatal=True):
    try:
        print(f"  {label:<28} {fn()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<28} {'FAILED' if fatal else 'warning'}: {type(exc).__name__}: {exc}")
        if fatal:
            FAILURES.append(label)


def vendored(mod):
    path = os.path.abspath(getattr(mod, "__file__", "") or "")
    tag = "VENDORED-OK" if path.startswith(REPO + os.sep) else "!! SITE-PACKAGES !!"
    if not path.startswith(REPO + os.sep):
        FAILURES.append(f"{mod.__name__} not vendored")
    return f"{getattr(mod, '__version__', '?')}  [{tag}]  {path}"


def cached_gsm8k():
    """gsm8k is downloaded at runtime by tarpo_gsm8k.py, so it must be cached
    before a job starts (compute nodes run with HF_HUB_OFFLINE=1)."""
    import glob
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for pattern in (os.path.join(hf_home, "hub", "datasets--openai--gsm8k"),
                    os.path.join(hf_home, "datasets", "*gsm8k*"),
                    os.path.join(hf_home, "datasets", "*", "*gsm8k*")):
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return "MISSING (run cluster/prepare_data.py on the login node)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="skip CUDA/model checks")
    ap.add_argument("--model", default=None, help="model dir to try loading through unsloth")
    args = ap.parse_args()

    print(f"cwd            : {os.getcwd()}")
    print(f"repo           : {REPO}")
    print(f"python         : {sys.version.split()[0]} ({sys.executable})")
    print(f"HOME           : {os.environ.get('HOME')}")
    print(f"TMPDIR         : {os.environ.get('TMPDIR')}")
    print(f"HF_HOME        : {os.environ.get('HF_HOME')}")
    print(f"WANDB_MODE     : {os.environ.get('WANDB_MODE')}")

    if os.getcwd() != REPO:
        print("\n!! Run this from the repo root, otherwise the vendored packages "
              "are not on sys.path and the ActionHead code will not be used.")
        FAILURES.append("wrong cwd")

    print("\ncore packages")
    import torch
    check("torch", lambda: f"{torch.__version__} (cuda build {torch.version.cuda})")
    if not args.no_gpu:
        check("torch.cuda.is_available", lambda: torch.cuda.is_available())
        check("gpu", lambda: torch.cuda.get_device_name(0))
        check("bf16 supported", lambda: torch.cuda.is_bf16_supported())

    print("\nvendored packages (must live inside the repo)")
    if not args.no_gpu:
        import unsloth  # noqa: F401  (must be imported before transformers/trl/peft)
        check("unsloth", lambda: vendored(sys.modules["unsloth"]))
    import transformers
    import trl
    check("transformers", lambda: vendored(transformers))
    check("trl", lambda: vendored(trl))
    check("ActionHead in qwen2", lambda: __import__(
        "transformers.models.qwen2.modeling_qwen2", fromlist=["ActionHead"]).ActionHead)

    print("\nsupport packages")
    from importlib.metadata import version as v
    for pkg in ("unsloth_zoo", "peft", "accelerate", "datasets", "bitsandbytes",
                "xformers", "wandb", "math-verify"):
        check(pkg, lambda p=pkg: v(p), fatal=pkg in ("unsloth_zoo", "peft", "accelerate", "datasets"))

    print("\ndata / models")
    check("data/MATH/train", lambda: f"{len(os.listdir('data/MATH/train'))} subject dirs"
          if os.path.isdir("data/MATH/train") else "MISSING (needed only by tarpo_math.py; "
          "run cluster/prepare_data.py)", fatal=False)
    check("gsm8k in HF cache", cached_gsm8k, fatal=False)

    if args.model and not args.no_gpu:
        print(f"\nloading {args.model} through unsloth (this takes a few minutes)")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=512,
            load_in_4bit=False, load_in_8bit=False, fast_inference=False,
        )
        check("model.action_head", lambda: type(model.action_head).__name__)
        check("action_head.custom_init", lambda: (model.action_head.custom_init(bias=[2.2, 0.0]), "ok")[1])
        check("tokenizer", lambda: type(tokenizer).__name__)

    print()
    if FAILURES:
        print("FAILED CHECKS: " + ", ".join(dict.fromkeys(FAILURES)))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
