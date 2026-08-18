#!/usr/bin/env python3
"""Fetch every dataset TARPO needs. RUN ON THE LOGIN NODE (needs the proxy).

    source cluster/env.sh && source $VENV_DIR/bin/activate
    python3 cluster/prepare_data.py

What it does:
  1. Materialises the Hendrycks MATH training set into  data/MATH/train/<type>/<i>.json
     with the {"problem": ..., "solution": ...} layout that tarpo_math.py expects
     (it walks data/MATH/<split>/<folder>/*.json and pulls \\boxed{} out of
     'solution', so solutions must keep their LaTeX \\boxed answers).
  2. Warms the HF cache for the datasets the eval scripts download at runtime
     (HuggingFaceH4/MATH-500, openai/gsm8k) so GPU jobs can run with
     HF_HUB_OFFLINE=1.

The MATH source repo on the Hub has moved around over the years, so several
candidates are tried in order; the first one that yields problem+solution wins.
"""
import argparse
import json
import os
import re
import sys

SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]

# (repo_id, list_of_configs_or_None, extra load_dataset kwargs)
MATH_CANDIDATES = [
    ("EleutherAI/hendrycks_math", SUBJECTS, {"trust_remote_code": True}),
    ("EleutherAI/hendrycks_math", SUBJECTS, {}),
    ("nlile/hendrycks-MATH-benchmark", [None], {}),
    ("qwedsacf/competition_math", [None], {}),
]

SLUG = re.compile(r"[^a-z0-9]+")


def slug(text):
    return SLUG.sub("_", str(text).strip().lower()).strip("_") or "unknown"


def rows_from(repo, configs, kwargs, split):
    from datasets import load_dataset
    out = []
    for cfg in configs:
        ds = load_dataset(repo, cfg, **kwargs) if cfg else load_dataset(repo, **kwargs)
        if split not in ds:
            raise KeyError(f"split '{split}' not in {repo} ({list(ds)})")
        d = ds[split]
        cols = set(d.column_names)
        if not {"problem", "solution"} <= cols:
            raise KeyError(f"{repo} has columns {sorted(cols)}, need problem+solution")
        subject = cfg or ("type" if "type" in cols else "subject" if "subject" in cols else None)
        for row in d:
            out.append({
                "problem": row["problem"],
                "solution": row["solution"],
                "level": row.get("level", ""),
                "type": row.get("type", row.get("subject", cfg)) or "all",
            })
    return out


def write_math(rows, root, split):
    dest = os.path.join(root, split)
    os.makedirs(dest, exist_ok=True)
    counters = {}
    boxed = 0
    for row in rows:
        folder = os.path.join(dest, slug(row["type"]))
        os.makedirs(folder, exist_ok=True)
        idx = counters.get(folder, 0)
        counters[folder] = idx + 1
        with open(os.path.join(folder, f"{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False)
        boxed += "\\boxed" in row["solution"]
    print(f"  wrote {len(rows)} problems into {dest} "
          f"({len(counters)} folders, {boxed} with a \\boxed answer)")
    if boxed < 0.9 * len(rows):
        print("  !! most solutions have no \\boxed{...}; the reward function will "
              "score everything 0. Check the source dataset.")


def do_math(args):
    root = args.math_root
    if os.path.isdir(os.path.join(root, "train")) and not args.force:
        n = sum(len(fs) for _, _, fs in os.walk(os.path.join(root, "train")))
        print(f"MATH: {root}/train already exists with {n} files - skipping (--force to redo)")
        return
    last = None
    for repo, configs, kwargs in MATH_CANDIDATES:
        try:
            print(f"MATH: trying {repo} (configs={configs}, kwargs={kwargs}) ...")
            rows = rows_from(repo, configs, kwargs, "train")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> {type(exc).__name__}: {exc}")
            last = exc
            continue
        write_math(rows, root, "train")
        try:
            write_math(rows_from(repo, configs, kwargs, "test"), root, "test")
        except Exception as exc:  # noqa: BLE001
            print(f"  (no test split from this source: {exc})")
        return
    print(f"\nCould not build data/MATH from any candidate. Last error: {last}", file=sys.stderr)
    print("Fix: download the original MATH archive by hand and unpack it so that\n"
          f"  {root}/train/<subject>/<n>.json  each contain 'problem' and 'solution'.",
          file=sys.stderr)
    sys.exit(1)


def do_cache(_args):
    from datasets import load_dataset
    for repo, cfg in [("HuggingFaceH4/MATH-500", None), ("openai/gsm8k", "main")]:
        try:
            ds = load_dataset(repo, cfg) if cfg else load_dataset(repo)
            print(f"cached {repo}: " + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))
        except Exception as exc:  # noqa: BLE001
            print(f"!! failed to cache {repo}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-root", default="data/MATH")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-math", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    a = ap.parse_args()
    if not a.skip_math:
        do_math(a)
    if not a.skip_cache:
        do_cache(a)
    print("\nDone. HF cache lives in", os.environ.get("HF_HOME", "~/.cache/huggingface"))
