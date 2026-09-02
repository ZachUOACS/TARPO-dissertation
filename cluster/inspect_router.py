#!/usr/bin/env python3
"""Did the TARPO action head (router) actually train?

    python3 cluster/inspect_router.py experiments/<run>/checkpoint-935

Runs on the login node in seconds - it only reads the adapter safetensors, no
GPU, no unsloth, no model load.

At initialization custom_init() sets
    head.weight ~ Normal(0, 0.001)      (transformers/models/qwen2/modeling_qwen2.py)
    head.bias   = action_bias, e.g. [4.6, 0.0]   index 0 = hard, 1 = soft
so a router that never moved still looks like that. The action logit for a
token is  w . h + b, so compare BOTH the bias drift and the weight scale: the
weights can shift the logit far faster than the bias alone because 1536 of them
contribute at once.
"""
import argparse
import glob
import math
import os
import sys


def load_tensors(ckpt):
    files = (glob.glob(os.path.join(ckpt, "*.safetensors"))
             + glob.glob(os.path.join(ckpt, "**", "*.safetensors"), recursive=True))
    if not files:
        sys.exit(f"no .safetensors under {ckpt}")
    from safetensors import safe_open
    found = {}
    for path in sorted(set(files)):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if "action_head" in key:
                    found[key] = (f.get_tensor(key), path)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--bias", type=float, nargs="+", default=[4.6, 0.0],
                    help="the --action_bias the run was launched with")
    args = ap.parse_args()

    found = load_tensors(args.checkpoint)
    if not found:
        sys.exit("no action_head tensors in this checkpoint - modules_to_save did "
                 "not capture the router, which would mean it was never trained")

    print(f"checkpoint: {args.checkpoint}\n")
    for key, (t, path) in sorted(found.items()):
        t = t.float()
        print(f"{key}")
        print(f"   file  {os.path.relpath(path, args.checkpoint)}")
        print(f"   shape {tuple(t.shape)}   mean {t.mean():+.6f}   std {t.std():.6f}   "
              f"absmax {t.abs().max():.6f}")
        if key.endswith("bias") and t.numel() == 2:
            hard, soft = t[0].item(), t[1].item()
            init_gap = args.bias[0] - args.bias[1]
            gap = hard - soft
            p_soft = 1.0 / (1.0 + math.exp(gap))
            print(f"   bias  hard={hard:+.4f}  soft={soft:+.4f}  gap={gap:+.4f} "
                  f"(init gap {init_gap:+.4f}, drift {gap - init_gap:+.4f})")
            print(f"   -> p(soft) from the bias alone = {100 * p_soft:.2f}% "
                  f"(init {100 / (1 + math.exp(init_gap)):.2f}%)")
        if key.endswith("weight"):
            print(f"   weight std vs init 0.001: ratio {t.std() / 0.001:.2f}x")
            if t.shape[0] == 2:
                d = (t[0] - t[1])
                print(f"   ||w_hard - w_soft|| = {d.norm():.4f}  "
                      f"(a hidden state of unit norm can shift the gap by up to this)")
        print()

    print("Reading it: if the bias gap is still ~the init gap AND the weight std is "
          "~1x of 0.001, the router never moved and every completion stayed hard-routed. "
          "A weight std well above 1x means the router IS learning, just not toward soft.")


if __name__ == "__main__":
    main()
