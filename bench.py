"""
Measure how much each speedup helps on this machine.

Starts from the plainest possible setup, then switches on one optimization at a
time and times a burst of training steps. Speedups that help a lot on one GPU
can do nothing (or fail) on another, so measure instead of guessing.

Run:  uv run bench.py                              # medium preset
      uv run bench.py --config configs/small.toml
      uv run bench.py --batch_size 128             # any train.py flag works
"""

import copy
import json
import sys
import time

import numpy as np
import torch

import train
from model import GPT, GPTConfig

WARMUP_STEPS = 10  # not timed: the first steps include one-off setup (and compiling)
TIMED_STEPS = 40

# Each variant adds one change to the one before it.
VARIANTS = [
    ("baseline: manual attention, float32", dict(sdpa=False, dtype="float32", fused=False, compile=False)),
    ("+ fused attention (sdpa)",            dict(sdpa=True)),
    ("+ bfloat16 autocast",                 dict(dtype="bfloat16")),
    ("+ fused AdamW",                       dict(fused=True)),
    ("+ torch.compile",                     dict(compile=True)),
]


def sync(device):
    """GPU work runs asynchronously; wait for it to finish before reading the clock."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_variant(args, data, vocab_size, device):
    torch.manual_seed(args.seed)
    cfg = GPTConfig(vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, sdpa=args.sdpa,
                    grad_checkpoint=args.grad_checkpoint)
    model = GPT(cfg).to(device)
    optimizer = train.make_optimizer(model, args)
    fast_model = torch.compile(model) if args.compile else model

    def step():
        x, y = train.get_batch(data, args.batch_size, args.block_size, device)
        return train.train_step(fast_model, optimizer, x, y, args, device)

    for _ in range(WARMUP_STEPS):
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    sync(device)
    return (time.perf_counter() - t0) / TIMED_STEPS


def main():
    if "--config" not in sys.argv:
        sys.argv += ["--config", "configs/medium.toml"]
    args = train.get_args()
    device = train.pick_device()
    data_dir = train.DATA_DIR / args.dataset
    vocab_size = len(json.loads((data_dir / "meta.json").read_text())["chars"])
    data = np.fromfile(data_dir / "train.bin", dtype=np.uint16)

    name = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"torch {torch.__version__} | {name} | n_layer={args.n_layer} n_embd={args.n_embd} "
          f"block_size={args.block_size} batch_size={args.batch_size}\n")
    print(f"{'variant':38s} {'ms/step':>8s} {'tokens/s':>10s} {'speedup':>8s}")

    current, baseline = args, None
    for label, changes in VARIANTS:
        trial = copy.copy(current)
        vars(trial).update(changes)
        try:
            sec = time_variant(trial, data, vocab_size, device)
        except Exception as e:  # e.g. torch.compile isn't supported on every backend
            print(f"{label:38s} failed: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
            continue  # skip this change, keep stacking the rest
        current = trial
        baseline = baseline or sec
        tokens_per_sec = args.batch_size * args.block_size / sec
        print(f"{label:38s} {sec * 1000:8.1f} {tokens_per_sec:10,.0f} {baseline / sec:7.2f}x")
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
