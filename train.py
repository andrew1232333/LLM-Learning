"""
Step 3: training. Show the model real text, measure how wrong its
next-token guesses are (the loss), and nudge every weight slightly in
the direction that would have made it less wrong. Repeat thousands of times.

Run:  uv run train.py
      uv run train.py --config configs/medium.toml        # use a preset
      uv run train.py --config configs/medium.toml --lr 3e-4   # preset, one change
      uv run train.py --max_iters 500                     # quick test
      uv run train.py --dtype float32                     # full precision, to compare
"""

import argparse
import contextlib
import json
import math
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "out"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, help="TOML file of settings; flags on the command line override it")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR, help="where the checkpoint and settings are saved")
    p.add_argument("--dataset", default="shakespeare", help="folder under data/ made by prepare.py")
    # model size
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10, help="print the training loss every N steps")
    p.add_argument("--ckpt_interval", type=int, default=100, help="save last.pt every N steps")
    p.add_argument("--eval_interval", type=int, default=250, help="measure train/val loss every N steps")
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    # speed: these change how fast training runs, not what the model learns
    # (see bench.py to measure each one on your machine)
    p.add_argument("--sdpa", action=argparse.BooleanOptionalAction, default=False,
                   help="fused attention kernel")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16",
                   help="precision for the forward/backward pass")
    p.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True,
                   help="fused AdamW optimizer step")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="torch.compile the model (slow first step, faster after)")
    # memory
    p.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                   help="recompute activations during backprop: less memory, ~30%% slower")

    # Two passes: read the config file first and make its values the new defaults,
    # then parse again so anything typed on the command line still wins.
    args, _ = p.parse_known_args()
    if args.config:
        settings = tomllib.loads(args.config.read_text())
        unknown = set(settings) - set(vars(args))
        if unknown:
            p.error(f"unknown settings in {args.config}: {', '.join(sorted(unknown))}")
        p.set_defaults(**settings)
    args = p.parse_args()
    args.out_dir = Path(__file__).parent / args.out_dir  # relative paths are relative to the project
    return args


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def get_batch(data, batch_size, block_size, device):
    """Grab random chunks of text. The target is the input shifted by one:
    for input  "hello worl", the targets are "ello world".
    So every position in every chunk is its own little prediction exercise."""
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, args, device):
    model.eval()  # turns off dropout
    out = {}
    for name, data in splits.items():
        with autocast(args, device):
            losses = [model(*get_batch(data, args.batch_size, args.block_size, device))[1].item()
                      for _ in range(args.eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def autocast(args, device):
    """Mixed precision: run the big matrix multiplies in bfloat16 (half the bits, so
    roughly twice the math per second on most GPUs) while the weights themselves
    stay in float32, so tiny updates aren't rounded away."""
    if args.dtype == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16)


def make_optimizer(model, args):
    # AdamW: gradient descent with momentum and per-weight step sizes. The default
    # choice for transformers. weight_decay gently pulls weights toward zero.
    # fused=True updates every weight in one GPU kernel instead of one per tensor.
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                             fused=True if args.fused else None)


def train_step(model, optimizer, x, y, args, device):
    """The four lines that are all of "training"."""
    with autocast(args, device):
        _, loss = model(x, y)              # 1. forward: make predictions, score them
    optimizer.zero_grad(set_to_none=True)  # 2. clear old gradients
    loss.backward()                        # 3. backprop: which way should each weight move?
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # (safety: cap huge updates)
    optimizer.step()                       # 4. move every weight a tiny bit
    return loss


def lr_at(it, args):
    """Warm up gently, then decay. Big steps early on can wreck a fresh model;
    small steps at the end let it settle into a good spot."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine down to 10%


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()

    data_dir = DATA_DIR / args.dataset
    if not (data_dir / "meta.json").exists():
        raise SystemExit(f"No data in {data_dir}. Run: uv run prepare.py --dataset {args.dataset}")
    chars = json.loads((data_dir / "meta.json").read_text())["chars"]
    splits = {
        "train": np.fromfile(data_dir / "train.bin", dtype=np.uint16),
        "val": np.fromfile(data_dir / "val.bin", dtype=np.uint16),
    }

    cfg = GPTConfig(vocab_size=len(chars), block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, sdpa=args.sdpa,
                    grad_checkpoint=args.grad_checkpoint)
    model = GPT(cfg).to(device)
    print(f"Device: {device} | Parameters: {model.num_params() / 1e6:.2f}M")
    optimizer = make_optimizer(model, args)
    # torch.compile traces the model once and generates fused kernels for it. We train
    # through the compiled version but save and sample from the plain one.
    fast_model = torch.compile(model) if args.compile else model
    if args.compile and device == "mps":
        print("Warning: torch.compile on Apple GPUs gave wrong gradients in testing. See README.")
    if args.grad_checkpoint and device == "mps" and args.dropout > 0:
        print("Warning: on Apple GPUs, checkpointing can't replay dropout exactly, so gradients"
              " are slightly off. Use --dropout 0 with it, or train on the halo.")

    # Record exactly what produced this run, so you can compare or repeat it later.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    best_val = float("inf")
    t0 = t_log = time.time()

    def save(path, step):
        # Everything needed to sample from this model, plus the optimizer state and
        # step number so a run could be picked back up from here.
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "chars": chars,
                    "optimizer": optimizer.state_dict(), "step": step}, path)

    for it in range(args.max_iters + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it, args)

        if it % args.eval_interval == 0:
            losses = estimate_loss(fast_model, splits, args, device)
            print(f"step {it:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
                  f" | {time.time() - t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(args.out_dir / "ckpt.pt", it)
            t_log = time.time()  # don't count evaluation time in the ms/step below

        if it == args.max_iters:
            break

        x, y = get_batch(splits["train"], args.batch_size, args.block_size, device)
        loss = train_step(fast_model, optimizer, x, y, args, device)
        step = it + 1  # number of weight updates done so far

        if step % args.log_interval == 0:
            # The loss on the one batch just trained on: noisy, but free. The eval lines
            # above average many batches, and include val, so they're the ones to trust.
            ms = (time.time() - t_log) / args.log_interval * 1000
            print(f"step {step:5d} | batch loss {loss.item():.3f} | lr {lr_at(it, args):.2e} | {ms:.0f} ms/step")
            t_log = time.time()

        if step % args.ckpt_interval == 0:
            save(args.out_dir / "last.pt", step)

    print(f"\nBest val loss {best_val:.3f}. Checkpoint saved to {args.out_dir / 'ckpt.pt'}")
    print("\n--- sample ---")
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    start = torch.tensor([[stoi["\n"]]], device=device)
    out = model.generate(start, max_new_tokens=400, temperature=0.8, top_k=40)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
