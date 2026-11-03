"""
Step 3: training. Show the model real text, measure how wrong its
next-token guesses are (the loss), and nudge every weight slightly in
the direction that would have made it less wrong. Repeat thousands of times.

Run:  uv run train.py
      uv run train.py --max_iters 500       # quick test
      uv run train.py --n_layer 6 --n_embd 256   # bigger model
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "out"


def get_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


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
        losses = [model(*get_batch(data, args.batch_size, args.block_size, device))[1].item()
                  for _ in range(args.eval_iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def lr_at(it, args):
    """Warm up gently, then decay. Big steps early on can wreck a fresh model;
    small steps at the end let it settle into a good spot."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))  # cosine down to 10%


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()

    meta = json.loads((DATA_DIR / "meta.json").read_text())
    chars = meta["chars"]
    splits = {
        "train": np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16),
        "val": np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16),
    }

    cfg = GPTConfig(vocab_size=len(chars), block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"Device: {device} | Parameters: {model.num_params() / 1e6:.2f}M")

    # AdamW: gradient descent with momentum and per-weight step sizes. The default
    # choice for transformers. weight_decay gently pulls weights toward zero.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    OUT_DIR.mkdir(exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for it in range(args.max_iters + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(it, args)

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, splits, args, device)
            print(f"step {it:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
                  f" | {time.time() - t0:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "chars": chars},
                           OUT_DIR / "ckpt.pt")

        if it == args.max_iters:
            break

        # The four lines that are all of "training":
        x, y = get_batch(splits["train"], args.batch_size, args.block_size, device)
        _, loss = model(x, y)                  # 1. forward: make predictions, score them
        optimizer.zero_grad(set_to_none=True)  # 2. clear old gradients
        loss.backward()                        # 3. backprop: which way should each weight move?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # (safety: cap huge updates)
        optimizer.step()                       # 4. move every weight a tiny bit

    print(f"\nBest val loss {best_val:.3f}. Checkpoint saved to {OUT_DIR / 'ckpt.pt'}")
    print("\n--- sample ---")
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    start = torch.tensor([[stoi["\n"]]], device=device)
    out = model.generate(start, max_new_tokens=400, temperature=0.8, top_k=40)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
