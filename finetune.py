"""
Step 5: instruction tuning. Turn a model that continues code into one that answers requests.

Pretraining teaches a model what code looks like: given the start of a file, it continues
it. Ask it "Write a function that reverses a string" and it continues *that* text, maybe
with more questions. Instruction tuning is more training, on examples written as a request
followed by a good answer:

  ### Instruction:
  Write a function that reverses a string.

  ### Response:
  ```python
  def reverse(s):
      return s[::-1]
  ```<end of text>

Two details make it work:
  - The loss only counts the answer. The model isn't asked to predict the instructions,
    only to answer them, so the prompt's targets are set to -100 (which the loss skips).
  - Every answer ends with the end-of-text token, so the model learns to stop.

The data is Magicoder's OSS-Instruct and Evol-Instruct sets, Python examples only
(download.sh fetches them). They were written by GPT-3.5/4 from real code snippets.

Run:  uv run finetune.py --ckpt out/big --out_dir out/big-sft
      uv run sample.py --ckpt out/big-sft --temperature 0 --prompt "$(uv run finetune.py --show 'Write a function that checks if a number is prime.')"
"""

import argparse
import json
import random
import time
import unicodedata
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from prepare import ASCII_FIXES
from tokenizer import EOT, Tokenizer
from train import autocast, make_optimizer, pick_device, train_step

PROJECT_DIR = Path(__file__).parent
RAW = PROJECT_DIR / "data" / "raw" / "instruct"
PROMPT = "### Instruction:\n{}\n\n### Response:\n"


def prompt(instruction):
    return PROMPT.format(instruction.strip())


def to_ascii(text):
    # Same treatment as the Python docs in prepare.py: curly quotes become straight ones,
    # accents are dropped. Examples with anything else left over are skipped.
    text = text.translate(ASCII_FIXES).replace("\r\n", "\n").expandtabs(8)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def load_examples():
    """(instruction, response) pairs, Python only."""
    pairs = []
    for line in open(RAW / "oss-instruct.jsonl"):
        r = json.loads(line)
        if r["lang"] == "python":
            pairs.append((r["problem"], r["solution"]))
    for line in open(RAW / "evol-instruct.jsonl"):
        r = json.loads(line)
        if "```python" in r["response"]:
            pairs.append((r["instruction"], r["response"]))
    return [(to_ascii(i), to_ascii(r).strip()) for i, r in pairs]


def encode(tok, examples, max_len):
    """Each example becomes (tokens, targets), with the prompt's targets set to -100."""
    out = []
    for instruction, response in examples:
        try:
            p = tok.encode(prompt(instruction))
            a = tok.encode(response + EOT)
        except ValueError:
            continue  # a character the tokenizer doesn't know
        if len(p) + len(a) > max_len:
            continue  # too long to fit in the context window
        out.append((p + a, [-100] * len(p) + a))
    return out


def pack(examples, block_size, rng):
    """Shuffle and lay examples end to end, then cut into block_size windows, so no compute
    is wasted on padding. An example can be split across two windows; that's fine."""
    rng.shuffle(examples)
    ids = np.fromiter((t for e in examples for t in e[0]), dtype=np.int64)
    tgt = np.fromiter((t for e in examples for t in e[1]), dtype=np.int64)
    n = (len(ids) - 1) // block_size
    x = torch.from_numpy(ids[:n * block_size].reshape(n, block_size))
    # The target for each position is the *next* token: shift by one.
    y = torch.from_numpy(tgt[1:n * block_size + 1].reshape(n, block_size))
    return x, y


@torch.no_grad()
def val_loss(model, x, y, args, device):
    model.eval()
    losses = []
    for i in range(0, len(x), args.batch_size):
        with autocast(args, device):
            _, loss = model(x[i:i + args.batch_size].to(device), y[i:i + args.batch_size].to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, help="the pretrained model (a run's out_dir or a .pt file)")
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sdpa", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--show", metavar="INSTRUCTION", help="just print the prompt for an instruction")
    args = p.parse_args()
    if args.show:
        print(prompt(args.show), end="")
        return

    device = pick_device()
    ckpt_path = PROJECT_DIR / args.ckpt
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "ckpt.pt"
    ck = torch.load(ckpt_path, map_location=device)
    tok = Tokenizer(ck["chars"], ck["merges"])
    if EOT not in tok.char_ids:
        raise SystemExit("This model's tokenizer has no end-of-text token: use one made by prepare_big.py")
    cfg = GPTConfig(**{**ck["config"], "dropout": 0.0, "sdpa": args.sdpa, "grad_checkpoint": False})
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"Loaded {ckpt_path} (step {ck['step']}, {model.num_params() / 1e6:.0f}M parameters)")

    t0 = time.time()
    examples = load_examples()
    random.Random(0).shuffle(examples)
    data = encode(tok, examples, cfg.block_size + 1)
    n_val = len(data) // 50  # 2% held out
    val, train = data[:n_val], data[n_val:]
    rng = random.Random(1)
    vx, vy = pack(val, cfg.block_size, rng)
    answer_tokens = sum(sum(t != -100 for t in e[1]) for e in train)
    total_tokens = sum(len(e[0]) for e in train)
    print(f"{len(examples):,} Python examples, {len(data):,} fit in {cfg.block_size} tokens "
          f"({len(train):,} train, {len(val):,} val) | {total_tokens / 1e6:.0f}M tokens per epoch, "
          f"{answer_tokens / total_tokens:.0%} of them answers | {time.time() - t0:.0f}s")

    batches_per_epoch = (total_tokens // cfg.block_size) // args.batch_size
    max_iters = args.epochs * batches_per_epoch
    optimizer = make_optimizer(model, args)
    out_dir = PROJECT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best = val_loss(model, vx, vy, args, device)
    print(f"step     0 | val loss {best:.3f} (answers only, before tuning)")

    def evaluate(it, best):
        v = val_loss(model, vx, vy, args, device)
        print(f"step {it:5d} | val loss {v:.3f}", flush=True)
        if v < best:
            torch.save({"model": model.state_dict(), "config": {**asdict(cfg), "sdpa": False},
                        "chars": tok.chars, "merges": tok.merges, "step": ck["step"],
                        "sft_step": it, "best_val": v, "prompt": PROMPT}, out_dir / "ckpt.pt")
            return v
        return best

    it, t0 = 0, time.time()
    for epoch in range(args.epochs):
        x, y = pack(train, cfg.block_size, rng)
        for i in range(0, len(x) - args.batch_size + 1, args.batch_size):
            # Warm up, then decay linearly to zero: the usual schedule for fine-tuning.
            lr = args.lr * max(0.0, min((it + 1) / args.warmup_iters, 1 - it / max_iters))
            for g in optimizer.param_groups:
                g["lr"] = lr
            loss = train_step(model, optimizer, x[i:i + args.batch_size].to(device),
                              y[i:i + args.batch_size].to(device), args, device)
            it += 1
            if it % 50 == 0:
                print(f"step {it:5d}/{max_iters} | epoch {epoch + 1} | batch loss {loss.item():.3f} "
                      f"| lr {lr:.1e} | {(time.time() - t0) / it * 1000:.0f} ms/step", flush=True)
            if it % args.eval_interval == 0:
                best = evaluate(it, best)
    best = evaluate(it, best)
    print(f"\nBest val loss {best:.3f}. Saved to {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
