"""
Step 1: get some text and turn it into numbers.

A language model never sees letters or words, only integers ("tokens").
Here we use the simplest possible tokenizer: one token per character.
Real LLMs use subword tokenizers (BPE) with ~50k-200k tokens, but the
idea is the same: text -> list of ints -> model -> list of ints -> text.

Run:  uv run prepare.py
"""

import json
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    raw_path = DATA_DIR / "input.txt"

    # Swap in any .txt file you like here: your own writing, song lyrics,
    # code, recipes... The model will learn to imitate whatever it sees.
    if not raw_path.exists():
        print(f"Downloading Tiny Shakespeare (~1 MB) to {raw_path}")
        # certifi supplies SSL certificates (the python.org macOS installer doesn't).
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATA_URL, context=ctx) as resp:
            raw_path.write_bytes(resp.read())

    text = raw_path.read_text(encoding="utf-8")
    print(f"Dataset has {len(text):,} characters")

    # Build the vocabulary: every unique character gets an id.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
    print(f"Vocabulary size: {len(chars)}")
    print(f"Vocabulary: {''.join(chars)!r}")

    # Encode the whole dataset to ints.
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)

    # Hold out the last 10% so we can check the model on text it never trained on.
    # If train loss keeps dropping but val loss rises, the model is memorizing
    # (overfitting) instead of learning general patterns.
    split = int(0.9 * len(ids))
    ids[:split].tofile(DATA_DIR / "train.bin")
    ids[split:].tofile(DATA_DIR / "val.bin")

    (DATA_DIR / "meta.json").write_text(json.dumps({"chars": chars}))
    print(f"Wrote {split:,} train tokens and {len(ids) - split:,} val tokens")

    sample = text[:40]
    print(f"\nExample:\n  text: {sample!r}\n  ids:  {[stoi[c] for c in sample]}")


if __name__ == "__main__":
    main()
