"""
Step 1: get some text and turn it into numbers.

A language model never sees letters or words, only integers ("tokens").
Here we use the simplest possible tokenizer: one token per character.
Real LLMs use subword tokenizers (BPE) with ~50k-200k tokens, but the
idea is the same: text -> list of ints -> model -> list of ints -> text.

Run:  uv run prepare.py                     # Tiny Shakespeare (~1 MB, downloaded)
      uv run prepare.py --dataset python    # Python's own source code (~8 MB, already on disk)

Each dataset gets its own folder under data/. Train on one with
`uv run train.py --dataset python` (or `dataset = "python"` in a config).
"""

import argparse
import json
import random
import ssl
import sysconfig
import urllib.request
from pathlib import Path

import certifi
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Parts of the standard library that aren't representative Python: test suites,
# GUI demos, and codec tables that are mostly giant literal strings.
PYTHON_SKIP = {"test", "tests", "idlelib", "turtledemo", "lib2to3", "encodings",
               "site-packages", "__pycache__"}


def load_shakespeare():
    print(f"Downloading Tiny Shakespeare from {SHAKESPEARE_URL}")
    # certifi supplies SSL certificates (the python.org macOS installer doesn't).
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(SHAKESPEARE_URL, context=ctx) as resp:
        return resp.read().decode("utf-8")


def load_python():
    """Python's standard library: hundreds of files of real, well-written code,
    installed along with Python itself, so there's nothing to download."""
    src = Path(sysconfig.get_paths()["stdlib"])
    print(f"Reading Python source from {src}")
    files = sorted(p for p in src.rglob("*.py")
                   if not PYTHON_SKIP & set(p.relative_to(src).parts))
    # Shuffle (with a fixed seed) so the val split is a random mix of modules,
    # not just whatever sorts last alphabetically.
    random.Random(0).shuffle(files)

    parts = []
    for path in files:
        code = path.read_text(encoding="utf-8", errors="replace")
        # Skip the few files with non-ASCII characters (accented names in comments,
        # etc.) so the vocabulary stays small.
        if code.isascii():
            parts.append(f"# file: {path.relative_to(src)}\n{code}\n")
    print(f"Using {len(parts)} of {len(files)} files")
    return "".join(parts)


DATASETS = {"shakespeare": load_shakespeare, "python": load_python}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="shakespeare")
    args = p.parse_args()

    out_dir = DATA_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "input.txt"

    # Want to train on something else? Put any text you like in data/<name>/input.txt
    # (your own writing, song lyrics, recipes...) and add a loader above.
    if not raw_path.exists():
        raw_path.write_text(DATASETS[args.dataset](), encoding="utf-8")
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
    ids[:split].tofile(out_dir / "train.bin")
    ids[split:].tofile(out_dir / "val.bin")

    (out_dir / "meta.json").write_text(json.dumps({"chars": chars}))
    print(f"Wrote {split:,} train tokens and {len(ids) - split:,} val tokens to {out_dir}")

    sample = text[:40]
    print(f"\nExample:\n  text: {sample!r}\n  ids:  {[stoi[c] for c in sample]}")


if __name__ == "__main__":
    main()
