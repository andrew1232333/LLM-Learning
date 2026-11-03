"""
Step 1: get some text and turn it into numbers.

A language model never sees letters or words, only integers ("tokens").
Here we use the simplest possible tokenizer: one token per character.
Real LLMs use subword tokenizers (BPE) with ~50k-200k tokens, but the
idea is the same: text -> list of ints -> model -> list of ints -> text.

Run:  uv run prepare.py                          # Tiny Shakespeare (~1 MB, downloaded)
      uv run prepare.py --dataset python         # Python's own source code (~8 MB, already on disk)
      uv run prepare.py --dataset python+docs    # that code plus the official docs (~22 MB)

Each dataset gets its own folder under data/. Train on one with
`uv run train.py --dataset python` (or `dataset = "python"` in a config).
"""

import argparse
import io
import json
import random
import ssl
import sys
import sysconfig
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

import certifi
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
DOCS_URL = f"https://docs.python.org/{PY_VERSION}/archives/python-{PY_VERSION}-docs-text.tar.bz2"

# Parts of the standard library that aren't representative Python: test suites,
# GUI demos, and codec tables that are mostly giant literal strings.
PYTHON_SKIP = {"test", "tests", "idlelib", "turtledemo", "lib2to3", "encodings",
               "site-packages", "__pycache__"}


# A handful of typographic characters show up in the docs. Spell them in ASCII so the
# vocabulary stays small; accented letters lose their accents (e -> e) further down.
ASCII_FIXES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                             "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u2192": "->",
                             "\u2500": "-", "\u2502": "|", "\u251c": "|", "\u2514": "`"})


def download(url):
    print(f"Downloading {url}")
    # certifi supplies SSL certificates (the python.org macOS installer doesn't).
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp:
        return resp.read()


def load_shakespeare():
    return [download(SHAKESPEARE_URL).decode("utf-8")]


def load_python():
    """Python's standard library: hundreds of files of real, well-written code,
    installed along with Python itself, so there's nothing to download."""
    src = Path(sysconfig.get_paths()["stdlib"])
    print(f"Reading Python source from {src}")
    files = sorted(p for p in src.rglob("*.py")
                   if not PYTHON_SKIP & set(p.relative_to(src).parts))
    parts = []
    for path in files:
        code = path.read_text(encoding="utf-8", errors="replace")
        # Skip the few files with non-ASCII characters (accented names in comments,
        # etc.) so the vocabulary stays small.
        if code.isascii():
            parts.append(f"# file: {path.relative_to(src)}\n{code}\n")
    print(f"Using {len(parts)} of {len(files)} source files")
    return parts


def load_python_docs():
    """The official Python documentation as plain text: the tutorial, the library
    reference for every module, the language reference, HOWTOs, FAQs."""
    archive = tarfile.open(fileobj=io.BytesIO(download(DOCS_URL)))
    parts = []
    for member in sorted(archive.getmembers(), key=lambda m: m.name):
        if not (member.isfile() and member.name.endswith(".txt")):
            continue
        text = archive.extractfile(member).read().decode("utf-8").translate(ASCII_FIXES)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        name = member.name.split("/", 1)[1]  # drop the archive's top-level folder
        parts.append(f"# docs: {name}\n{text}\n")
    print(f"Using {len(parts)} documentation pages")
    return parts


# Each loader returns a list of documents (files, pages, or one big text).
DATASETS = {
    "shakespeare": load_shakespeare,
    "python": load_python,
    "python+docs": lambda: load_python() + load_python_docs(),
}


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
        documents = DATASETS[args.dataset]()
        # Shuffle (with a fixed seed) so the val split is a random mix of documents,
        # not just whatever sorts last alphabetically.
        random.Random(0).shuffle(documents)
        raw_path.write_text("".join(documents), encoding="utf-8")
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
