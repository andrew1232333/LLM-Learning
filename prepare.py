"""
Step 1: get some text and turn it into numbers.

A language model never sees letters or words, only integers ("tokens").
By default we use the simplest possible tokenizer: one token per character.
Real LLMs use subword tokenizers (BPE, see tokenizer.py) with ~50k-200k tokens,
but the idea is the same: text -> list of ints -> model -> list of ints -> text.

Run:  uv run prepare.py                          # Tiny Shakespeare (~1 MB, downloaded)
      uv run prepare.py --dataset python         # Python's own source code (~8 MB, already on disk)
      uv run prepare.py --dataset python+docs    # that code plus the official docs (~22 MB)
      uv run prepare.py --dataset python+docs+packages   # plus every installed package's code
      uv run prepare.py --dataset python+docs --vocab_size 2048   # BPE instead of characters

Each dataset gets its own folder under data/ (BPE ones get a suffix, e.g.
data/python+docs-bpe2048). Train on one with `uv run train.py --dataset python`
(or `dataset = "python"` in a config).
"""

import argparse
import io
import json
import random
import ssl
import string
import sys
import sysconfig
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

import certifi
import numpy as np

from tokenizer import Tokenizer

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


def load_packages():
    """Every third-party package installed alongside Python (site-packages and, on
    Debian/Ubuntu, dist-packages). How much this is depends on the machine: a fresh
    venv has a few MB, a system with torch, scipy, sympy and friends has 100+ MB."""
    dirs = sorted({Path(p) for p in sys.path
                   if p.endswith(("site-packages", "dist-packages")) and Path(p).is_dir()})
    allowed = set(string.printable) - set("\t\r\x0b\x0c") | {"\n"}  # same characters as python+docs
    parts, seen = [], set()
    for root in dirs:
        print(f"Reading packages from {root}")
        for path in sorted(root.rglob("*.py")):
            if {"test", "tests", "testing", "__pycache__"} & set(path.relative_to(root).parts):
                continue
            try:
                code = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            code = code.replace("\r\n", "\n").expandtabs(8)  # tabs indent to 8, as Python reads them
            if (code in seen or not set(code) <= allowed
                    or max(map(len, code.splitlines()), default=0) > 1000):
                continue  # duplicates (vendored copies), odd characters, generated data tables
            seen.add(code)
            parts.append(f"# file: {path.relative_to(root)}\n{code}\n")
    print(f"Using {len(parts)} package files")
    return parts


def load_python_docs_packages():
    # Put the packages just before python+docs' val part, so the 90/10 split below keeps
    # exactly that val text in val. A model trained here never sees it, so its loss on it
    # can be compared with models trained on python+docs alone.
    base_path = DATA_DIR / "python+docs" / "input.txt"
    if not base_path.exists():
        raise SystemExit("Run `uv run prepare.py --dataset python+docs` first")
    base = base_path.read_text(encoding="utf-8")
    packages = load_packages()
    random.Random(0).shuffle(packages)
    cut = int(0.9 * len(base))
    return [base[:cut] + "".join(packages) + base[cut:]]


# Each loader returns a list of documents (files, pages, or one big text).
DATASETS = {
    "shakespeare": load_shakespeare,
    "python": load_python,
    "python+docs": lambda: load_python() + load_python_docs(),
    "python+docs+packages": load_python_docs_packages,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default="shakespeare")
    p.add_argument("--vocab_size", type=int, default=0,
                   help="train a BPE tokenizer with this many tokens (default: one token per character)")
    args = p.parse_args()

    raw_dir = DATA_DIR / args.dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "input.txt"

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

    if args.vocab_size:
        print(f"Training a BPE tokenizer with {args.vocab_size} tokens")
        tok = Tokenizer.train(text, args.vocab_size)
        out_dir = DATA_DIR / f"{args.dataset}-bpe{args.vocab_size}"
    else:
        # Every unique character gets an id, and that's the whole tokenizer.
        tok = Tokenizer(sorted(set(text)))
        out_dir = raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Vocabulary size: {tok.vocab_size}")
    print(f"Characters: {''.join(tok.chars)!r}")

    # Encode the whole dataset to ints.
    ids = np.array(tok.encode(text), dtype=np.uint16)
    print(f"{len(text) / len(ids):.2f} characters per token")

    # Hold out the last 10% so we can check the model on text it never trained on.
    # If train loss keeps dropping but val loss rises, the model is memorizing
    # (overfitting) instead of learning general patterns.
    split = int(0.9 * len(ids))
    ids[:split].tofile(out_dir / "train.bin")
    ids[split:].tofile(out_dir / "val.bin")

    # chars_per_token lets train.py report loss per character, which can be compared
    # across tokenizers (loss per token can't: a bigger token is harder to guess).
    val_chars = len(tok.decode(ids[split:].tolist()))
    meta = {"chars": tok.chars, "merges": tok.merges, "chars_per_token": val_chars / (len(ids) - split)}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"Wrote {split:,} train tokens and {len(ids) - split:,} val tokens to {out_dir}")

    sample = text[:60]
    print(f"\nExample:\n  text:   {sample!r}\n  tokens: {[tok.decode([i]) for i in tok.encode(sample)]}")


if __name__ == "__main__":
    main()
