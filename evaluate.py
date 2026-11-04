"""
Step 6: does the model write code that works? Run its code against real tests.

Loss says how well a model predicts code; it doesn't say whether the code runs. These
two standard benchmarks do:

  HumanEval  164 problems. The model gets a function's signature and docstring and
             writes the body. Hidden unit tests check it.
  MBPP       500 problems (the test split). The model gets a one-line description, like
             "Write a function to find the shared elements from the given two lists",
             plus one example assert, and writes the whole function. Three asserts check it.

The score is pass@1: the share of problems solved on the first try, always taking the
likeliest token (temperature 0). For scale: GPT-2-sized models trained on general text
score ~0%; CodeGen-350M-mono, trained on 577B tokens, scores 12.8% on HumanEval.

A base model just continues text, so it gets the problem as code to continue. An
instruction-tuned model (finetune.py) gets it as a request in its prompt format.

WARNING: this runs the code the model writes. Each program runs in its own Python process
with a time limit, in a temporary folder, but it is not a sandbox.

Run:  uv run evaluate.py --ckpt out/big
      uv run evaluate.py --ckpt out/big-sft --limit 50    # a quick look
"""

import argparse
import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import EOT, Tokenizer
from train import pick_device

PROJECT_DIR = Path(__file__).parent
RAW = PROJECT_DIR / "data" / "raw" / "eval"


def humaneval():
    for r in map(json.loads, gzip.open(RAW / "HumanEval.jsonl.gz", "rt")):
        yield {"id": r["task_id"], "prompt": r["prompt"], "head": r["prompt"],
               "request": f"Complete this Python function:\n```python\n{r['prompt']}```",
               "test": f"{r['test']}\ncheck({r['entry_point']})\n",
               "stop": re.compile(r"\n(?=\S)")}  # the body is done at the first unindented line


def mbpp():
    # The standard test split is task ids 11-510.
    for r in map(json.loads, open(RAW / "mbpp.jsonl")):
        if 11 <= r["task_id"] <= 510:
            task = f"{r['text']}\nYour code should pass this test:\n{r['test_list'][0]}"
            yield {"id": f"MBPP/{r['task_id']}", "prompt": f'"""\n{task}\n"""\n', "head": "",
                   "request": f"Write a Python function for this task: {task}",
                   "test": r["test_setup_code"] + "\n" + "\n".join(r["test_list"]) + "\n",
                   # Whole functions (and helpers, imports) are fine; tests or prose mean it's done.
                   "stop": re.compile(r"\n(?=assert |print\(|if __name__|\"\"\"|#)")}


def complete(model, tok, text, device, max_tokens, stop_at):
    """Greedy completion of text, generated in chunks until stop_at(completion) finds an end."""
    ids = tok.encode(text)
    n = len(ids)
    out = ""
    while len(ids) - n < max_tokens:
        idx = torch.tensor([ids[-model.cfg.block_size:]], device=device)
        ids = ids[:-idx.size(1)] + model.generate(idx, 64, temperature=0, stop=tok.char_ids[EOT]
                                                  if EOT in tok.char_ids else None)[0].tolist()
        out = tok.decode(ids[n:])
        end = stop_at(out)
        if end is not None:
            return out[:end]
    return out


def first(out, *ends):
    """Where the earliest of these ends is in out (None if none are)."""
    cut = [i for i in ends if i is not None and i >= 0]
    return min(cut) if cut else None


def base_stop(regex):
    # A base model doesn't know when to stop, so cut its output where the code it was
    # asked for is clearly finished.
    return lambda out: first(out, out.find(EOT), (m := regex.search(out)) and m.start())


def chat_stop(out):
    # Instruction-tuned: the code ends with the closing ``` of its code block, or EOT.
    return first(out, out.find("```"), out.find(EOT))


def run(program, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, timeout=timeout,
                               capture_output=True, text=True)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--bench", choices=["humaneval", "mbpp", "both"], default="both")
    p.add_argument("--limit", type=int, default=0, help="only the first N problems of each")
    p.add_argument("--max_tokens", type=int, default=384)
    p.add_argument("--show", type=int, default=3, help="print this many solutions of each")
    args = p.parse_args()

    device = pick_device()
    path = PROJECT_DIR / args.ckpt
    if path.is_dir():
        path = path / "ckpt.pt"
    ck = torch.load(path, map_location=device)
    tok = Tokenizer(ck["chars"], ck.get("merges", []))
    model = GPT(GPTConfig(**{**ck["config"], "dropout": 0.0, "sdpa": False})).to(device).eval()
    model.load_state_dict(ck["model"])
    chat = "prompt" in ck  # finetune.py saves its prompt format
    print(f"{path} | step {ck['step']} | {'instruction-tuned' if chat else 'base model'}")

    benches = {"humaneval": humaneval, "mbpp": mbpp}
    results = {}
    for name in (benches if args.bench == "both" else [args.bench]):
        problems = list(benches[name]())[:args.limit or None]
        t0 = time.time()
        programs = []
        for i, pr in enumerate(problems):
            if chat:
                # Start the answer for it: the model continues from inside a code block.
                text = ck["prompt"].format(pr["request"]) + "```python\n" + pr["head"]
                body = complete(model, tok, text, device, args.max_tokens, chat_stop)
            else:
                body = complete(model, tok, pr["prompt"], device, args.max_tokens, base_stop(pr["stop"]))
            code = pr["head"] + body
            programs.append(code)
            if i < args.show:
                print(f"\n--- {pr['id']}\n{(pr['prompt'] if not chat else pr['request']).strip()}\n"
                      f"--- model:\n{code.rstrip()}")
        # Run every program against its tests, several at a time.
        with ThreadPoolExecutor(8) as ex:
            passed = list(ex.map(run, [c + "\n\n" + pr["test"] for c, pr in zip(programs, problems)]))
        results[name] = (sum(passed), len(problems))
        print(f"\n{name}: {sum(passed)}/{len(problems)} passed = {sum(passed) / len(problems):.1%} pass@1"
              f" | {time.time() - t0:.0f}s")
        solved = [pr["id"] for pr, ok in zip(problems, passed) if ok]
        if solved:
            print("solved:", ", ".join(solved))
    print("\n" + " | ".join(f"{k}: {a}/{b} ({a / b:.1%})" for k, (a, b) in results.items()))


if __name__ == "__main__":
    main()
