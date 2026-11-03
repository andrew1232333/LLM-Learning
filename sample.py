"""
Step 4: talk to your model.

Run:  uv run sample.py
      uv run sample.py --prompt "ROMEO:" --temperature 0.5
      uv run sample.py --temperature 1.5    # wilder, more random
      uv run sample.py --ckpt out/medium    # a different run
"""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig
from train import pick_device

PROJECT_DIR = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("out"),
                   help="checkpoint file, or a run's out_dir containing ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="<1 = safer/more repetitive, >1 = more creative/chaotic")
    p.add_argument("--top_k", type=int, default=40,
                   help="only sample from the k most likely next characters")
    args = p.parse_args()

    device = pick_device()
    ckpt_path = PROJECT_DIR / args.ckpt
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    chars = ckpt["chars"]
    stoi = {c: i for i, c in enumerate(chars)}

    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    unknown = set(args.prompt) - set(stoi)
    if unknown:
        raise SystemExit(f"Prompt has characters the model never saw: {unknown}")

    idx = torch.tensor([[stoi[c] for c in args.prompt]], device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
