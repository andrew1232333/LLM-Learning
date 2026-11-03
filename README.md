# tiny-llm

A small GPT-style language model built from scratch in PyTorch, sized to train on a
laptop in a few minutes. Every line is meant to be read; the comments explain *why*.

## Run it

```bash
uv run prepare.py   # download text, turn characters into numbers
uv run train.py     # ~2-3 min on an M5 MacBook, ~0.8M parameters
uv run sample.py --prompt "ROMEO:"
```

## Datasets

| `--dataset` | What | Size |
|---|---|---|
| `shakespeare` (default) | Tiny Shakespeare, downloaded | 1.1M characters |
| `python` | Python's standard library source, read from your own Python install | 7.8M characters |

```bash
uv run prepare.py --dataset python
uv run train.py --config configs/code.toml
uv run sample.py --ckpt out/code --prompt "def "
```

Each dataset lives in `data/<name>/`. `input.txt` there is the exact text the model trains on,
so open it to see what the model is learning from. To add your own, write a loader in
`prepare.py`: any function that returns one big string works.

## Configuring a run

Every setting is a flag on `train.py` (`uv run train.py --help` lists them). To keep a setup
you like, put it in a TOML file under `configs/` and pass `--config`. Flags typed on the
command line override the file, so you can try one change without editing it.

```bash
uv run train.py --config configs/small.toml
uv run train.py --config configs/medium.toml --dropout 0.3 --out_dir out/medium-drop3
uv run sample.py --ckpt out/medium-drop3
```

| Preset | Params | Time |
|---|---|---|
| `configs/small.toml` | 0.8M | ~2-3 min on an M5 Mac or the halo |
| `configs/medium.toml` | 10.8M | ~40 min on an M5 Mac, ~15 min on the halo |
| `configs/code.toml` | 10.8M | same as medium, trains on Python code |

Each run writes to its own `out_dir`:

- `ckpt.pt`: the model with the best val loss so far (checked every `eval_interval` steps)
- `last.pt`: the latest model, saved every `ckpt_interval` steps (default 100)
- `settings.json`: every setting that produced the run

Both checkpoints work with `sample.py --ckpt`, so you can sample a run while it's still training.
Every `log_interval` steps (default 10) training prints the loss on the batch it just trained
on, plus the learning rate and speed. That number is noisy; the `train loss | val loss` lines,
which average many batches, are the ones to compare. Give experiments different `out_dir`s so they don't overwrite
each other. To make your own preset, copy one of the files; any setting you leave out uses
the default from `train.py`, and a misspelled setting is an error rather than silently ignored.

## Speed

`uv run bench.py` times a burst of training steps with each speedup switched on one at a
time. Results for the medium preset:

| Setting | M5 Mac | Halo (Radeon 8060S) |
|---|---|---|
| Baseline (float32, manual attention) | 762 ms/step | 592 ms/step |
| `--dtype bfloat16` (mixed precision) | 1.6x faster | **3.2x faster** |
| `--fused` AdamW | no change | no change |
| `--sdpa` (fused attention) | no change | crashes in the backward pass |
| `--compile` | 2.6x faster, but **wrong gradients** | unavailable (no Triton) |

The defaults are bfloat16 on, sdpa and compile off. bfloat16 reached the same loss as float32
in side-by-side runs. `torch.compile` on the Mac computed gradients that were nowhere near
the real ones: loss matched to 6 decimals, gradients differed by ~10x. Training still crept
along (AdamW mostly cares about direction) but learned measurably worse. The lesson: a
speedup is only a speedup if the numbers are still right, so check before trusting one.

## Read the code in this order

| File | What it teaches |
|---|---|
| `prepare.py` | Tokenization: text becomes integers. Train/val split. |
| `model.py` | The transformer: embeddings, attention, MLP, residuals, layer norm. |
| `train.py` | The training loop: batches, loss, backprop, optimizer, learning-rate schedule. |
| `sample.py` | Generation: temperature, top-k, feeding outputs back in. |
| `bench.py` | Measuring speed, and checking a speedup didn't change the math. |

## What the numbers mean

- **Loss ~4.17 at step 0.** That's `ln(65)`: random guessing over 65 characters.
- **Loss ~1.5 after training.** On average the model gives the right next character
  about `e^-1.5 ≈ 22%` probability. It has learned spelling, line breaks, speaker names
  and rough grammar, but not meaning.
- **Train loss below val loss.** The gap is overfitting. If val loss starts *rising*
  while train loss keeps falling, the model is memorizing.

## Experiments to try (roughly in order)

1. **Watch it learn.** Run `train.py --max_iters 200`, then `500`, then `3000`, and sample each time.
   Gibberish → word-shaped blobs → real words → Shakespeare-ish.
2. **Temperature.** `sample.py --temperature 0.3` vs `1.5`. Why is low temperature repetitive?
3. **Break it on purpose.** Try `--lr 1e-2` (unstable), `--lr 1e-5` (barely learns),
   `--n_layer 1`, `--block_size 8` (can't see enough context).
4. **Scale up.** `--config configs/medium.toml` (~10M params).
   Does val loss improve? Does the gap with train loss grow? Try `--dropout 0.1` vs `0.3`.
5. **Different data.** Train on `--dataset python` and compare it with Shakespeare: which reaches
   a lower loss, and why? (Code is more predictable: indentation, `self.`, closing brackets.)
   Then add a loader for your own text in `prepare.py`.
6. **Remove a piece.** Delete the residual connections (`x + ...`) in `Block`, or the causal
   mask in attention. Predict what happens first, then check.

## Where to go next

- **Subword tokenization (BPE).** Write a byte-pair encoder so tokens are word pieces, not
  characters. This is how every real LLM tokenizes.
- **Fine-tuning a pretrained model.** Take an existing small open model (e.g. from Hugging Face)
  and fine-tune it on your own data, with LoRA to keep memory low. This is where most
  real-world "training an LLM" actually happens.
- **Instruction tuning.** Train on prompt → response pairs so the model follows instructions
  instead of just continuing text. This is the step that turns a base model into a chat assistant.
