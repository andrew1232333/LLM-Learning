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
| `python+docs` | That source plus the official Python docs (tutorial, library reference, HOWTOs), downloaded from python.org | 21.6M characters |

```bash
uv run prepare.py --dataset python
uv run train.py --config configs/code.toml
uv run sample.py --ckpt out/code --prompt "def "
```

Each dataset lives in `data/<name>/`. `input.txt` there is the exact text the model trains on,
so open it to see what the model is learning from. To add your own, write a loader in
`prepare.py`: any function that returns one big string works.

## Tokenizers: characters or BPE

By default every character is one token. `--vocab_size` trains a byte-pair encoding (BPE)
tokenizer instead (`tokenizer.py`), the kind every real LLM uses:

```bash
uv run prepare.py --dataset python+docs --vocab_size 2048   # writes data/python+docs-bpe2048
uv run train.py --config configs/code-docs.toml --dataset python+docs-bpe2048 --out_dir out/bpe
uv run tokenizer.py --dataset python+docs                    # just show how code gets split
```

BPE starts from characters and keeps merging the most common neighbouring pair into a new
token. Before counting pairs it splits the text along Python's structure, so merges never
cross from a name into punctuation, and a newline plus the next line's indentation stays
together. (Leading indentation is 20% of the standard library's characters.) A line of code
comes out like this:

```
def | ( | self | , |  name | ): | \n         | return |  f | " | H | el | lo
```

At 2048 tokens the text shrinks to ~3 characters per token, so the same `block_size` sees 3x
more code. Things to know:

- **Loss per token isn't comparable with characters.** A token is ~3 characters, so it's
  harder to guess. With BPE the eval lines also print val loss per character
  (`val loss 4.687 (1.512/char)`); compare that number against character-level runs.
- **Step 0 loss is `ln(vocab_size)`**: ~7.6 for 2048 tokens.
- **Keep the vocabulary small for small data.** Real models use 50k-200k tokens, but each
  extra token is seen less often, so with ~20M characters 2k-4k is plenty.
- **Don't end a prompt with a space.** In training the space is part of the next word
  (`" name"`), so a prompt ending in `"def "` leaves the model in an odd spot. Use `"def"`.

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
| `configs/code-docs.toml` | 10.8M | ~30 min on the halo: code + docs, 10,000 steps |

Each run writes to its own `out_dir`:

- `ckpt.pt`: the model with the best val loss so far (checked every `eval_interval` steps)
- `last.pt`: the latest model, saved every `ckpt_interval` steps (default 100)
- `settings.json`: every setting that produced the run

Both checkpoints work with `sample.py --ckpt`, so you can sample a run while it's still training.

To continue a run you stopped, pass `--resume` with its `out_dir`. It reloads `last.pt` (weights,
optimizer state and step) and the run's `settings.json`, so nothing else needs retyping:

```bash
uv run train.py --resume --out_dir out/medium                   # finish where it stopped
uv run train.py --resume --out_dir out/medium --max_iters 8000  # or train for longer
```

It restarts from the last `ckpt_interval` save, so up to 99 steps are redone. Raising `max_iters`
stretches the learning-rate schedule, so the rate jumps back up a little when the run continues.
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

## Long context (large `block_size`)

Attention compares every character in the window with every other one, so each head in each
layer builds a `block_size × block_size` table. Double the context and those tables get 4x
bigger. At long context they're nearly all of the memory, which means:

- **`n_head` is the biggest lever.** Each head has its own table. Fewer heads means less
  memory and more speed, with the same parameter count (each head just gets wider).
- **`batch_size`** scales memory linearly. Keep `batch_size × block_size` around 16k-130k characters per step.
- **`--grad_checkpoint`** keeps only each block's input and recomputes the rest during
  backprop: ~3.5x less memory for ~28% less speed. Gradients are identical on the halo. On
  Apple GPUs it can't replay dropout exactly, so use `--dropout 0` with it there.
- `n_embd` barely matters for memory at long context; it mostly sets the parameter count.

Measured on the halo (6 layers unless noted), memory per sequence in the batch:

| `block_size` | Model | Normal | `--grad_checkpoint` | Speed with checkpointing |
|---|---|---|---|---|
| 4096 | `n_embd 1024, n_head 16` (80M) | 14.5 GB | 4.4 GB | 2,900 chars/s |
| 4096 | `n_embd 1024, n_head 4` (80M) | 4.3 GB | 1.2 GB | 7,500 chars/s |
| 8192 | `n_embd 1024, n_head 4` (84M) | | 4.5 GB | 4,400 chars/s |
| 8192 | `n_embd 512, n_head 4, n_layer 4` (17M) | | 4.5 GB | 8,500 chars/s |

Pick `batch_size` so the total stays under ~90 GB on the halo. The real fix is a fused
attention kernel that never stores the whole table (`--sdpa`), but that crashes during
training on the halo's current PyTorch build.

## Read the code in this order

| File | What it teaches |
|---|---|
| `prepare.py` | Tokenization: text becomes integers. Train/val split. |
| `tokenizer.py` | BPE: learning subword tokens from data, encoding and decoding. |
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

- **Fine-tuning a pretrained model.** Take an existing small open model (e.g. from Hugging Face)
  and fine-tune it on your own data, with LoRA to keep memory low. This is where most
  real-world "training an LLM" actually happens.
- **Instruction tuning.** Train on prompt → response pairs so the model follows instructions
  instead of just continuing text. This is the step that turns a base model into a chat assistant.
