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
| `python+docs+packages` | That plus the code of every package installed alongside Python (site-packages / dist-packages). Needs `python+docs` prepared first | depends on the machine: 155M characters on the halo |

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
- `last.pt`: the latest model, saved every `ckpt_interval` steps (default 100). With
  several machines, each one keeps its own
- `settings.json`: every setting that produced the run

Both checkpoints work with `sample.py --ckpt`, so you can sample a run while it's still training.

To continue a run you stopped, pass `--resume` with its `out_dir`. It reloads `last.pt` (weights,
optimizer state and step) and the run's `settings.json`, so nothing else needs retyping:

```bash
uv run train.py --resume --out_dir out/medium                   # finish where it stopped
uv run train.py --resume --out_dir out/medium --max_iters 8000  # or train for longer
```

It restarts from the last `ckpt_interval` save, so up to `ckpt_interval` steps are redone. Raising `max_iters`
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

## Training on several machines

`--sync_every N` trains one model on several machines at once (`diloco.py`). Each machine
trains its own copy on its own batches for N seconds, then they average how far their
weights moved (weighted by steps done, so a faster machine counts for more) and carry on
from the same weights. One machine also runs the server with `--serve`:

```bash
# halo (serves and trains)
python train.py --config configs/code-docs.toml --sync_every 60 --serve --workers 2
# Mac (trains), same settings, pointed at the halo
python train.py --config configs/code-docs.toml --sync_every 60 --sync_addr 192.168.4.103:29500
```

`max_iters`, the learning-rate schedule and `eval_interval` count steps across all machines.
Only the serving machine evaluates and saves checkpoints. After each sync every machine
prints the same `weight norm`, which shows the copies agree again.

Averaging two machines' moves is like one step with a batch twice as big, not two steps:
on the halo + Studio it ran 1.9x the steps per second but reached the same loss only ~1.25x
sooner. `--outer_lr` and `--outer_momentum` (set on the serving machine) treat each averaged
move like a gradient for an outer optimizer with Nesterov momentum, as DiLoCo does. On the
same halo + Studio run, `--outer_momentum 0.3` beat plain averaging at every checkpoint
(best 0.541 vs 0.558 per character in the same time), reaching plain averaging's final loss
~20% sooner, and improved both code and docs on python+docs' val text. The DiLoCo paper's
0.9 overshot in a small test (rounds there were ~100 steps; the paper's were 500).

How often to sync barely matters: in a small test, syncing every ~5 steps and every ~44
steps gave the same loss. What costs is splitting the work at all, since averaged moves act
like one step on a bigger batch. A faster network only shortens the sync itself.

If any machine stops (crash, Wi-Fi drop), the others stop with an error too rather than
waiting forever, and every machine can `--resume` from its own `last.pt`.

Ordinary data parallelism averages gradients after every step, so it needs a network
nearly as fast as the GPU (10+ Gb/s). Averaging once a minute sends the model across once per
round instead: a 27M-parameter model is 55 MB in bfloat16, a few seconds over Wi-Fi.

## A bigger run: 236M parameters on 6B tokens of code

The steps above scale up to a GPT-2-small-sized model trained for about a week on two
machines, then instruction-tuned and scored on real coding benchmarks:

```bash
bash download.sh                                    # ~13 GB: GitHub Python, instructions, benchmarks
uv run prepare_big.py --shards 24 --vocab_size 16384   # -> data/code-bpe16384, 6.0B tokens
uv run train.py --dataset code-bpe16384 --n_layer 16 --n_embd 1024 --n_head 16 \
    --block_size 1024 --batch_size 16 --dropout 0 --max_iters 280000 --lr 3e-4 ...
uv run finetune.py --ckpt out/code-236m --out_dir out/code-236m-sft
uv run evaluate.py --ckpt out/code-236m-sft
```

- **Data**: `codeparrot/codeparrot-clean`, ~5M Python files from GitHub with duplicates
  removed. The first 24 of its 54 shards, after dropping non-ASCII files, are 22.8G
  characters. `prepare_big.py` streams them through 25 processes (~8 minutes on the halo).
- **End-of-text token**: every file ends with one, so the model learns where a file stops.
  Instruction tuning puts one after each answer, so the model learns to stop talking.
- **Tokenizer**: 16,384 tokens learned from 200 MB of the code, ~3.8 characters per token.
- **Sizing**: at ~4,600 tokens/s per machine, a week on two machines is ~5B tokens, about
  20 tokens per parameter for 236M parameters (the "Chinchilla" rule of thumb).
- **Dropout 0**: the run sees each token less than once, so there's nothing to memorize.
- **Instruction tuning** (`finetune.py`): ~85k Python request → answer pairs from
  Magicoder. Only the answer tokens count toward the loss.
- **Scoring** (`evaluate.py`): HumanEval and MBPP, pass@1. It runs the generated code
  against the benchmark's tests. The 27M model above scores 0% on both.

## Read the code in this order

| File | What it teaches |
|---|---|
| `prepare.py` | Tokenization: text becomes integers. Train/val split. |
| `tokenizer.py` | BPE: learning subword tokens from data, encoding and decoding. |
| `model.py` | The transformer: embeddings, attention, MLP, residuals, layer norm. |
| `train.py` | The training loop: batches, loss, backprop, optimizer, learning-rate schedule. |
| `sample.py` | Generation: temperature, top-k, feeding outputs back in, the KV cache. |
| `prepare_big.py` | Preparing tens of GB: streaming, many processes, memory-mapped files. |
| `finetune.py` | Instruction tuning: prompt format, training only on the answers. |
| `evaluate.py` | Scoring generated code by running it against unit tests. |
| `bench.py` | Measuring speed, and checking a speedup didn't change the math. |
| `diloco.py` | Training on several machines by averaging weights every so often. |

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
