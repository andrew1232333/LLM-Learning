# tiny-llm

A small GPT-style language model built from scratch in PyTorch, sized to train on a
laptop in a few minutes. Every line is meant to be read; the comments explain *why*.

## Run it

```bash
uv run prepare.py   # download text, turn characters into numbers
uv run train.py     # ~3 min on an M5 MacBook, ~0.8M parameters
uv run sample.py --prompt "ROMEO:"
```

## Read the code in this order

| File | What it teaches |
|---|---|
| `prepare.py` | Tokenization: text becomes integers. Train/val split. |
| `model.py` | The transformer: embeddings, attention, MLP, residuals, layer norm. |
| `train.py` | The training loop: batches, loss, backprop, optimizer, learning-rate schedule. |
| `sample.py` | Generation: temperature, top-k, feeding outputs back in. |

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
4. **Scale up.** `--n_layer 6 --n_head 6 --n_embd 384 --block_size 256` (~10M params, ~20-30 min).
   Does val loss improve? Does the gap with train loss grow? Try raising `--dropout` to 0.2.
5. **Your own data.** Replace `data/input.txt` with any text (delete the `.bin` files and rerun
   `prepare.py`). Code, song lyrics, your own notes.
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
