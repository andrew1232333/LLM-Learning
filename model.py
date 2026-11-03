"""
Step 2: the model. A small GPT-style transformer, written out in full.

The whole thing answers one question: "given these tokens, what token
comes next?" It outputs a score (logit) for every token in the vocabulary.

Shapes in comments use:
  B = batch size, T = sequence length (time), C = embedding size (channels)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 4096  # max context length: how many tokens the model can look back
    n_layer: int = 16       # number of transformer blocks stacked on top of each other
    n_head: int = 16        # attention heads per block
    n_embd: int = 256      # size of each token's vector
    dropout: float = 0.1
    sdpa: bool = False     # use PyTorch's fused attention kernel (same math; see README for when it helps)


class CausalSelfAttention(nn.Module):
    """
    Attention lets each token look at earlier tokens and pull in information
    from the ones that matter. Each token produces:
      - a query  ("what am I looking for?")
      - a key    ("what do I contain?")
      - a value  ("what will I share if you pick me?")
    The query-key dot product decides how much each earlier token's value
    gets mixed in. "Causal" means a token can only look backwards; otherwise
    it could cheat by peeking at the answer it's trying to predict.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.sdpa = cfg.sdpa
        self.dropout_p = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q, k, v in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: position t may attend to positions 0..t only.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # Split C into n_head smaller heads so each can specialize: (B, n_head, T, head_size)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if self.sdpa:
            # Exactly the four lines in the else-branch, done in one fused GPU kernel.
            # It never writes the full (T, T) score matrix to memory, which is the slow part.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0)
        else:
            # Attention scores: how relevant is every token to every other token. (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)  # each row now sums to 1
            att = self.dropout(att)
            y = att @ v  # weighted mix of values: (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # glue heads back together
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    """After tokens talk to each other (attention), each one 'thinks' on its own."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (MLP).

    The `x + ...` residual connections give gradients a straight path back
    through many layers, which is what makes deep networks trainable.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # position -> vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector -> logits
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy: how surprised was the model by the real next token?
            # A random guess over 65 characters gives ln(65) ~= 4.17.
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text one token at a time, feeding each output back in."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # only the last position matters
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample, don't just take the max
            idx = torch.cat([idx, next_id], dim=1)
        return idx
