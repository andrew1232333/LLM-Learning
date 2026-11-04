"""
Byte-pair encoding (BPE): the tokenizer real LLMs use, with a split rule tuned for Python.

Start with one token per character, then repeatedly find the most common pair of
neighbouring tokens in the training text and make it a new token. After a few thousand
merges, common things like "self", "return", "):" and a newline plus 8 spaces of
indentation are each a single token, while rare names are spelled from smaller pieces
("_parse" + "_header" + "_params"). Shorter sequences mean the model sees more text
in the same context window and spends less effort on spelling.

A tokenizer with no merges is exactly the character-level tokenizer, so this one class
handles both.

Run:  uv run tokenizer.py --dataset python+docs   # show how some code gets split
"""

import argparse
import heapq
import re
from collections import Counter, defaultdict
from pathlib import Path

# Before counting pairs, split the text into chunks that merges may never cross. This is
# where the Python knowledge lives: a token can be "self" or "):" but never "f.n", and the
# indentation that starts a line is kept together so it becomes one token.
SPLIT = re.compile(r"""
      \x03                         # end of text: always a token of its own (see EOT)
    | \n[ ]*                       # newline + the next line's indentation (Python's blocks)
    | [ ]?[A-Za-z_][A-Za-z0-9_]*   # a name or keyword, with the space before it
    | [ ]?[0-9]{1,3}               # numbers, at most 3 digits per chunk
    | [ ]?[^\sA-Za-z0-9_\x03]+     # operators and punctuation: ), ==, \"\"\", ->
    | [ ]+                         # other runs of spaces (alignment, trailing)
    | \s                           # any other whitespace character
""", re.VERBOSE)

# End of text. Big datasets put it between documents, so the model learns where one file
# stops and the next, unrelated one starts; instruction tuning puts it after each answer,
# so the model learns to stop. It's a control character that never appears in the data.
EOT = "\x03"


class Tokenizer:
    def __init__(self, chars, merges=()):
        self.chars = list(chars)
        self.merges = [tuple(m) for m in merges]  # merge i creates token id len(chars) + i
        self.vocab = list(self.chars)              # id -> the text it stands for
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.char_ids = {c: i for i, c in enumerate(self.chars)}
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        unknown = set(text) - set(self.char_ids)
        if unknown:
            raise ValueError(f"characters the tokenizer never saw: {unknown}")
        ids = []
        for chunk in SPLIT.findall(text):
            if chunk not in self.cache:  # the same chunks repeat endlessly, so remember them
                if len(self.cache) > 1_000_000:  # but don't let rare ones fill the memory
                    self.cache.clear()
                self.cache[chunk] = self.encode_chunk(chunk)
            ids.extend(self.cache[chunk])
        return ids

    def encode_chunk(self, chunk):
        # Replay the merges in the order they were learned: always apply the earliest one
        # that fits, exactly as training would have.
        ids = [self.char_ids[c] for c in chunk]
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = merge(ids, pair, len(self.chars) + self.rank[pair])
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    @classmethod
    def train(cls, text, vocab_size, chars=None):
        """Learn merges from text until the vocabulary reaches vocab_size. Pass `chars` to
        allow characters the training text happens not to contain."""
        chars = sorted(set(text) | set(chars or ()))
        char_ids = {c: i for i, c in enumerate(chars)}
        # Work on distinct chunks with a count, not the whole text: "self" appears
        # ~100k times but only needs to be merged once.
        counts = Counter(SPLIT.findall(text))
        words = [[char_ids[c] for c in chunk] for chunk in counts]
        freq = list(counts.values())

        pairs = Counter()             # (a, b) -> how often it occurs in the text
        where = defaultdict(set)      # (a, b) -> which words contain it
        for w, ids in enumerate(words):
            for p in zip(ids, ids[1:]):
                pairs[p] += freq[w]
                where[p].add(w)

        # Finding the most common pair by scanning all of them is slow once there are
        # millions, so keep them in a heap too. Counts change as words merge, so the heap
        # holds stale entries; skip any whose count no longer matches.
        heap = [(-n, p) for p, n in pairs.items()]
        heapq.heapify(heap)
        changed = set()  # pairs whose count changed in this merge
        merges = []
        while len(chars) + len(merges) < vocab_size and heap:
            n, best = heapq.heappop(heap)
            if pairs.get(best) != -n:
                continue
            new_id = len(chars) + len(merges)
            merges.append(best)
            # Only words containing the pair change. Take their old pairs out of the
            # counts, merge, and put the new pairs back in.
            for w in where.pop(best):
                old = words[w]
                for p in zip(old, old[1:]):
                    pairs[p] -= freq[w]
                    changed.add(p)
                    if pairs[p] == 0:
                        del pairs[p]
                words[w] = merge(old, best, new_id)
                for p in zip(words[w], words[w][1:]):
                    pairs[p] += freq[w]
                    where[p].add(w)
                    changed.add(p)
            for p in changed:
                if p in pairs:
                    heapq.heappush(heap, (-pairs[p], p))
            changed.clear()
        return cls(chars, merges)


def merge(ids, pair, new_id):
    """Replace every occurrence of pair in ids with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def main():
    # A quick look at what the tokenizer learns, without preparing a whole dataset.
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="python")
    p.add_argument("--vocab_size", type=int, default=2048)
    args = p.parse_args()
    text = (Path(__file__).parent / "data" / args.dataset / "input.txt").read_text()
    tok = Tokenizer.train(text, args.vocab_size)
    sample = 'def greet(self, name):\n        """Say hello."""\n        return f"Hello, {name}!"\n'
    ids = tok.encode(sample)
    print(f"{len(sample)} characters -> {len(ids)} tokens:")
    print(" | ".join(repr(tok.decode([i]))[1:-1] for i in ids))
    print("\nLast merges learned:", [tok.vocab[i] for i in range(tok.vocab_size - 15, tok.vocab_size)])


if __name__ == "__main__":
    main()
