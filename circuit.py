"""Tools to look at the residual stream of the math answer, block by block.

The data is two-digit addition, "What is A + B?", with a three-digit sum, in
the direct prompt followed by "The answer is " and the sum. Each digit of the
sum is one token, so the positions of all examples line up.

  lens:  the residual after each block, read through the final norm and the
         output head ("logit lens"); the share of examples where the right
         digit is the top token, for the hundreds, tens and units digit.
  patch: pairs that differ only in the units digit of B. The attention or MLP
         output of one block, at the position that writes one answer digit,
         is copied from the source run into the base run. The score is
         (lp_src - lp_base after the patch - before) / (full source - before),
         where lp_x is the log probability of digit x: 0 means no effect,
         1 means the patch alone moves the answer to the source digit.

Example:
    uv run python circuit.py lens
    uv run python circuit.py patch
"""

import argparse
import random

import mlx.core as mx
from mlx_lm import load

from evalmath import DEFAULT_MODEL, head_f32, scoring_ids

DIGITS = ["hundreds", "tens", "units"]


def additions(n: int, seed: int) -> list[tuple[int, int]]:
    r = random.Random(seed)
    out = []
    while len(out) < n:
        a, b = r.randint(10, 99), r.randint(10, 99)
        if a + b >= 100:
            out.append((a, b))
    return out


def batch_ids(tok, pairs: list[tuple[int, int]]) -> tuple[mx.array, int]:
    """Token ids [N, L] and the position of the first answer digit."""
    rows = []
    for a, b in pairs:
        ids, n = scoring_ids(tok, f"What is {a} + {b}?", a + b)
        if n != len(ids) - 3:
            raise ValueError(f"answer {a + b} is not three tokens")
        rows.append(ids)
    if len({len(x) for x in rows}) != 1:
        raise ValueError("prompts differ in length")
    return mx.array(rows), len(rows[0]) - 3


def run_blocks(model, x: mx.array, keep: bool = False, patch=None):
    """Forward pass with access to each block. patch = (layer, part, pos, value)
    replaces the attention ("attn") or MLP ("mlp") output of one block at one
    position. Returns the final hidden state and, with keep, the residual
    after each block and the outputs of each part."""
    h = model.model.embed_tokens(x)
    resid, parts = [], {}
    for i, blk in enumerate(model.model.layers):
        r = blk.self_attn(blk.input_layernorm(h), "causal", None)
        if patch and patch[0] == i and patch[1] == "attn":
            r[:, patch[2]] = patch[3]
        h = h + r
        m = blk.mlp(blk.post_attention_layernorm(h))
        if patch and patch[0] == i and patch[1] == "mlp":
            m[:, patch[2]] = patch[3]
        h = h + m
        if keep:
            resid.append(h)
            parts[(i, "attn")] = r
            parts[(i, "mlp")] = m
    return model.model.norm(h), resid, parts


def digit_logprobs(model, h: mx.array, pos: int, tok) -> mx.array:
    """log p over the ten digit tokens at one position: [N, 10]."""
    ids = mx.array([tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)])
    logits = head_f32(model, h[:, pos])
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return lp[:, ids]


def lens(model, tok, n: int) -> None:
    pairs = additions(n, seed=0)
    x, first = batch_ids(tok, pairs)
    _, resid, _ = run_blocks(model, x, keep=True)
    sums = [a + b for a, b in pairs]
    want = [[int(str(s)[k]) for s in sums] for k in range(3)]
    print("Share of examples where the right digit is the top token (logit lens)")
    print("block  " + "  ".join(f"{d:>8}" for d in DIGITS))
    for i, h in enumerate(resid):
        hn = model.model.norm(h)
        row = []
        for k in range(3):
            pos = first - 1 + k  # this position predicts digit k
            logits = head_f32(model, hn[:, pos])
            top = mx.argmax(logits, axis=-1).tolist()
            right = [tok.encode(str(d), add_special_tokens=False)[0] for d in want[k]]
            row.append(sum(t == r for t, r in zip(top, right, strict=True)) / n)
        print(f"{i:5d}  " + "  ".join(f"{v:8.2f}" for v in row))


def patch(model, tok, n: int) -> None:
    r = random.Random(1)
    base, src = [], []
    while len(base) < n:
        a, b = r.randint(10, 99), r.randint(10, 99)
        b2 = b - b % 10 + r.randint(0, 9)
        if b2 != b and a + b >= 100 and a + b2 >= 100 and (a + b) % 10 != (a + b2) % 10:
            base.append((a, b))
            src.append((a, b2))
    xb, first = batch_ids(tok, base)
    xs, _ = batch_ids(tok, src)
    hb, _, _ = run_blocks(model, xb)
    hs, _, parts_s = run_blocks(model, xs, keep=True)
    rows = mx.arange(n)
    print("Patch score (0 = no effect, 1 = the answer digit moves to the source)")
    print(
        "block   " + "  ".join(f"{p}@{d:<8}" for p in ("attn", "mlp") for d in DIGITS)
    )
    targets = []
    for k in range(3):
        db = mx.array([int(str(a + b)[k]) for a, b in base])
        ds = mx.array([int(str(a + b)[k]) for a, b in src])
        targets.append((db, ds))
    clean = []
    for k in range(3):
        pos = first - 1 + k
        db, ds = targets[k]
        lb = digit_logprobs(model, hb, pos, tok)
        ls = digit_logprobs(model, hs, pos, tok)
        before = lb[rows, ds] - lb[rows, db]
        full = ls[rows, ds] - ls[rows, db]
        clean.append((before, full))
    for i in range(len(model.model.layers)):
        cells = []
        for part in ("attn", "mlp"):
            for k in range(3):
                pos = first - 1 + k
                db, ds = targets[k]
                before, full = clean[k]
                value = parts_s[(i, part)][:, pos]
                h, _, _ = run_blocks(model, xb, patch=(i, part, pos, value))
                lp = digit_logprobs(model, h, pos, tok)
                after = lp[rows, ds] - lp[rows, db]
                gap = full - before
                ok = mx.abs(gap) > 1e-3
                score = mx.where(ok, (after - before) / mx.where(ok, gap, 1), 0)
                cells.append(score.mean().item())
        print(f"{i:5d}   " + "  ".join(f"{v:13.2f}" for v in cells), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["lens", "patch"])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    model, tok = load(a.model)
    if a.what == "lens":
        lens(model, tok, a.n)
    else:
        patch(model, tok, a.n)


if __name__ == "__main__":
    main()
