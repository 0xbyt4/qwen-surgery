"""Activation patching for two-digit products, block by block and head by head.

Pairs of products A * B and A * B' differ only in the units digit of B, so
the hundreds and tens digits of the product change. In the model's reply
format ("A × B = P", teacher forcing), both runs get the digits of the base
product A * B as the forced answer, so the two inputs differ only in B. The
source digit is the one A * B' would have. The attention or MLP output of one
block (or the output of one head) at the position that writes a digit is
copied from the source run into the base run. Score, per digit:
(lp_src - lp_base after the patch - before) / (full source - before),
where lp_x is the log probability of digit x. 0 = no effect, 1 = the patch
alone moves the digit to the source value.

Example:
    uv run python mulpatch.py blocks
    uv run python mulpatch.py heads --layers 18,19,20
"""

import argparse
import json
import random

import mlx.core as mx
from mlx_lm import load

from circuit import digit_logprobs, run_blocks
from evalmath import DEFAULT_MODEL, ROOT
from probe import reply_ids

DIGITS = {1: "hundreds", 2: "tens"}


def pairs(n: int, seed: int) -> tuple[list, list]:
    r = random.Random(seed)
    base, src = [], []
    while len(base) < n:
        a, b = r.randint(10, 99), r.randint(10, 99)
        b2 = b - b % 10 + r.randint(0, 9)
        pa, pb = str(a * b), str(a * b2)
        if b2 != b and len(pa) == len(pb) == 4 and pa[1:3] != pb[1:3]:
            base.append((a, b))
            src.append((a, b2))
    return base, src


def head_outputs(blk, h: mx.array) -> mx.array:
    """Per-head attention output before o_proj: [N, L, heads, head_dim]."""
    store = {}
    attn = blk.self_attn
    o_proj = attn.o_proj

    class Hook:
        def __call__(self, z):
            store["z"] = z
            return o_proj(z)

    attn.o_proj = Hook()
    try:
        attn(blk.input_layernorm(h), "causal", None)
    finally:
        attn.o_proj = o_proj
    z = store["z"]
    return z.reshape(*z.shape[:2], attn.n_heads, -1)


def run_head_patch(model, x, layer: int, head: int, pos: int, value: mx.array):
    """Forward pass with the pre-o_proj output of one head at pos replaced."""
    blk = model.model.layers[layer]
    attn = blk.self_attn
    o_proj = attn.o_proj
    d = o_proj.weight.shape[1] // attn.n_heads

    class Hook:
        def __call__(self, z):
            z[:, pos, head * d : (head + 1) * d] = value
            return o_proj(z)

    attn.o_proj = Hook()
    try:
        return run_blocks(model, x)[0]
    finally:
        attn.o_proj = o_proj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["blocks", "heads"])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--layers", default="16,17,18,19,20,21,22,23")
    a = p.parse_args()

    model, tok = load(a.model)
    base, src = pairs(a.n, 1)
    xb, first = reply_ids(tok, base)
    xs, _ = reply_ids(tok, src, [x * y for x, y in base])
    hb, _, _ = run_blocks(model, xb)
    hs, resid_s, parts_s = run_blocks(model, xs, keep=True)
    rows = mx.arange(a.n)
    clean = {}
    for k in DIGITS:
        pos = first - 1 + k
        db = mx.array([int(str(x * y)[k]) for x, y in base])
        ds = mx.array([int(str(x * y)[k]) for x, y in src])
        lb = digit_logprobs(model, hb, pos, tok)
        ls = digit_logprobs(model, hs, pos, tok)
        clean[k] = (
            pos,
            db,
            ds,
            lb[rows, ds] - lb[rows, db],
            ls[rows, ds] - ls[rows, db],
        )

    def score(k: int, h: mx.array) -> float:
        pos, db, ds, before, full = clean[k]
        lp = digit_logprobs(model, h, pos, tok)
        after = lp[rows, ds] - lp[rows, db]
        gap = full - before
        ok = mx.abs(gap) > 1e-3
        return mx.where(ok, (after - before) / mx.where(ok, gap, 1), 0).mean().item()

    out = {}
    if a.what == "blocks":
        print(
            "block  "
            + "  ".join(f"{p}@{d:<8}" for p in ("attn", "mlp") for d in DIGITS.values())
        )
        for i in range(len(model.model.layers)):
            cells = []
            for part in ("attn", "mlp"):
                for k in DIGITS:
                    pos = clean[k][0]
                    h, _, _ = run_blocks(
                        model, xb, patch=(i, part, pos, parts_s[(i, part)][:, pos])
                    )
                    cells.append(score(k, h))
            out[i] = cells
            print(f"{i:5d}  " + "  ".join(f"{v:13.2f}" for v in cells), flush=True)
    else:
        for layer in [int(v) for v in a.layers.split(",")]:
            prev = resid_s[layer - 1] if layer > 0 else model.model.embed_tokens(xs)
            z = head_outputs(model.model.layers[layer], prev)
            n_heads = z.shape[2]
            cells = {}
            for k, name in DIGITS.items():
                pos = clean[k][0]
                cells[name] = [
                    score(k, run_head_patch(model, xb, layer, hd, pos, z[:, pos, hd]))
                    for hd in range(n_heads)
                ]
            out[layer] = cells
            for name, v in cells.items():
                print(
                    f"block {layer:2d} {name:8s} " + " ".join(f"{x:5.2f}" for x in v),
                    flush=True,
                )
    (ROOT / "runs" / f"mulpatch_{a.what}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
