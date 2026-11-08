"""Collect MLP inputs for writing new neurons by hand (no training).

For blocks in --blocks, saves the input of the block MLP (the normed
residual after the attention of that block) at:
  products:  the four answer positions of A * B (probe.mul2x2, teacher
             forced as in probe.py), fit set seed 2 and test set seed 3;
  stories:   every position of data/stories.jsonl lines 0-99 (up to 256
             tokens each), as states where new neurons must stay silent.
Output: runs/carry/mlpin_L{block}.npz (float16), with the operand digits.

Example:
    uv run python carrycollect.py --blocks 19,21,23,25
"""

import argparse
import json

import mlx.core as mx
import numpy as np
from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT
from probe import WIDTH, mul2x2, reply_ids

OUT = ROOT / "runs" / "carry"


def mlp_inputs(model, x: mx.array, blocks: list[int]) -> dict[int, mx.array]:
    h = model.model.embed_tokens(x)
    got = {}
    for i, blk in enumerate(model.model.layers[: max(blocks) + 1]):
        h = h + blk.self_attn(blk.input_layernorm(h), "causal", None)
        n = blk.post_attention_layernorm(h)
        if i in blocks:
            got[i] = n
        h = h + blk.mlp(n)
    return got


def product_states(
    model, tok, pairs, blocks, chunk: int = 250
) -> dict[int, np.ndarray]:
    x, first = reply_ids(tok, pairs)
    pos = mx.arange(first - 1, first - 1 + WIDTH)
    parts: dict[int, list[np.ndarray]] = {b: [] for b in blocks}
    for i in range(0, len(pairs), chunk):
        got = mlp_inputs(model, x[i : i + chunk], blocks)
        for b in blocks:
            parts[b].append(np.array(got[b][:, pos].astype(mx.float16)))
    return {b: np.concatenate(v) for b, v in parts.items()}


def story_states(model, tok, blocks, n: int = 100, max_tokens: int = 256):
    lines = (ROOT / "data" / "stories.jsonl").read_text().splitlines()[:n]
    parts: dict[int, list[np.ndarray]] = {b: [] for b in blocks}
    for line in lines:
        ids = tok.encode(json.loads(line)["text"])[:max_tokens]
        got = mlp_inputs(model, mx.array([ids]), blocks)
        for b in blocks:
            parts[b].append(np.array(got[b][0].astype(mx.float16)))
    return {b: np.concatenate(v) for b, v in parts.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--blocks", default="19,21,23,25")
    p.add_argument("--n-fit", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    a = p.parse_args()

    blocks = [int(b) for b in a.blocks.split(",")]
    model, tok = load(a.model)
    fit = mul2x2(a.n_fit, 2)
    test = mul2x2(a.n_test, 3, exclude=set(fit))
    pf = product_states(model, tok, fit, blocks)
    pt = product_states(model, tok, test, blocks)
    st = story_states(model, tok, blocks)
    OUT.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        np.savez(
            OUT / f"mlpin_L{b}.npz",
            fit=pf[b],
            test=pt[b],
            stories=st[b],
            fit_pairs=np.array(fit),
            test_pairs=np.array(test),
        )
        print(f"block {b}: fit {pf[b].shape} test {pt[b].shape} stories {st[b].shape}")


if __name__ == "__main__":
    main()
