"""Find MLP neurons that encode a digit of a two-digit sum, and test them.

At the position that writes the units digit of A + B (see circuit.py), the
script records the hidden activations of one MLP (the input of down_proj).
For each neuron, eta squared (the share of activation variance explained by
a label) is computed for four labels: the units digit of the sum, the units
digit of A, the units digit of B, and the carry (A and B units sum to 10 or
more).

Then the top neurons for the sum units digit are zeroed (k of them, and a
random set of k as control), and the units-digit accuracy is measured on new
additions: the share where the right digit is the top token at that
position.

Example:
    uv run python tuning.py --layer 21
"""

import argparse
import json
import random

import mlx.core as mx
from mlx.nn import silu
from mlx_lm import load

from circuit import additions, batch_ids, digit_logprobs, run_blocks
from evalmath import DEFAULT_MODEL, ROOT
from surgery import Surgeon


def mlp_hidden(model, x: mx.array, layer: int, pos: int) -> mx.array:
    """Input of down_proj of one block at one position: [N, n_neurons]."""
    store = {}
    blk = model.model.layers[layer]
    mlp = blk.mlp

    class Hook:
        def __call__(self, h):
            a = silu(mlp.gate_proj(h)) * mlp.up_proj(h)
            store["a"] = a[:, pos]
            return mlp.down_proj(a)

    blk.mlp = Hook()
    try:
        run_blocks(model, x)
    finally:
        blk.mlp = mlp
    return store["a"].astype(mx.float32)


def eta_squared(act: mx.array, labels: list[int], n_classes: int) -> mx.array:
    """Share of each neuron's variance explained by the label: [n_neurons]."""
    y = mx.array(labels)
    total = ((act - act.mean(0)) ** 2).sum(0)
    between = mx.zeros(act.shape[1])
    for c in range(n_classes):
        mask = (y == c)[:, None]
        n_c = mask.sum()
        if n_c.item() == 0:
            continue
        mean_c = (act * mask).sum(0) / n_c
        between = between + n_c * (mean_c - act.mean(0)) ** 2
    return between / mx.maximum(total, 1e-12)


def class_mean(v: mx.array, y: mx.array, c: int) -> float:
    mask = y == c
    return ((v * mask).sum() / mx.maximum(mask.sum(), 1)).item()


def units_accuracy(model, tok, pairs) -> float:
    x, first = batch_ids(tok, pairs)
    h, _, _ = run_blocks(model, x)
    lp = digit_logprobs(model, h, first + 1, tok)
    want = mx.array([(a + b) % 10 for a, b in pairs])
    return (mx.argmax(lp, axis=-1) == want).mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ks", default="4,16,64")
    a = p.parse_args()

    model, tok = load(a.model)
    pairs = additions(a.n, seed=2)
    x, first = batch_ids(tok, pairs)
    acts = []
    for i in range(0, a.n, 250):
        acts.append(mlp_hidden(model, x[i : i + 250], a.layer, first + 1))
    act = mx.concatenate(acts)
    labels = {
        "sum_units": ([(u + v) % 10 for u, v in pairs], 10),
        "a_units": ([u % 10 for u, _ in pairs], 10),
        "b_units": ([v % 10 for _, v in pairs], 10),
        "carry": ([int(u % 10 + v % 10 >= 10) for u, v in pairs], 2),
    }
    eta = {k: eta_squared(act, y, c) for k, (y, c) in labels.items()}
    order = mx.argsort(-eta["sum_units"]).tolist()
    print(f"Block {a.layer} MLP, {act.shape[1]} neurons, {a.n} additions")
    for t in (0.3, 0.5, 0.7):
        counts = {k: int((v > t).sum().item()) for k, v in eta.items()}
        print(f"neurons with eta^2 > {t}: {counts}")
    print("\ntop neurons for the sum units digit:")
    print(
        "neuron  sum_units  a_units  b_units  carry  mean activation per sum digit 0-9"
    )
    y = mx.array(labels["sum_units"][0])
    top_rows = []
    for j in order[:12]:
        means = [class_mean(act[:, j], y, d) for d in range(10)]
        cells = " ".join(f"{m:5.2f}" for m in means)
        e = {k: eta[k][j].item() for k in eta}
        print(
            f"{j:6d}  {e['sum_units']:9.2f}  {e['a_units']:7.2f}  {e['b_units']:7.2f}"
            f"  {e['carry']:5.2f}  {cells}"
        )
        top_rows.append({"neuron": j, **e, "means": means})

    test = additions(500, seed=3)
    s = Surgeon(model)
    base = units_accuracy(model, tok, test)
    print(f"\nunits-digit accuracy on 500 new additions: base {base:.3f}")
    rng = random.Random(0)
    ablation = {"base": base}
    for k in [int(v) for v in a.ks.split(",")]:
        s.zero_neurons(a.layer, order[:k])
        top = units_accuracy(model, tok, test)
        s.restore()
        s.zero_neurons(a.layer, rng.sample(range(act.shape[1]), k))
        rand = units_accuracy(model, tok, test)
        s.restore()
        ablation[f"top{k}"], ablation[f"rand{k}"] = top, rand
        print(f"zero top {k:3d}: {top:.3f}   zero random {k:3d}: {rand:.3f}")

    out = ROOT / "runs" / f"tuning{a.layer}.json"
    out.write_text(
        json.dumps(
            {
                "layer": a.layer,
                "order": order[:256],
                "top": top_rows,
                "ablation": ablation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
