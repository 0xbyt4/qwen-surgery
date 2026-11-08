"""Find the MLP neurons of one block that the math answers depend on, then zero
them and measure the effect.

Attribution: the output of each neuron (its column of down_proj) is scaled by
m_j = 1. The gradient of the summed answer log probability of math20.py with
respect to m gives, to first order, the change when neuron j is zeroed:
delta_j = -grad_j. "top" neurons have the largest grad_j (the answers lose the
most without them), "bottom" neurons the most negative grad_j.

Each edit is tested on math20.py (the questions used for the choice) and on a
held-out set from mathgen.py. Random sets of the same size are the control.

Example:
    uv run python neurons.py --layer 20 --name neurons20
"""

import argparse
import json
import random

import mlx.core as mx
from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, default_qa, run, scoring_ids, sequence_logprob
from mathgen import questions
from surgery import Surgeon


def attribution(model, tok, layer: int, qa: list[tuple[str, int]]) -> mx.array:
    lin = model.model.layers[layer].mlp.down_proj
    w0 = lin.weight

    def logprob(m: mx.array, ids: list[int], n_prompt: int) -> mx.array:
        lin.weight = w0 * m[None, :].astype(w0.dtype)
        return sequence_logprob(model, ids, n_prompt)

    grad = mx.grad(logprob)
    m = mx.ones(w0.shape[1], dtype=mx.float32)
    total = mx.zeros_like(m)
    for q, a in qa:
        total = total + grad(m, *scoring_ids(tok, q, a))
        mx.eval(total)
    lin.weight = w0
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--ks", default="1,4,16,64,256")
    p.add_argument("--random-seeds", type=int, default=3)
    p.add_argument("--heldout", type=int, default=40)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    model, tok = load(a.model)
    s = Surgeon(model)
    test = default_qa()
    held = questions(a.heldout, seed=1)
    grad = attribution(model, tok, a.layer, test)
    order = mx.argsort(-grad).tolist()  # largest gradient first
    top = {k: order[:k] for k in ks}
    bottom = {k: order[::-1][:k] for k in ks if k >= 16}

    edits: list[tuple[str, list[int]]] = [("base", [])]
    edits += [(f"top{k}", top[k]) for k in ks]
    edits += [(f"bottom{k}", v) for k, v in bottom.items()]
    for seed in range(a.random_seeds):
        r = random.Random(seed)
        edits += [(f"rand{k}_s{seed}", r.sample(range(s.n_neurons), k)) for k in ks]

    results = {}
    for key, neurons in edits:
        if neurons:
            s.zero_neurons(a.layer, neurons)
        row = {"neurons": neurons, "estimate": -sum(grad[j].item() for j in neurons)}
        for split, qa in (("math20", test), ("heldout", held)):
            items = run(model, tok, "direct", 256, qa)
            row[split] = {
                "correct": sum(x.correct for x in items),
                "total": len(items),
                "mean_logprob": sum(x.logprob for x in items) / len(items),
            }
        s.restore()
        results[key] = row
        t, h = row["math20"], row["heldout"]
        print(
            f"{key:>12}: math20 {t['correct']:2d}/{t['total']} "
            f"lp {t['mean_logprob']:7.3f} (est. sum change {row['estimate']:7.2f})"
            " | heldout "
            f"{h['correct']:2d}/{h['total']} lp {h['mean_logprob']:7.3f}",
            flush=True,
        )
    out = ROOT / "runs" / f"{a.name}.json"
    out.parent.mkdir(exist_ok=True)
    config = {"model": a.model, "layer": a.layer, "heldout": a.heldout}
    top_list = [{"neuron": j, "grad": grad[j].item()} for j in order[:32]]
    out.write_text(
        json.dumps({"config": config, "top": top_list, "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
