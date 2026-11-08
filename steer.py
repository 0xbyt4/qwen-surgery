"""Write an answer digit into the model by setting a few MLP activations.

Sufficiency test for the digit neurons of tuning.py. For each digit d, the
mean activation of each neuron over examples whose answer digit is d is
taken from one set of examples. On new examples, a target digit t (never the
right one) is chosen, and the activations of a neuron set are replaced by
their means for t at the position that writes the digit. The score is the
share of examples whose top digit becomes t.

Neuron sets: the top k for the answer digit (from runs/tuning<L>*.json), a
random set of k, and all neurons of the MLP. With --layers, only all neurons
are set, in each listed block, to find the blocks that write the digit.

Example:
    uv run python steer.py --layer 21 --task add_units
    uv run python steer.py --task mul_units --layers 14-27
"""

import argparse
import json
import random

import mlx.core as mx
from mlx.nn import silu
from mlx_lm import load

from circuit import batch_ids, digit_logprobs, run_blocks
from evalmath import DEFAULT_MODEL, ROOT
from tuning import TASKS, Task, activations


def class_means(act: mx.array, y: list[int]) -> mx.array:
    """[10, n_neurons] mean activation per answer digit."""
    yy = mx.array(y)
    rows = []
    for d in range(10):
        mask = (yy == d)[:, None]
        rows.append((act * mask).sum(0) / mx.maximum(mask.sum(), 1))
    return mx.stack(rows)


def steered_digits(
    model, tok, task: Task, pairs, layer: int, neurons: list[int], values: mx.array
) -> list[int]:
    """Top digit after setting the activations of `neurons` to `values` [N, k]."""
    x, first = batch_ids(tok, pairs, task.op)
    pos = first - 1 + task.digit
    blk = model.model.layers[layer]
    mlp = blk.mlp
    n_neurons = mlp.down_proj.weight.shape[1]
    sel = mx.zeros(n_neurons).at[mx.array(neurons)].add(1.0)
    full = mx.zeros((len(pairs), n_neurons)).at[:, mx.array(neurons)].add(values)

    class Hook:
        def __call__(self, h):
            a = silu(mlp.gate_proj(h)) * mlp.up_proj(h)
            at = a[:, pos].astype(mx.float32)
            a[:, pos] = (at * (1 - sel) + full * sel).astype(a.dtype)
            return mlp.down_proj(a)

    blk.mlp = Hook()
    try:
        h, _, _ = run_blocks(model, x)
    finally:
        blk.mlp = mlp
    lp = digit_logprobs(model, h, pos, tok)
    return mx.argmax(lp, axis=-1).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--task", choices=sorted(TASKS), default="add_units")
    p.add_argument("--ks", default="4,16,64,256")
    p.add_argument("--layers", help="first-last: set all neurons, block by block")
    a = p.parse_args()
    task = TASKS[a.task]

    model, tok = load(a.model)
    fit = task.data(1000, 2)
    fit_y = [task.target(x, y) for x, y in fit]
    test = task.data(500, 3)
    rng = random.Random(0)
    right = [task.target(x, y) for x, y in test]
    goal = [(d + rng.randint(1, 9)) % 10 for d in right]
    goal_rows = mx.array(goal)

    def score(layer: int, means: mx.array, neurons: list[int]) -> tuple[float, float]:
        values = means[goal_rows][:, mx.array(neurons)]
        top = steered_digits(model, tok, task, test, layer, neurons, values)
        hit = sum(t == g for t, g in zip(top, goal, strict=True)) / len(test)
        kept = sum(t == r for t, r in zip(top, right, strict=True)) / len(test)
        return hit, kept

    if a.layers:
        first, last = (int(v) for v in a.layers.split("-"))
        print(f"Task {a.task}, all neurons of one MLP set: moved (still right)")
        results = {}
        for layer in range(first, last + 1):
            means = class_means(activations(model, tok, task, fit, layer), fit_y)
            results[layer] = score(layer, means, list(range(means.shape[1])))
            hit, kept = results[layer]
            print(f"block {layer:2d}: {hit:.3f} ({kept:.3f})", flush=True)
        out = ROOT / "runs" / f"steer_layers_{a.task}.json"
        out.write_text(json.dumps({"task": a.task, **results}, indent=2))
        return

    name = f"tuning{a.layer}" + ("" if a.task == "add_units" else f"_{a.task}")
    order = json.loads((ROOT / "runs" / f"{name}.json").read_text())["order"]
    means = class_means(activations(model, tok, task, fit, a.layer), fit_y)
    n_neurons = means.shape[1]

    print(
        f"Block {a.layer}, task {a.task}: share moved to the target digit "
        "(share still right)"
    )
    results = {}
    for k in [int(v) for v in a.ks.split(",")]:
        top = score(a.layer, means, order[:k])
        rand = score(a.layer, means, rng.sample(range(n_neurons), k))
        results[f"top{k}"], results[f"rand{k}"] = top, rand
        print(
            f"k {k:4d}: top {top[0]:.3f} ({top[1]:.3f})   "
            f"random {rand[0]:.3f} ({rand[1]:.3f})",
            flush=True,
        )
    every = score(a.layer, means, list(range(n_neurons)))
    results["all"] = every
    print(f"all {n_neurons}: {every[0]:.3f} ({every[1]:.3f})")
    out = ROOT / "runs" / f"steer{a.layer}_{a.task}.json"
    out.write_text(json.dumps({"layer": a.layer, "task": a.task, **results}, indent=2))


if __name__ == "__main__":
    main()
