"""Find MLP neurons that encode one answer digit, and test them.

Tasks (see circuit.py for the data; each answer has three digit tokens):
  add_units: the units digit of A + B (two-digit A and B);
  add_tens:  the tens digit of A + B;
  mul_units: the units digit of A * B (two-digit A, one-digit B).

At the position that writes that digit, the script records the hidden
activations of one MLP (the input of down_proj). For each neuron, eta squared
(the share of activation variance explained by a label) is computed for the
answer digit and for the operand digits (and the carry for additions).

Then the top neurons for the answer digit are zeroed (k of them, and a random
set of k as control), and the digit accuracy is measured on new examples: the
share where the right digit is the top token at that position.

Output: runs/tuning<L>.json for add_units, runs/tuning<L>_<task>.json for the
other tasks.

Example:
    uv run python tuning.py --layer 21
    uv run python tuning.py --layer 21 --task mul_units
"""

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from mlx.nn import silu
from mlx_lm import load

from circuit import additions, batch_ids, digit_logprobs, products, run_blocks
from evalmath import DEFAULT_MODEL, ROOT
from surgery import Surgeon


@dataclass
class Task:
    data: Callable[[int, int], list[tuple[int, int]]]
    op: str
    digit: int  # 0 hundreds, 1 tens, 2 units
    target: Callable[[int, int], int]
    labels: dict[str, tuple[Callable[[int, int], int], int]]


def carry(a: int, b: int) -> int:
    return int(a % 10 + b % 10 >= 10)


TASKS = {
    "add_units": Task(
        additions,
        "+",
        2,
        lambda a, b: (a + b) % 10,
        {
            "sum_units": (lambda a, b: (a + b) % 10, 10),
            "a_units": (lambda a, b: a % 10, 10),
            "b_units": (lambda a, b: b % 10, 10),
            "carry": (carry, 2),
        },
    ),
    "add_tens": Task(
        additions,
        "+",
        1,
        lambda a, b: (a + b) // 10 % 10,
        {
            "sum_tens": (lambda a, b: (a + b) // 10 % 10, 10),
            "a_tens": (lambda a, b: a // 10, 10),
            "b_tens": (lambda a, b: b // 10, 10),
            "carry": (carry, 2),
        },
    ),
    "mul_units": Task(
        products,
        "*",
        2,
        lambda a, b: a * b % 10,
        {
            "prod_units": (lambda a, b: a * b % 10, 10),
            "a_units": (lambda a, b: a % 10, 10),
            "b": (lambda a, b: b, 10),
            "a_tens": (lambda a, b: a // 10, 10),
        },
    ),
}


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


def activations(model, tok, task: Task, pairs, layer: int) -> mx.array:
    x, first = batch_ids(tok, pairs, task.op)
    pos = first - 1 + task.digit
    parts = [
        mlp_hidden(model, x[i : i + 250], layer, pos) for i in range(0, len(pairs), 250)
    ]
    return mx.concatenate(parts)


def digit_accuracy(model, tok, task: Task, pairs) -> float:
    x, first = batch_ids(tok, pairs, task.op)
    h, _, _ = run_blocks(model, x)
    lp = digit_logprobs(model, h, first - 1 + task.digit, tok)
    want = mx.array([task.target(a, b) for a, b in pairs])
    return (mx.argmax(lp, axis=-1) == want).mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--layer", type=int, default=21)
    p.add_argument("--task", choices=sorted(TASKS), default="add_units")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ks", default="4,16,64")
    a = p.parse_args()
    task = TASKS[a.task]

    model, tok = load(a.model)
    pairs = task.data(a.n, 2)
    act = activations(model, tok, task, pairs, a.layer)
    eta = {
        name: eta_squared(act, [f(x, y) for x, y in pairs], c)
        for name, (f, c) in task.labels.items()
    }
    main_label = next(iter(task.labels))
    order = mx.argsort(-eta[main_label]).tolist()
    print(f"Block {a.layer} MLP, task {a.task}, {act.shape[1]} neurons, {a.n} examples")
    for t in (0.3, 0.5, 0.7):
        counts = {k: int((v > t).sum().item()) for k, v in eta.items()}
        print(f"neurons with eta^2 > {t}: {counts}")
    names = list(task.labels)
    print(f"\ntop neurons for {main_label}:")
    print("neuron  " + "  ".join(f"{k:>10}" for k in names) + "  mean per digit 0-9")
    y = mx.array([task.target(x1, x2) for x1, x2 in pairs])
    top_rows = []
    for j in order[:12]:
        means = [class_mean(act[:, j], y, d) for d in range(10)]
        e = {k: eta[k][j].item() for k in names}
        cells = " ".join(f"{m:6.2f}" for m in means)
        print(f"{j:6d}  " + "  ".join(f"{e[k]:10.2f}" for k in names) + f"  {cells}")
        top_rows.append({"neuron": j, **e, "means": means})

    test = task.data(500, 3)
    s = Surgeon(model)
    base = digit_accuracy(model, tok, task, test)
    print(f"\ndigit accuracy on 500 new examples: base {base:.3f}")
    rng = random.Random(0)
    ablation = {"base": base}
    for k in [int(v) for v in a.ks.split(",")]:
        s.zero_neurons(a.layer, order[:k])
        top = digit_accuracy(model, tok, task, test)
        s.restore()
        s.zero_neurons(a.layer, rng.sample(range(act.shape[1]), k))
        rand = digit_accuracy(model, tok, task, test)
        s.restore()
        ablation[f"top{k}"], ablation[f"rand{k}"] = top, rand
        print(f"zero top {k:3d}: {top:.3f}   zero random {k:3d}: {rand:.3f}")

    name = f"tuning{a.layer}" + ("" if a.task == "add_units" else f"_{a.task}")
    (ROOT / "runs" / f"{name}.json").write_text(
        json.dumps(
            {
                "layer": a.layer,
                "task": a.task,
                "order": order[:256],
                "top": top_rows,
                "ablation": ablation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
