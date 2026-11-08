"""Score head edits on two skills at once: sums of four numbers and products.

Generation in direct mode (greedy), counted as right when the last number
of the reply is the answer; "cut" counts replies that hit --max-tokens:
  add4:   A + B + C + D, each three-digit;
  mul2x2: A * B, both two-digit, four-digit product;
  mul3x2: A * B, A three-digit, B two-digit (as in the hard set).

Example:
    uv run python tradeoff.py --seed 2 --edits "head:20:12*1.5;head:20:13*1.5"
"""

import argparse
import json
import random

from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, run
from probe import mul2x2
from surgery import Surgeon, apply_spec


def add4(n: int, seed: int) -> list[tuple[str, int]]:
    r = random.Random(seed)
    out = []
    for _ in range(n):
        xs = [r.randint(100, 999) for _ in range(4)]
        out.append((f"What is {' + '.join(map(str, xs))}?", sum(xs)))
    return out


def mul3x2(n: int, seed: int) -> list[tuple[str, int]]:
    r = random.Random(seed)
    pairs = [(r.randint(101, 999), r.randint(11, 99)) for _ in range(n)]
    return [(f"What is {a} * {b}?", a * b) for a, b in pairs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n-add", type=int, default=200)
    p.add_argument("--n-mul", type=int, default=300)
    p.add_argument("--n-mul3", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--edits", required=True, help='";"-separated; "none" = base')
    p.add_argument("--name", default="tradeoff")
    a = p.parse_args()

    model, tok = load(a.model)
    s = Surgeon(model)
    sets = {
        "add4": add4(a.n_add, a.seed),
        "mul2x2": [(f"What is {x} * {y}?", x * y) for x, y in mul2x2(a.n_mul, a.seed)],
    }
    if a.n_mul3:
        sets["mul3x2"] = mul3x2(a.n_mul3, a.seed)
    results = {}
    print(f"seed {a.seed}: " + ", ".join(f"{k} of {len(v)}" for k, v in sets.items()))
    for spec in a.edits.split(";"):
        if spec != "none":
            apply_spec(s, spec)
        row, cut = {}, {}
        for k, qa in sets.items():
            items = run(model, tok, "direct", a.max_tokens, qa)
            row[k] = sum(i.correct for i in items)
            cut[k] = sum(i.finish == "length" for i in items)
        s.restore()
        results[spec] = {**row, "cut": cut}
        cells = "  ".join(f"{k} {v:4d} (cut {cut[k]})" for k, v in row.items())
        print(f"{spec:40s} {cells}", flush=True)
    out = ROOT / "runs" / f"{a.name}_s{a.seed}.json"
    out.write_text(json.dumps({"config": vars(a), "results": results}, indent=2))


if __name__ == "__main__":
    main()
