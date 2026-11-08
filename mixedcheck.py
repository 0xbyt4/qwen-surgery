"""Score edits on expressions with a pending subtraction.

Sets (generation in direct mode, greedy, last number of the reply):
  mixed:   A * B + C - D * A (as in the hard set). After the two products,
           a reply often writes "P + C - Q = " in one line. A known error
           writes P + C there and stops: the "- Q" is dropped;
  three:   X + Y - Z, all three-digit, X + Y > Z;
  sub_big: five-digit minus four-digit (as in the hard set).

For mixed, wrong answers equal to P + C are counted as "dropped".

Example:
    uv run python mixedcheck.py --seed 12 --edits "none;out:481*1.1"
"""

import argparse
import json
import random

from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, run
from surgery import Surgeon, apply_spec


def mixed(n: int, seed: int) -> list[tuple[str, int, int]]:
    """(question, answer, P + C)."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = r.randint(12, 99), r.randint(3, 9)
        c, d = r.randint(100, 500), r.randint(2, 9)
        out.append(
            (f"What is {a} * {b} + {c} - {d} * {a}?", a * b + c - d * a, a * b + c)
        )
    return out


def three(n: int, seed: int) -> list[tuple[str, int]]:
    r = random.Random(seed)
    out = []
    while len(out) < n:
        x, y, z = (r.randint(100, 999) for _ in range(3))
        if x + y > z:
            out.append((f"What is {x} + {y} - {z}?", x + y - z))
    return out


def sub_big(n: int, seed: int) -> list[tuple[str, int]]:
    r = random.Random(seed)
    pairs = [(r.randint(10000, 99999), r.randint(1000, 9999)) for _ in range(n)]
    return [(f"What is {a} - {b}?", a - b) for a, b in pairs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--edits", required=True, help='";"-separated; "none" = base')
    a = p.parse_args()

    model, tok = load(a.model)
    s = Surgeon(model)
    mx_set = mixed(a.n, a.seed)
    others = {"three": three(a.n, a.seed), "sub_big": sub_big(a.n, a.seed)}
    results = {}
    for spec in a.edits.split(";"):
        if spec != "none":
            apply_spec(s, spec)
        items = run(model, tok, "direct", a.max_tokens, [(q, x) for q, x, _ in mx_set])
        row = {
            "mixed": sum(i.correct for i in items),
            "dropped": sum(
                not i.correct and i.predicted == pc
                for i, (_, _, pc) in zip(items, mx_set, strict=True)
            ),
        }
        for name, qa in others.items():
            row[name] = sum(i.correct for i in run(model, tok, "direct", 512, qa))
        s.restore()
        results[spec] = row
        print(f"{spec:44s} {row}", flush=True)
    out = ROOT / "runs" / f"mixedcheck_s{a.seed}.json"
    out.write_text(json.dumps({"config": vars(a), "results": results}, indent=2))


if __name__ == "__main__":
    main()
