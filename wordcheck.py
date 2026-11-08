"""Score edits on word problems with a remainder, and on rounding questions.

word2: the hard-set word problem (mathgen), 300 per seed. "Any extra kept
aside" asks for the whole part of the share. When the gift is larger than
the stock (A * B <= C) the set expects 0; those questions are ill-posed and
are counted apart.

Counter-sets, where rounding to the nearest whole number is right (the
fraction is 0.5 or more), 100 each:
  nearest_explicit: "What is X.YZ rounded to the nearest whole number?"
  nearest_division: "Divide A by B and round the result to the nearest ..."
  share_round:      a trip of A km in B days, km per day to the nearest km.

Example:
    uv run python wordcheck.py --seed 6 --edits "none;out:1495*1.1"
"""

import argparse
import json
import random

from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, run
from mathgen import HARD_KINDS, hard_questions
from surgery import Surgeon, apply_spec


def word2(n: int, seed: int) -> list[tuple[str, int]]:
    k = HARD_KINDS.index("word2")
    every = hard_questions(n * len(HARD_KINDS), seed)
    return [x for i, x in enumerate(every) if i % len(HARD_KINDS) == k]


def rounding_sets(n: int, seed: int) -> dict[str, list[tuple[str, int]]]:
    r = random.Random(seed)
    explicit, division, share = [], [], []
    while len(explicit) < n:
        x = r.randint(1, 99) + r.randint(50, 99) / 100
        explicit.append(
            (f"What is {x:.2f} rounded to the nearest whole number?", round(x))
        )
    while len(division) < n:
        a, b = r.randint(10, 200), r.randint(3, 9)
        if (a % b) / b > 0.5:
            division.append(
                (
                    f"Divide {a} by {b} and round the result to the nearest "
                    "whole number.",
                    round(a / b),
                )
            )
    while len(share) < n:
        a, b = r.randint(20, 300), r.randint(3, 9)
        if (a % b) / b > 0.5:
            share.append(
                (
                    f"A trip is {a} km long and takes {b} days. About how many km per "
                    "day is that, to the nearest km?",
                    round(a / b),
                )
            )
    return {
        "nearest_explicit": explicit,
        "nearest_division": division,
        "share_round": share,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--rounding", action="store_true", help="also the counter-sets")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--edits", required=True, help='";"-separated; "none" = base')
    a = p.parse_args()

    model, tok = load(a.model)
    s = Surgeon(model)
    qa = word2(a.n, a.seed)
    well = [i for i, (_, ans) in enumerate(qa) if ans != 0]
    extra = rounding_sets(100, 7) if a.rounding else {}
    results = {}
    for spec in a.edits.split(";"):
        if spec != "none":
            apply_spec(s, spec)
        items = run(model, tok, "direct", a.max_tokens, qa)
        row = {
            "well_posed": sum(items[i].correct for i in well),
            "n_well_posed": len(well),
            "all": sum(i.correct for i in items),
        }
        for name, qs in extra.items():
            row[name] = sum(i.correct for i in run(model, tok, "direct", 512, qs))
        s.restore()
        results[spec] = row
        print(f"{spec:32s} {row}", flush=True)
    out = ROOT / "runs" / f"wordcheck_s{a.seed}.json"
    out.write_text(json.dumps({"config": vars(a), "results": results}, indent=2))


if __name__ == "__main__":
    main()
