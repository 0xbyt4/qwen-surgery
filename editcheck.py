"""Measure edits on the hard math set, the easy sets and general text.

For each edit (see surgery.apply_spec):
  hard:     generated answers (thinking off, batch) on mathgen.hard_questions,
            correct per kind, and the mean answer logprob;
  math20, held-out: mean answer logprob (one batched forward pass);
  text_nll: loss per token on data/stories.jsonl.

Choose edits on one seed of the hard set (--seed, the "dev" set) and confirm
the chosen ones on another seed.

Example:
    uv run python editcheck.py --name scale_dev --seed 2 \
        --edits "head:20:12,13,14,15*1.5" "head:20:12,13,14,15*2"
"""

import argparse
import json
from collections import Counter

from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, answer_logprobs, default_qa, run
from mathgen import HARD_KINDS, hard_questions, questions
from surgery import Surgeon, apply_spec
from textcheck import text_nll


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--edits", nargs="*", default=[], help="one spec per edit")
    p.add_argument("--seed", type=int, default=2, help="seed of the hard set")
    p.add_argument("--n", type=int, default=120, help="hard questions")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--stories", type=int, default=200)
    p.add_argument("--name", required=True)
    a = p.parse_args()

    lines = (ROOT / "data" / "stories.jsonl").read_text().splitlines()
    texts = [json.loads(x)["text"] for x in lines[: a.stories]]
    model, tok = load(a.model)
    s = Surgeon(model)
    hard = hard_questions(a.n, a.seed)
    easy, held = default_qa(), questions(40, seed=1)
    results = {}
    for spec in ["base", *a.edits]:
        if spec != "base":
            apply_spec(s, spec)
        items = run(model, tok, "direct", a.max_tokens, hard)
        per_kind = Counter()
        for i, x in enumerate(items):
            per_kind[HARD_KINDS[i % len(HARD_KINDS)]] += x.correct
        r = {
            "hard_correct": sum(x.correct for x in items),
            "hard_total": len(items),
            "hard_per_kind": dict(per_kind),
            "hard_logprob": mean([x.logprob for x in items]),
            "hard_cut": sum(x.finish == "length" for x in items),
            "math20": mean(answer_logprobs(model, tok, easy)),
            "heldout": mean(answer_logprobs(model, tok, held)),
            "text_nll": text_nll(model, tok, texts),
        }
        s.restore()
        results[spec] = r
        kinds = " ".join(f"{k} {per_kind[k]}" for k in HARD_KINDS)
        print(
            f"{spec:>34}: hard {r['hard_correct']:3d}/{len(items)} "
            f"lp {r['hard_logprob']:6.2f} cut {r['hard_cut']} [{kinds}] | "
            f"math20 {r['math20']:6.3f} held {r['heldout']:6.3f} "
            f"text {r['text_nll']:.4f}",
            flush=True,
        )
    out = ROOT / "runs" / f"{a.name}.json"
    config = {"model": a.model, "seed": a.seed, "n": a.n, "max_tokens": a.max_tokens}
    out.write_text(json.dumps({"config": config, "results": results}, indent=2))


if __name__ == "__main__":
    main()
