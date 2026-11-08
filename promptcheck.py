"""Score models on the hard set with several prompts, per kind.

Does an output-row edit only undo the "Answer with only the final number."
request of the direct prompt? The same questions are asked in each mode of
evalmath.INSTRUCTION (thinking off for all but "think"): direct, plain (the
question alone), steps (ask for steps, then the number).

Example:
    uv run python promptcheck.py --seed 8 --modes direct plain steps \
        --models models/Qwen3-0.6B-bf16 models/Qwen3-0.6B-to125-down110-minus110
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mlx_lm import load

from evalmath import ROOT
from gsm8k import run_shared
from mathgen import HARD_KINDS, hard_questions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--modes", nargs="+", default=["direct", "plain", "steps"])
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--name", default="promptcheck")
    a = p.parse_args()

    qa = hard_questions(a.n, a.seed)
    results = {}
    for path in a.models:
        model, tok = load(path)
        for mode in a.modes:
            items = []
            for i in range(0, len(qa), a.batch):
                batch = qa[i : i + a.batch]
                items += run_shared(model, tok, mode, a.max_tokens, batch)
            kinds = Counter()
            for i, it in enumerate(items):
                kinds[HARD_KINDS[i % len(HARD_KINDS)]] += it.correct
            row = {
                "all": sum(i.correct for i in items),
                "kinds": dict(kinds),
                "cut": sum(i.finish == "length" for i in items),
                "mean_tokens": sum(i.generation_tokens for i in items) / len(items),
            }
            results[f"{Path(path).name} {mode}"] = row
            print(f"{Path(path).name:40s} {mode:7s} {row}", flush=True)
            out = ROOT / "runs" / f"{a.name}_s{a.seed}.json"
            out.write_text(
                json.dumps({"config": vars(a), "results": results}, indent=2)
            )


if __name__ == "__main__":
    main()
