"""Score models on the GSM8K test set (1319 grade-school word problems).

Data: data/gsm8k/test.jsonl from github.com/openai/grade-school-math (the
answer is the number after "####"). Generation in direct mode (thinking off,
greedy, evalmath.run), counted right when the last number of the reply is the
answer. Questions whose answer is not a whole number are left out.

Example:
    uv run python gsm8k.py --models models/Qwen3-0.6B-bf16 \
        models/Qwen3-0.6B-to125-down110-minus110
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from evalmath import ROOT, run

MAX_TRIES = 10
DATA = ROOT / "data" / "gsm8k" / "test.jsonl"


def gsm8k(limit: int | None = None) -> list[tuple[str, int]]:
    out = []
    for line in DATA.read_text().splitlines():
        row = json.loads(line)
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        if answer.lstrip("-").isdigit():
            out.append((row["question"], int(answer)))
    return out[:limit]


def run_shared(model, tok, max_tokens: int, qa: list[tuple[str, int]]) -> list:
    """evalmath.run, again after a pause when other GPU jobs use the memory."""
    for _ in range(MAX_TRIES - 1):
        try:
            return run(model, tok, "direct", max_tokens, qa)
        except RuntimeError as e:
            if "Insufficient Memory" not in str(e):
                raise
            print("GPU out of memory; waiting", flush=True)
            mx.clear_cache()
            time.sleep(60)
    return run(model, tok, "direct", max_tokens, qa)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--name", default="gsm8k")
    a = p.parse_args()

    qa = gsm8k(a.limit)
    print(f"{len(qa)} questions", flush=True)
    results = {}
    for path in a.models:
        model, tok = load(path)
        items = []
        for i in range(0, len(qa), a.batch):
            items += run_shared(model, tok, a.max_tokens, qa[i : i + a.batch])
        right = sum(i.correct for i in items)
        cut = sum(i.finish == "length" for i in items)
        tokens = sum(i.generation_tokens for i in items) / len(items)
        name = Path(path).name
        results[name] = {
            "right": right,
            "of": len(items),
            "cut": cut,
            "mean_tokens": tokens,
            "items": [asdict(i) for i in items],
        }
        print(
            f"{name:40s} {right}/{len(items)} ({right / len(items):.3f})  "
            f"cut {cut}  tokens {tokens:.1f}",
            flush=True,
        )
        out = ROOT / "runs" / f"{a.name}.json"
        out.write_text(json.dumps({"config": vars(a), "results": results}, indent=2))


if __name__ == "__main__":
    main()
