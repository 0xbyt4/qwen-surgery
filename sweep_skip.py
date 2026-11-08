"""Remove one transformer block at a time and run the math test.

Writes runs/<name>.json with the summary of each run (and the base run) and
prints one line per removed layer.

Example:
    uv run python sweep_skip.py --name sweep_skip_direct
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, run, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--mode", choices=["direct", "think"], default="direct")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--name", required=True)
    a = p.parse_args()

    model, tok = load(a.model)
    layers = list(model.model.layers)
    results = {}
    for skip in [None, *range(len(layers))]:
        model.model.layers = [x for i, x in enumerate(layers) if i != skip]
        mx.reset_peak_memory()
        items = run(model, tok, a.mode, a.max_tokens)
        s = summary(items)
        s["wrong"] = [i + 1 for i, x in enumerate(items) if not x.correct]
        key = "base" if skip is None else str(skip)
        results[key] = s
        print(
            f"skip {key:>4}: {s['correct']:2d}/20, logprob {s['mean_logprob']:7.3f}, "
            f"{s['generation_tokens']:5d} tok, {s['generation_tps']:5.1f} tok/s, "
            f"wrong {s['wrong']}",
            flush=True,
        )
    model.model.layers = layers
    out = ROOT / "runs" / f"{a.name}.json"
    out.parent.mkdir(exist_ok=True)
    config = {"model": a.model, "mode": a.mode, "max_tokens": a.max_tokens}
    out.write_text(json.dumps({"config": config, "results": results}, indent=2))


if __name__ == "__main__":
    main()
