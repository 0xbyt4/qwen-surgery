"""Zero the attention or the MLP of one block, or one attention head, and run
the math test (direct mode) after each change.

Parts:
  blocks: for each block, attention off, then MLP off (56 runs).
  heads:  for each block in --layers, each of the 16 heads off.

Writes runs/<name>.json and prints one line per run.

Example:
    uv run python sweep_parts.py --part blocks --name sweep_blocks
    uv run python sweep_parts.py --part heads --layers 17,18,20,21 --name sweep_heads
"""

import argparse
import json
from collections.abc import Callable

import mlx.core as mx
from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, run, summary
from surgery import Surgeon


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--part", choices=["blocks", "heads"], required=True)
    p.add_argument("--layers", default="", help="comma list (heads only)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--name", required=True)
    a = p.parse_args()

    model, tok = load(a.model)
    s = Surgeon(model)
    n_layers = len(model.model.layers)
    edits: list[tuple[str, Callable[[], None]]] = [("base", lambda: None)]
    if a.part == "blocks":
        for i in range(n_layers):
            edits.append((f"{i}.attn", lambda i=i: s.zero_attention(i)))
            edits.append((f"{i}.mlp", lambda i=i: s.zero_mlp(i)))
    else:
        for i in [int(x) for x in a.layers.split(",")]:
            for h in range(s.n_heads):
                edits.append((f"{i}.h{h}", lambda i=i, h=h: s.zero_heads(i, [h])))

    results = {}
    for key, edit in edits:
        edit()
        mx.reset_peak_memory()
        items = run(model, tok, "direct", a.max_tokens)
        s.restore()
        r = summary(items)
        r["wrong"] = [i + 1 for i, x in enumerate(items) if not x.correct]
        results[key] = r
        print(
            f"{key:>8}: {r['correct']:2d}/20, logprob {r['mean_logprob']:7.3f}, "
            f"{r['generation_tokens']:5d} tok, wrong {r['wrong']}",
            flush=True,
        )
    out = ROOT / "runs" / f"{a.name}.json"
    out.parent.mkdir(exist_ok=True)
    config = {"model": a.model, "part": a.part, "layers": a.layers}
    out.write_text(json.dumps({"config": config, "results": results}, indent=2))


if __name__ == "__main__":
    main()
