"""Search for a weight scale that makes two-digit products more accurate.

Score: on two-digit by two-digit products (probe.mul2x2) in the model's own
reply format, the share of examples whose four answer digits are all the top
digit token (teacher forcing), and the accuracy of each digit.

Edits tried, each alone:
  code: the block 25 one-hot digit neurons (top k of runs/tuning25_add_tens.json)
        scaled by F;
  attn: all heads of one block scaled by F;
  mlp:  all neurons of one block scaled by F.

The search runs on the dev set (seed 2). Pass --edit to score one edit (in
surgery.apply_spec form, or "attn:L*F" / "mlp:L*F") on the test set (seed 3,
without the dev products).

Example:
    uv run python mulfix.py --search
    uv run python mulfix.py --edit "mlp:24*1.25"
"""

import argparse
import json

import mlx.core as mx
import numpy as np
from mlx_lm import load

from circuit import digit_logprobs, run_blocks
from evalmath import DEFAULT_MODEL, ROOT
from probe import WIDTH, mul2x2, reply_ids
from surgery import Surgeon, apply_spec


def score(model, tok, pairs, chunk: int = 250) -> tuple[float, list[float]]:
    x, first = reply_ids(tok, pairs)
    want = np.array([[int(c) for c in str(a * b)] for a, b in pairs])
    tops = []
    for i in range(0, len(pairs), chunk):
        h, _, _ = run_blocks(model, x[i : i + chunk])
        tops.append(
            np.stack(
                [
                    np.array(
                        mx.argmax(digit_logprobs(model, h, first - 1 + k, tok), -1)
                    )
                    for k in range(WIDTH)
                ],
                axis=1,
            )
        )
    ok = np.concatenate(tops) == want
    return float(ok.all(1).mean()), [float(v) for v in ok.mean(0)]


def apply(s: Surgeon, spec: str) -> None:
    """apply_spec plus attn:L*F and mlp:L*F (scale a whole part)."""
    for part in spec.split("+"):
        body, _, factor = part.partition("*")
        kind, *rest = body.split(":")
        if kind in ("attn", "mlp") and factor:
            layer = int(rest[0])
            if kind == "attn":
                s.scale_heads(layer, list(range(s.n_heads)), float(factor))
            else:
                s.scale_neurons(layer, list(range(s.n_neurons)), float(factor))
        else:
            apply_spec(s, part)


def code_spec(k: int, factor: float) -> str:
    order = json.loads((ROOT / "runs" / "tuning25_add_tens.json").read_text())["order"]
    return f"neurons:25:{','.join(map(str, order[:k]))}*{factor}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--search", action="store_true")
    p.add_argument("--edit")
    p.add_argument("--layers", default="16-27")
    p.add_argument("--n", type=int, default=1000)
    a = p.parse_args()

    model, tok = load(a.model)
    s = Surgeon(model)
    if a.edit:
        test = mul2x2(a.n, 3, exclude=set(mul2x2(a.n, 2)))
        base = score(model, tok, test)
        apply(s, a.edit)
        edited = score(model, tok, test)
        print(f"test set: base {base}  edit {a.edit}: {edited}")
        return

    dev = mul2x2(a.n, 2)
    base = score(model, tok, dev)
    print(f"dev base: all {base[0]:.3f} digits {base[1]}", flush=True)
    specs = [code_spec(k, f) for k in (16, 64) for f in (1.25, 1.5, 2.0)]
    first, last = (int(v) for v in a.layers.split("-"))
    specs += [
        f"{part}:{layer}*{f}"
        for layer in range(first, last + 1)
        for part in ("attn", "mlp")
        for f in (0.8, 1.25)
    ]
    rows = []
    for spec in specs:
        apply(s, spec)
        all_ok, per = score(model, tok, dev)
        s.restore()
        label = (
            spec
            if not spec.startswith("neurons:25")
            else (f"code{len(spec.split(':')[2].split(','))}*{spec.split('*')[1]}")
        )
        rows.append({"edit": label, "spec": spec, "all": all_ok, "digits": per})
        print(
            f"{label:16s} all {all_ok:.3f} ({all_ok - base[0]:+.3f})  "
            + " ".join(f"{v:.3f}" for v in per),
            flush=True,
        )
    (ROOT / "runs" / "mulfix_search.json").write_text(
        json.dumps({"base": base, "rows": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
