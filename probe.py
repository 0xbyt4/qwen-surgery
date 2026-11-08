"""Does the model hold the right digit inside when it writes a wrong one?

Task: A * B with A and B two-digit and a four-digit product, in the model's
own reply format "A × B = P" after the direct prompt (teacher forcing: the
digits before each scored digit are the right ones).

For each answer digit and each block, a ridge-regression probe maps the
residual after the block (at the position that writes the digit) to the
one-hot digit. It is fit on one set of products and tested on another, with
no shared product. The script prints the probe accuracy, the model's own
accuracy (top digit token), and the probe accuracy on the examples where the
model is wrong.

Example:
    uv run python probe.py
"""

import argparse
import json
import random

import mlx.core as mx
import numpy as np
from mlx_lm import load

from circuit import digit_logprobs, run_blocks
from evalmath import DEFAULT_MODEL, ROOT, prompt_ids

WIDTH = 4


def mul2x2(
    n: int, seed: int, exclude: set[tuple[int, int]] | None = None
) -> list[tuple[int, int]]:
    """n pairs with a four-digit product. With exclude (for a test set), the
    pairs are unique and none of them is in exclude (for example the fit or
    dev pairs); without it, pairs can repeat, as in the earlier runs."""
    r = random.Random(seed)
    out: list[tuple[int, int]] = []
    seen = set(exclude) if exclude is not None else None
    while len(out) < n:
        a, b = r.randint(10, 99), r.randint(10, 99)
        if a * b < 1000:
            continue
        if seen is not None:
            if (a, b) in seen:
                continue
            seen.add((a, b))
        out.append((a, b))
    return out


def reply_ids(
    tok, pairs: list[tuple[int, int]], answers: list[int] | None = None
) -> tuple[mx.array, int]:
    """Prompt + "A × B = P" token ids [N, L] and the position of the first
    answer digit. P is A * B, or the given answers (to fix the forced digits)."""
    rows = []
    for i, (a, b) in enumerate(pairs):
        ans = a * b if answers is None else answers[i]
        reply = tok.encode(f"{a} × {b} = {ans}", add_special_tokens=False)
        tail = [tok.decode([t]) for t in reply[-WIDTH:]]
        if tail != list(str(ans)):
            raise ValueError(f"{ans} is not {WIDTH} digit tokens: {tail}")
        rows.append(prompt_ids(tok, f"What is {a} * {b}?", "direct") + reply)
    if len({len(x) for x in rows}) != 1:
        raise ValueError("sequences differ in length")
    return mx.array(rows), len(rows[0]) - WIDTH


def digits(pairs: list[tuple[int, int]], k: int) -> np.ndarray:
    return np.array([int(str(a * b)[k]) for a, b in pairs])


def states(model, tok, pairs, chunk: int = 250):
    """Residual after each block at the answer positions [blocks][N, WIDTH, d]
    and the model's top digit [N, WIDTH]."""
    x, first = reply_ids(tok, pairs)
    pos = mx.arange(first - 1, first - 1 + WIDTH)
    resid, top = [], []
    for i in range(0, len(pairs), chunk):
        h, r, _ = run_blocks(model, x[i : i + chunk], keep=True)
        resid.append([np.array(v[:, pos].astype(mx.float32)) for v in r])
        top.append(
            np.stack(
                [
                    np.array(mx.argmax(digit_logprobs(model, h, int(p), tok), -1))
                    for p in pos
                ],
                axis=1,
            )
        )
    n_blocks = len(resid[0])
    return [
        np.concatenate([c[b] for c in resid]) for b in range(n_blocks)
    ], np.concatenate(top)


def ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    z = np.hstack([z, np.ones((len(z), 1))])
    target = np.eye(10)[y]
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ target)
    return w, np.stack([mu, sd])


def predict(x: np.ndarray, w: np.ndarray, norm: np.ndarray) -> np.ndarray:
    z = (x - norm[0]) / norm[1]
    z = np.hstack([z, np.ones((len(z), 1))])
    return (z @ w).argmax(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--n-fit", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--lam", type=float, default=100.0)
    a = p.parse_args()

    model, tok = load(a.model)
    fit = mul2x2(a.n_fit, 2)
    test = mul2x2(a.n_test, 3, exclude=set(fit))
    fit_r, _ = states(model, tok, fit)
    test_r, top = states(model, tok, test)

    results = {}
    for k in range(WIDTH):
        y_fit, y_test = digits(fit, k), digits(test, k)
        own = top[:, k] == y_test
        print(
            f"\ndigit {k} ({['thousands', 'hundreds', 'tens', 'units'][k]}): "
            f"model {own.mean():.3f}, wrong in {int((~own).sum())} of {len(test)}"
        )
        print("block  probe  probe on model-wrong  probe agrees with model")
        rows = []
        for b in range(len(fit_r)):
            w, norm = ridge(fit_r[b][:, k], y_fit, a.lam)
            guess = predict(test_r[b][:, k], w, norm)
            acc = (guess == y_test).mean()
            on_wrong = (guess == y_test)[~own].mean() if (~own).any() else float("nan")
            agree = (guess == top[:, k]).mean()
            rows.append(
                {"block": b, "probe": acc, "on_wrong": on_wrong, "agree": agree}
            )
            print(f"{b:5d}  {acc:5.3f}  {on_wrong:20.3f}  {agree:23.3f}")
        results[k] = {"model": float(own.mean()), "blocks": rows}
    (ROOT / "runs" / "probe_mul2x2.json").write_text(
        json.dumps(results, indent=2, default=float)
    )


if __name__ == "__main__":
    main()
