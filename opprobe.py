"""Can new neurons read the operand digits where the answer is written?

Task: A * B with A and B two-digit (probe.mul2x2), teacher forced in the
model's reply format "A × B = P". At each answer position (the positions
that write the thousands, hundreds, tens and units digit) and after each
block, a ridge probe maps the residual to:
  a_t, a_u, b_t, b_u: the four operand digits;
  carry_h:            the carry into the hundreds digit,
                      (a_t*b_u + a_u*b_t + (a_u*b_u) // 10) // 10.
Fit on one product set, tested on another without shared products.

If the operand digits are read near 100% at a late block, the carry is a
fixed function of readable inputs, and new MLP neurons with written (not
trained) weights can compute it.

Example:
    uv run python opprobe.py
"""

import argparse
import json

import numpy as np
from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT
from probe import mul2x2, predict, states

POSITIONS = ["thousands", "hundreds", "tens", "units"]
CLASSES = 20  # the carry goes up to 17


def ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """probe.ridge with CLASSES one-hot targets."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
    target = np.eye(CLASSES)[y]
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ target)
    return w, np.stack([mu, sd])


def targets(pairs: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    a = np.array(pairs)
    a_t, a_u, b_t, b_u = a[:, 0] // 10, a[:, 0] % 10, a[:, 1] // 10, a[:, 1] % 10
    carry = (a_t * b_u + a_u * b_t + (a_u * b_u) // 10) // 10
    return {"a_t": a_t, "a_u": a_u, "b_t": b_t, "b_u": b_u, "carry_h": carry}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--n-fit", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--lam", type=float, default=100.0)
    p.add_argument("--blocks", default="15,18,20,21,22,23,24,25,26,27")
    a = p.parse_args()

    model, tok = load(a.model)
    fit = mul2x2(a.n_fit, 2)
    test = mul2x2(a.n_test, 3, exclude=set(fit))
    fit_r, _ = states(model, tok, fit)
    test_r, _ = states(model, tok, test)
    y_fit, y_test = targets(fit), targets(test)
    blocks = [int(b) for b in a.blocks.split(",")]

    results: dict[str, dict[str, list[float]]] = {}
    for k, pos in enumerate(POSITIONS):
        results[pos] = {}
        print(f"\nposition writing the {pos} digit; columns: blocks {blocks}")
        for name in y_fit:
            accs = []
            for b in blocks:
                w, norm = ridge(fit_r[b][:, k], y_fit[name], a.lam)
                accs.append(
                    float((predict(test_r[b][:, k], w, norm) == y_test[name]).mean())
                )
            results[pos][name] = accs
            print(f"  {name:8s} " + " ".join(f"{v:.3f}" for v in accs), flush=True)
    out = ROOT / "runs" / "opprobe_mul2x2.json"
    out.write_text(json.dumps({"config": vars(a), "results": results}, indent=2))


if __name__ == "__main__":
    main()
