"""Do the circuit's readouts hold on prompt templates they were not fit on?

The carry circuit (carrybuild.py) reads, at the MLP input of blocks 15 and
16, a context readout (1 at the positions that predict the hundreds and tens
digit of a product, 0 elsewhere) and the four operand digits. Fit on one
prompt template, the circuit failed on other templates (carrybuild
--general). Here the readouts are fit either on that one template or on
many (carrydata.FIT_*), and tested on held-out templates (carrydata.HELD_*):
  ctx:      held positives below the fit threshold, held negatives above it;
  operands: argmax accuracy, and per-value threshold flags as the circuit
            uses them (right value fires, wrong value fires).

Example:
    uv run python carrygate.py
"""

import argparse
import json

import numpy as np
from mlx_lm import load

import carrydata as cd
from carrybuild import ridge, story_inputs
from evalmath import DEFAULT_MODEL, ROOT
from probe import mul2x2

HUNDREDS, TENS = 1, 2


def digits(pairs: np.ndarray) -> dict[str, np.ndarray]:
    x, y = pairs[:, 0], pairs[:, 1]
    return {"a_t": x // 10, "a_u": x % 10, "b_t": y // 10, "b_u": y % 10}


def ctx_test(used, other, story, h_used, h_other) -> dict:
    allx = np.concatenate([used, other, story])
    target = np.r_[np.ones(len(used)), np.zeros(len(other) + len(story))]
    w = ridge(allx, target[:, None])[:, 0]
    w /= np.median(used @ w)
    neg, pos = np.r_[other @ w, story @ w], used @ w
    lo, hi = np.percentile(neg, 99.99), np.percentile(pos, 0.01)
    mid = (lo + hi) / 2
    hp, hn = h_used @ w, h_other @ w
    return {
        "fit_gap": float(hi - lo),
        "held_pos_below_mid": float((hp < mid).mean()),
        "held_neg_above_mid": float((hn > mid).mean()),
        "held_pos_p1": float(np.percentile(hp, 1)),
        "held_neg_p99_9": float(np.percentile(hn, 99.9)),
    }


def operand_test(used, y, h_used, hy) -> dict:
    out = {}
    for name in y:
        target = np.eye(10)[np.repeat(y[name], 2)]
        w = ridge(used, target)
        sc, yy = used @ w, np.repeat(y[name], 2)
        hs, hyy = h_used @ w, np.repeat(hy[name], 2)
        right_fire, wrong_fire = [], []
        for v in range(10):
            if not (yy == v).any() or not (hyy == v).any():
                continue
            lo, hi = sc[yy != v, v].max(), sc[yy == v, v].min()
            mid, half = (lo + hi) / 2, max(hi - lo, 0.02) / 2
            on = hs[:, v] > mid - half / 2
            right_fire.append(on[hyy == v].mean())
            wrong_fire.append(on[hyy != v].mean())
        out[name] = {
            "argmax_acc": float((hs.argmax(1) == hyy).mean()),
            "right_fires": float(np.mean(right_fire)),
            "wrong_fires": float(np.mean(wrong_fire)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--blocks", default="15,16")
    p.add_argument("--n-direct", type=int, default=2400)
    p.add_argument("--n-word", type=int, default=800)
    p.add_argument("--n-held", type=int, default=400)
    a = p.parse_args()

    model, tok = load(a.model)
    log: dict = {"config": vars(a)}
    fit_pairs = mul2x2(a.n_direct + a.n_word, 2)
    held_pairs = mul2x2(2 * a.n_held, 21, exclude=set(fit_pairs))
    fit_d = cd.collect(
        model, tok, fit_pairs[: a.n_direct], cd.FIT_DIRECT, "direct", 64, 100, log
    )
    fit_w = cd.collect(
        model, tok, fit_pairs[a.n_direct :], cd.FIT_WORD, "plain", 200, 50, log
    )
    held = {
        "direct": cd.collect(
            model, tok, held_pairs[: a.n_held], cd.HELD_DIRECT, "direct", 64, 100, log
        ),
        "word": cd.collect(
            model, tok, held_pairs[a.n_held :], cd.HELD_WORD, "plain", 200, 50, log
        ),
    }
    print("missing lines per template:", json.dumps(log["missing"]), flush=True)
    print(
        f"sequences: fit direct {len(fit_d)}, fit word {len(fit_w)}, held "
        f"direct {len(held['direct'])}, held word {len(held['word'])}",
        flush=True,
    )
    single = fit_d.subset(np.flatnonzero(fit_d.template == 0))
    lines = (ROOT / "data" / "stories.jsonl").read_text().splitlines()[:100]
    rng = np.random.default_rng(0)

    def few(seqs: cd.Seqs, n: int) -> cd.Seqs:
        return seqs.subset(rng.permutation(len(seqs))[:n])

    log["results"] = {}
    for block in [int(b) for b in a.blocks.split(",")]:
        story = story_inputs(model, tok, block, lines)
        h = {}
        for name, seqs in held.items():
            h[name] = (
                cd.answer_states(model, seqs, block)[:, [HUNDREDS, TENS]],
                cd.other_states(model, few(seqs, 150), block),
                digits(seqs.pairs),
            )
        fits = {}
        for name, parts in [("single", [single]), ("multi", [fit_d, fit_w])]:
            used = np.concatenate(
                [cd.answer_states(model, q, block)[:, [HUNDREDS, TENS]] for q in parts]
            )
            other = np.concatenate(
                [
                    cd.other_states(model, few(q, 300 // len(parts)), block)
                    for q in parts
                ]
            )
            pairs = np.concatenate([q.pairs for q in parts])
            fits[name] = (used.reshape(-1, used.shape[-1]), other, digits(pairs))
        for fname, (used, other, y) in fits.items():
            for hname, (hu, ho, hy) in h.items():
                hu2 = hu.reshape(-1, hu.shape[-1])
                r = {
                    "ctx": ctx_test(used, other, story, hu2, ho),
                    "operands": operand_test(used, y, hu2, hy),
                }
                log["results"][f"{block}/{fname}/{hname}"] = r
                ops = " ".join(
                    f"{k} {v['argmax_acc']:.3f}/{v['right_fires']:.2f}/"
                    f"{v['wrong_fires']:.3f}"
                    for k, v in r["operands"].items()
                )
                c = r["ctx"]
                print(
                    f"block {block} fit {fname:6s} held {hname:6s}: ctx gap "
                    f"{c['fit_gap']:.3f}, held pos below {c['held_pos_below_mid']:.3f},"
                    f" held neg above {c['held_neg_above_mid']:.5f} | operands "
                    f"acc/right fires/wrong fires: {ops}",
                    flush=True,
                )
    (ROOT / "runs" / "carrygate.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
