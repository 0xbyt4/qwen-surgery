"""Check edits on general text and on the math answers, without generation.

For each edit (see surgery.apply_spec) this gives:
  text_nll: mean negative log likelihood per token on data/stories.jsonl
            (SimpleStories test stories, plain text, no chat template);
  math20 and held-out: mean logprob of the right answer (as in evalmath.py).

A surgery that helps math but raises text_nll costs general language skill.

Example:
    uv run python textcheck.py --name text_neurons20 \
        --edits nfile:runs/neurons20.json:top16 nfile:runs/neurons20.json:bottom64
"""

import argparse
import json

import mlx.core as mx
from mlx_lm import load

from evalmath import DEFAULT_MODEL, ROOT, answer_logprob, default_qa
from mathgen import questions
from surgery import Surgeon, apply_spec


def text_nll(model, tok, texts: list[str], max_tokens: int = 512) -> float:
    total, count = 0.0, 0
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)[:max_tokens]
        logits = model(mx.array([ids]))[0, :-1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target = mx.array(ids[1:])[:, None]
        total -= mx.take_along_axis(lp, target, axis=-1).sum().item()
        count += len(ids) - 1
    return total / count


def mean_logprob(model, tok, qa: list[tuple[str, int]]) -> float:
    return sum(answer_logprob(model, tok, q, a) for q, a in qa) / len(qa)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--edits", nargs="+", required=True, help="one spec per edit")
    p.add_argument("--stories", type=int, default=200)
    p.add_argument("--name", required=True)
    a = p.parse_args()

    lines = (ROOT / "data" / "stories.jsonl").read_text().splitlines()
    texts = [json.loads(x)["text"] for x in lines[: a.stories]]
    model, tok = load(a.model)
    s = Surgeon(model)
    test, held = default_qa(), questions(40, seed=1)
    results = {}
    for spec in ["base", *a.edits]:
        if spec != "base":
            for part in spec.split("+"):
                apply_spec(s, part)
        r = {
            "text_nll": text_nll(model, tok, texts),
            "math20": mean_logprob(model, tok, test),
            "heldout": mean_logprob(model, tok, held),
        }
        s.restore()
        results[spec] = r
        print(
            f"{spec:>40}: text nll {r['text_nll']:.4f}, math20 lp "
            f"{r['math20']:7.3f}, held-out lp {r['heldout']:7.3f}",
            flush=True,
        )
    out = ROOT / "runs" / f"{a.name}.json"
    config = {"model": a.model, "stories": a.stories}
    out.write_text(json.dumps({"config": config, "results": results}, indent=2))


if __name__ == "__main__":
    main()
