"""Run the 20-question math test (math20.py) on a model, with optional surgery.

Modes:
  direct: thinking off, the model is asked for only the number (up to 256 tokens;
          word problems still get a short worked answer).
  think:  thinking on, the model reasons first (up to --max-tokens tokens).

Each run also gives the log probability of the correct answer under the
direct prompt (teacher forced). This number changes smoothly, so it shows
small effects of a surgery that the 20 right/wrong answers cannot show.

Speed is logged per question: prompt and generation tokens per second, the
number of generated tokens, and the peak memory.

Example:
    uv run python evalmath.py --mode direct --name base_direct
    uv run python evalmath.py --mode think --name base_think
    uv run python evalmath.py --skip-layers 3 --name skip3_direct
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

from math20 import ANSWERS, QUESTIONS

ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3-0.6B-bf16"
INSTRUCTION = {
    "direct": "Answer with only the final number.",
    "think": "Give the final answer as a number on the last line.",
}
NUMBER = re.compile(r"-?\d[\d,]*")
# The direct replies look like "47 + 38 = 85"; score the number after a fixed lead.
ANSWER_PREFIX = "The answer is "


@dataclass
class Item:
    question: str
    answer: int
    predicted: int | None
    correct: bool
    logprob: float  # log p(correct answer | direct prompt)
    output: str
    finish: str | None
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    seconds: float


def prompt_ids(tok: TokenizerWrapper, question: str, mode: str) -> list[int]:
    messages = [{"role": "user", "content": f"{question}\n{INSTRUCTION[mode]}"}]
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=mode == "think"
    )


def parse(text: str, mode: str) -> int | None:
    """The last number of the reply (after </think> in think mode)."""
    if mode == "think":
        if "</think>" not in text:
            return None
        text = text.split("</think>", 1)[1]
    found = NUMBER.findall(text)
    return int(found[-1].replace(",", "")) if found else None


def scoring_ids(tok: TokenizerWrapper, question: str, answer: int) -> tuple[list, int]:
    """Direct prompt + ANSWER_PREFIX + answer digits, and the prompt length."""
    ids = prompt_ids(tok, question, "direct")
    head = tok.encode(ANSWER_PREFIX, add_special_tokens=False)
    full = tok.encode(ANSWER_PREFIX + str(answer), add_special_tokens=False)
    if full[: len(head)] != head:
        raise ValueError(f"prefix tokens change before {answer}")
    return ids + full, len(ids) + len(head)


def sequence_logprob(model, ids: list[int], n_prompt: int) -> mx.array:
    """Sum of log p(token) over ids[n_prompt:], teacher forced."""
    logits = model(mx.array([ids]))[0, n_prompt - 1 : -1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target = mx.array(ids[n_prompt:])[:, None]
    return mx.take_along_axis(lp, target, axis=-1).sum()


def answer_logprob(model, tok: TokenizerWrapper, question: str, answer: int) -> float:
    """log p(answer digits | direct prompt + ANSWER_PREFIX), teacher forced."""
    return sequence_logprob(model, *scoring_ids(tok, question, answer)).item()


def default_qa() -> list[tuple[str, int]]:
    return [(q, a) for (q, _), a in zip(QUESTIONS, ANSWERS, strict=True)]


def run(
    model,
    tok: TokenizerWrapper,
    mode: str,
    max_tokens: int,
    qa: list[tuple[str, int]] | None = None,
) -> list[Item]:
    items = []
    for question, answer in qa or default_qa():
        start = time.perf_counter()
        text, last = "", None
        for r in stream_generate(
            model, tok, prompt_ids(tok, question, mode), max_tokens=max_tokens
        ):
            text += r.text
            last = r
        seconds = time.perf_counter() - start
        predicted = parse(text, mode)
        items.append(
            Item(
                question=question,
                answer=answer,
                predicted=predicted,
                correct=predicted == answer,
                logprob=answer_logprob(model, tok, question, answer),
                output=text,
                finish=last.finish_reason,
                prompt_tokens=last.prompt_tokens,
                generation_tokens=last.generation_tokens,
                prompt_tps=last.prompt_tps,
                generation_tps=last.generation_tps,
                seconds=seconds,
            )
        )
    return items


def summary(items: list[Item]) -> dict:
    gen = sum(x.generation_tokens for x in items)
    gen_time = sum(x.generation_tokens / x.generation_tps for x in items)
    return {
        "correct": sum(x.correct for x in items),
        "total": len(items),
        "mean_logprob": sum(x.logprob for x in items) / len(items),
        "generation_tokens": gen,
        "generation_tps": gen / gen_time,
        "mean_prompt_tps": sum(x.prompt_tps for x in items) / len(items),
        "seconds": sum(x.seconds for x in items),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }


def skip_layers(model, skip: list[int]) -> None:
    """Remove whole transformer blocks (the residual stream passes unchanged)."""
    model.model.layers = [x for i, x in enumerate(model.model.layers) if i not in skip]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--mode", choices=["direct", "think"], default="direct")
    p.add_argument("--max-tokens", type=int, default=0, help="0: default for the mode")
    p.add_argument("--skip-layers", default="", help="comma list of layer indices")
    p.add_argument("--name", required=True)
    a = p.parse_args()
    max_tokens = a.max_tokens or (256 if a.mode == "direct" else 2048)
    skip = [int(x) for x in a.skip_layers.split(",") if x]

    model, tok = load(a.model)
    skip_layers(model, skip)
    mx.reset_peak_memory()
    items = run(model, tok, a.mode, max_tokens)
    s = summary(items)
    out = ROOT / "runs" / f"{a.name}.json"
    out.parent.mkdir(exist_ok=True)
    config = {"model": a.model, "mode": a.mode, "max_tokens": max_tokens, "skip": skip}
    out.write_text(
        json.dumps(
            {"config": config, "summary": s, "items": [asdict(x) for x in items]},
            indent=2,
        )
    )
    for i, x in enumerate(items):
        mark = "ok " if x.correct else "BAD"
        print(
            f"{i + 1:2d} {mark} want {x.answer:5d} got {x.predicted!s:>6} "
            f"lp {x.logprob:7.2f} gen {x.generation_tokens:4d} tok "
            f"{x.generation_tps:6.1f} tok/s"
        )
    print(
        f"{a.name}: {s['correct']}/{s['total']} correct, mean logprob "
        f"{s['mean_logprob']:.3f}, {s['generation_tokens']} tokens at "
        f"{s['generation_tps']:.1f} tok/s, prompt {s['mean_prompt_tps']:.0f} tok/s, "
        f"{s['seconds']:.1f} s, peak {s['peak_memory_gb']:.2f} GB"
    )


if __name__ == "__main__":
    main()
