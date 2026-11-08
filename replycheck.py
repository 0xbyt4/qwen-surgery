"""Compare general replies of edits with the replies of a reference edit.

Greedy generation (thinking off) on general prompts, in a plain chat turn
without the math instruction of evalmath.prompt_ids. Many prompts have
hyphens, ranges, lists, negative numbers and subtraction, so that an edit of
a common token row shows up. Prints, per edit, how many replies are the same
as the reference, the mean shared start of the replies (in words), the mean
reply length, and the first differing replies.

Example:
    uv run python replycheck.py --ref "out:1249*1.25" \
        --edits "out:1249*1.25+out:481*1.1"
"""

import argparse
import json

from mlx_lm import batch_generate, load

from evalmath import DEFAULT_MODEL, ROOT
from surgery import Surgeon, apply_spec

PROMPTS = [
    "Give me three tips for a good night's sleep.",
    "Write a short poem about the sea.",
    "What is the capital of Australia?",
    "Explain what a black hole is in two sentences.",
    "List five fruits that are red.",
    "How do I calm down before an exam?",
    "Translate 'good morning' into French and Spanish.",
    "What is a well-known example of a self-driving car company?",
    "Write a to-do list for moving to a new flat.",
    "What does the term 'long-term memory' mean?",
    "Describe the water cycle step by step.",
    "What was the temperature range on Mars, roughly?",
    "Write a Python function that returns the difference of two numbers.",
    "What is -7 plus 3?",
    "What years did World War II last?",
    "Give a pros and cons list for working from home.",
    "Write the dates of the first week of March 2024 as a list.",
    "What is the opening hours format 9am-5pm called?",
    "Summarize the plot of Romeo and Juliet in three bullet points.",
    "How is a hyphen different from a dash?",
    "Name three state-of-the-art image models.",
    "What is 1000 minus 1?",
    "Write a short email to ask for a day off.",
    "Explain the difference between weather and climate.",
    "What are the steps to boil an egg?",
    "Give me a shopping list for a pancake breakfast.",
    "How many days are there between March 1 and March 15?",
    "What does a minus sign before a number mean?",
    "Write a two-line dialogue between a cat and a dog.",
    "What is the score format in tennis, for example 6-4?",
]


def chat_ids(tok, text: str) -> list[int]:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def shared_words(x: str, y: str) -> int:
    n = 0
    for a, b in zip(x.split(), y.split(), strict=False):
        if a != b:
            break
        n += 1
    return n


def replies(model, tok, max_tokens: int) -> list[str]:
    prompts = [chat_ids(tok, p) for p in PROMPTS]
    r = batch_generate(
        model,
        tok,
        prompts,
        max_tokens=max_tokens,
        completion_batch_size=len(prompts),
        prefill_batch_size=len(prompts),
    )
    return r.texts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--ref", default="none", help='reference edit; "none" = base')
    p.add_argument("--edits", required=True, help='";"-separated specs')
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--show", type=int, default=5, help="differing replies shown")
    p.add_argument("--name", default="replycheck")
    a = p.parse_args()

    model, tok = load(a.model)
    s = Surgeon(model)
    texts = {}
    for spec in [a.ref, *a.edits.split(";")]:
        if spec != "none":
            apply_spec(s, spec)
        texts[spec] = replies(model, tok, a.max_tokens)
        s.restore()
    ref = texts[a.ref]
    results = {}
    for spec in a.edits.split(";"):
        same = [x == y for x, y in zip(texts[spec], ref, strict=True)]
        words = sum(len(x.split()) for x in texts[spec]) / len(PROMPTS)
        shared = [shared_words(x, y) for x, y in zip(texts[spec], ref, strict=True)]
        results[spec] = {
            "same": sum(same),
            "of": len(PROMPTS),
            "shared_words": shared,
            "mean_words": words,
        }
        print(
            f"{spec:40s} same {sum(same)}/{len(PROMPTS)}  "
            f"shared start {sum(shared) / len(shared):.1f}  words {words:.1f}"
        )
        for i in [i for i, ok in enumerate(same) if not ok][: a.show]:
            k = shared[i]
            print(
                f"  [{PROMPTS[i]}] differs after {k} words\n"
                f"  ref:  {' '.join(ref[i].split()[max(k - 5, 0) : k + 15])!r}\n"
                f"  edit: {' '.join(texts[spec][i].split()[max(k - 5, 0) : k + 15])!r}"
            )
    ref_words = sum(len(x.split()) for x in ref) / len(PROMPTS)
    print(f"reference words {ref_words:.1f}")
    out = ROOT / "runs" / f"{a.name}.json"
    out.write_text(
        json.dumps({"config": vars(a), "results": results, "texts": texts}, indent=2)
    )


if __name__ == "__main__":
    main()
