"""Teacher-forced product sequences in many prompt forms.

For each pair (A, B) and prompt template, the base model replies greedily.
The reply is cut right after the first line "X × Y = " (or "X \\times Y = ",
"X * Y = ", "X x Y = ") whose operands are A and B in either order, and the
right product X * Y is appended as four digit tokens. Pairs whose reply has
no such line are left out. The operands are kept in the order of the reply
line, since the circuit reads them there.

Template sets: FIT_* for fitting readouts, HELD_* (the phrasings and word
problems of carrybuild --general) only for tests.
"""

import re
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm import batch_generate

from carrycollect import mlp_inputs
from evalmath import prompt_ids

WIDTH = 4
FIT_DIRECT = [
    "What is {a} * {b}?",
    "Compute {a} * {b}.",
    "What is {a} times {b}?",
    "Find {a} x {b}.",
    "How much is {a} multiplied by {b}?",
    "Work out {a} × {b}.",
    "Please multiply {a} and {b}.",
    "Give me {a} * {b}.",
]
FIT_WORD = [
    "A school has {a} classes with {b} students in each class. How many "
    "students are there?",
    "A shop sells {a} bags of rice each day. How many bags does it sell in {b} days?",
    "Each crate has {a} bottles. How many bottles are in {b} crates?",
    "Each bus carries {a} people. How many people do {b} buses carry?",
    "A baker makes {a} cookies every hour. How many cookies does she make in "
    "{b} hours?",
    "A garden has {a} rows of flowers with {b} flowers in each row. How many "
    "flowers are in the garden?",
    "Each jar holds {a} candies. How many candies are in {b} jars?",
    "A worker earns {a} dollars per day. How much does he earn in {b} days?",
    "There are {a} baskets and each basket contains {b} eggs. How many eggs "
    "are there in total?",
    "A parking garage has {a} levels with {b} cars on each level. How many "
    "cars are parked?",
    "Each pack holds {a} cards. How many cards are in {b} packs?",
    "A train has {a} cars and each car seats {b} passengers. How many "
    "passengers can the train seat?",
]
HELD_DIRECT = [
    "Calculate {a} × {b}.",
    "{a} times {b} = ?",
    "Multiply {a} by {b}.",
    "What is the product of {a} and {b}?",
]
HELD_WORD = [
    "A box holds {a} pencils. How many pencils are in {b} boxes?",
    "A theater has {a} rows with {b} seats in each row. How many seats are "
    "there in all?",
    "Tom reads {a} pages a day for {b} days. How many pages does he read?",
    "Each ticket costs {a} dollars. How much do {b} tickets cost?",
]
LINE = re.compile(r"(\d+)\s*(?:×|\\times|\*|x|\\cdot|·)\s*(\d+)\s*=\s*")


@dataclass
class Seqs:
    ids: list[list[int]]
    first: np.ndarray  # index of the first answer digit
    pairs: np.ndarray  # [N, 2], operands in the order of the reply line
    template: np.ndarray  # template index per sequence
    asked: np.ndarray  # [N, 2], the pair as asked

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, keep) -> "Seqs":
        keep = list(keep)
        return Seqs(
            [self.ids[i] for i in keep],
            self.first[keep],
            self.pairs[keep],
            self.template[keep],
            self.asked[keep],
        )


def collect(
    model,
    tok,
    pairs,
    templates: list[str],
    mode: str,
    max_tokens: int,
    batch: int,
    log: dict | None = None,
) -> Seqs:
    """Sequences for pairs, template i % len(templates) for pair i."""
    ids, first, line, tmpl, asked = [], [], [], [], []
    miss = [0] * len(templates)
    for s in range(0, len(pairs), batch):
        part = [tuple(map(int, p)) for p in pairs[s : s + batch]]
        prompts = [
            prompt_ids(tok, templates[(s + i) % len(templates)].format(a=a, b=b), mode)
            for i, (a, b) in enumerate(part)
        ]
        r = batch_generate(
            model,
            tok,
            prompts,
            max_tokens=max_tokens,
            completion_batch_size=len(prompts),
            prefill_batch_size=len(prompts),
        )
        for i, ((a, b), p, text) in enumerate(zip(part, prompts, r.texts, strict=True)):
            t = (s + i) % len(templates)
            m = next(
                (
                    m
                    for m in LINE.finditer(text)
                    if sorted((int(m[1]), int(m[2]))) == sorted((a, b))
                ),
                None,
            )
            if m is None:
                miss[t] += 1
                continue
            x, y = int(m[1]), int(m[2])
            reply = tok.encode(text[: m.end()] + str(x * y), add_special_tokens=False)
            if [tok.decode([k]) for k in reply[-WIDTH:]] != list(str(x * y)):
                miss[t] += 1
                continue
            ids.append(p + reply)
            first.append(len(p) + len(reply) - WIDTH)
            line.append((x, y))
            tmpl.append(t)
            asked.append((a, b))
    if log is not None:
        log.setdefault("missing", []).append(
            {"templates": templates, "mode": mode, "missing": miss}
        )
    return Seqs(ids, np.array(first), np.array(line), np.array(tmpl), np.array(asked))


def states(
    model, seqs: Seqs, block: int, positions, chunk: int = 64
) -> list[np.ndarray]:
    """MLP input of block at positions(i, seq) for each sequence, as a list
    of [k_i, d] arrays. Sequences run in right-padded batches of similar
    length; the padding comes after every position read, so it does not
    change them (causal attention)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs.ids[i]))
    got: dict[int, np.ndarray] = {}
    for s in range(0, len(order), chunk):
        idx = order[s : s + chunk]
        width = max(len(seqs.ids[i]) for i in idx)
        x = mx.array([seqs.ids[i] + [0] * (width - len(seqs.ids[i])) for i in idx])
        n = mlp_inputs(model, x, [block])[block]
        for row, i in enumerate(idx):
            pos = positions(i, seqs)
            got[i] = np.array(n[row, mx.array(pos)].astype(mx.float32))
    return [got[i] for i in range(len(seqs))]


def answer_states(model, seqs: Seqs, block: int) -> np.ndarray:
    """[N, 4, d] at the positions that predict the four answer digits."""
    got = states(
        model,
        seqs,
        block,
        lambda i, q: list(range(q.first[i] - 1, q.first[i] + WIDTH - 1)),
    )
    return np.stack(got)


def other_states(model, seqs: Seqs, block: int, skip=(1, 2)) -> np.ndarray:
    """[M, d] at every position except the ones that predict the answer
    digits in skip (1: hundreds, 2: tens)."""

    def pos(i, q):
        drop = {q.first[i] - 1 + k for k in skip}
        return [j for j in range(len(q.ids[i])) if j not in drop]

    return np.concatenate(states(model, seqs, block, pos))


def concat(parts: list[Seqs], offsets: list[int]) -> Seqs:
    """One Seqs from several; offsets shift the template index of each part."""
    return Seqs(
        [i for q in parts for i in q.ids],
        np.concatenate([q.first for q in parts]),
        np.concatenate([q.pairs for q in parts]),
        np.concatenate([q.template + o for q, o in zip(parts, offsets, strict=True)]),
        np.concatenate([q.asked for q in parts]),
    )


def answer_right(model, tok, seqs: Seqs, chunk: int = 16) -> np.ndarray:
    """[N, 4] booleans: is the top token at each answer position the right
    digit (teacher forced)."""
    digit = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    out = np.zeros((len(seqs), WIDTH), dtype=bool)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs.ids[i]))
    for s in range(0, len(order), chunk):
        idx = order[s : s + chunk]
        width = max(len(seqs.ids[i]) for i in idx)
        x = mx.array([seqs.ids[i] + [0] * (width - len(seqs.ids[i])) for i in idx])
        h = model.model(x)
        pos = mx.array(
            [[int(seqs.first[i]) - 1 + k for k in range(WIDTH)] for i in idx]
        )
        sel = mx.take_along_axis(h, pos[:, :, None], axis=1)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(sel)
        else:
            logits = model.lm_head(sel)
        top = np.array(mx.argmax(logits, axis=-1))
        for row, i in enumerate(idx):
            want = [digit[int(c)] for c in str(int(seqs.pairs[i].prod()))]
            out[i] = top[row] == np.array(want)
    return out
