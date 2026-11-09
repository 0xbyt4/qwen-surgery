"""Add hand-written neurons that compute the hundreds and tens digit of A * B.

No training. The model already holds the four operand digits at the answer
positions (opprobe.py: read 99-100% after blocks 15 to 20; carrygate.py: on
prompt templates outside the fit only from block 20 on), but not the
carry. New MLP neurons compute the digits with the school method, and every
value between the steps is a one-hot flag (0 or 1) in its own new residual
direction, so errors do not add up from step to step:

  A = 10 a_t + a_u, B = 10 b_t + b_u
  P1 = a_t b_u, P2 = a_u b_t, U = (a_u b_u) // 10, X = a_t b_t
  S2 = P1 % 10 + P2 % 10 + U        tens digit t = S2 % 10, k = S2 // 10
  K = P1 // 10 + P2 // 10           hundreds digit h = (X % 10 + K + k) % 10

Stages (each is new neurons in the MLP of one block; default blocks):
  C (block 19): a context flag (1 only at the hundreds and tens answer
            positions).
  B0 (block 20): flags of a_t, a_u, b_t, b_u (40 flags), times the context
            flag. A linear
            readout of the model state always ranks the right digit first,
            but one threshold for all values does not separate it; one
            threshold per value does.
  B1 (block 21): one AND neuron pair per digit pair (a_t = i, b_u = j)
            writes the flags of P1 % 10 and P1 // 10, and so on for P2, U
            and X % 10.
  B2 (block 22): AND over (P1 % 10, P2 % 10, U) writes the flags of t and k;
            AND over (P1 // 10, P2 // 10) writes the flags of K.
  B3 (block 23): AND over (X % 10, K, k) writes the flag of h.
  relay (--relay, off by default): reads t and h and writes them again,
            fresh, into new directions (a flag fades over several blocks).
  B4 (block 24): t at the tens position and h at the hundreds position add
            the logit direction of that digit.
Each stage erases the flags it read for the last time. Flags sit in
residual directions that the later blocks read and write least.

A neuron is SwiGLU: silu(gate . n) * (up . n) * down, with n the normed MLP
input. A clamp to [0, 1] uses two neurons: (silu(beta z) - silu(beta z - S))
/ S. An AND of m flags is a clamp on (sum of flags) - (m - 0.5).

The only fitted parts are linear readouts (closed-form ridge, no gradient)
of what the model or the new neurons hold: the operand digits, the flags, a
constant, the context (product answer position against story text) and the
answer position. Fits use teacher-forced product sequences in many prompt
templates (carrydata.FIT_*, pairs of seed 2; --one-template: only "What is
A * B?") and stories 0-99. Scores use new products (seed 3, "What is A *
B?"), held-out templates (carrydata.HELD_*, seed 21) and stories 100-199.

Example:
    uv run python carrybuild.py --gen 1000
"""

import argparse
import itertools
import json
import math
from collections.abc import Callable

import mlx.core as mx
import numpy as np
from mlx_lm import batch_generate, load

import carrydata as cd
from carrycollect import mlp_inputs
from circuit import run_blocks
from evalmath import DEFAULT_MODEL, ROOT, prompt_ids, run
from gsm8k import gsm8k
from mathgen import hard_questions
from mixedcheck import sub_big, three
from mulfix import score
from probe import WIDTH, mul2x2, reply_ids
from replycheck import PROMPTS, chat_ids, replies
from surgery import Surgeon
from textcheck import text_nll

HUNDREDS, TENS = 1, 2  # answer positions (0 = thousands digit)
BETA, S = 50.0, 20.0  # a clamp is 0 for z <= -0.05 and 1 for z >= S / BETA
OPERANDS = ["a_t", "a_u", "b_t", "b_u"]
USED_WEIGHT = 10.0  # weight of the hundreds and tens states in flag readouts
SIZES = {"P1u": 10, "P1t": 9, "P2u": 10, "P2t": 9, "U": 9, "Xu": 10}
SIZES |= {"t": 10, "k": 3, "K": 17, "h": 10, "t2": 10, "h2": 10, "ctx": 1}


def variables(pairs: np.ndarray) -> dict[str, np.ndarray]:
    a, b = pairs[:, 0], pairs[:, 1]
    v = {"a_t": a // 10, "a_u": a % 10, "b_t": b // 10, "b_u": b % 10}
    p1, p2, x = v["a_t"] * v["b_u"], v["a_u"] * v["b_t"], v["a_t"] * v["b_t"]
    v |= {"P1u": p1 % 10, "P1t": p1 // 10, "P2u": p2 % 10, "P2t": p2 // 10}
    v |= {"U": (v["a_u"] * v["b_u"]) // 10, "Xu": x % 10}
    s2 = v["P1u"] + v["P2u"] + v["U"]
    v |= {"t": s2 % 10, "k": s2 // 10, "K": v["P1t"] + v["P2t"]}
    v["h"] = (v["Xu"] + v["K"] + v["k"]) % 10
    if not ((v["t"] == a * b // 10 % 10).all() and (v["h"] == a * b // 100 % 10).all()):
        raise ValueError("digit formulas are wrong")
    return v


def ridge(
    x: np.ndarray, y: np.ndarray, lam: float = 1.0, weight: np.ndarray | None = None
) -> np.ndarray:
    """Closed-form ridge readout; weight gives each row a weight."""
    x = x.astype(np.float64)
    xw = x if weight is None else x * weight[:, None]
    return np.linalg.solve(xw.T @ x + lam * np.eye(x.shape[1]), xw.T @ y)


def story_inputs(model, tok, block: int, lines: list[str], max_tokens=256):
    out = []
    for line in lines:
        ids = tok.encode(json.loads(line)["text"])[:max_tokens]
        n = mlp_inputs(model, mx.array([ids]), [block])[block]
        out.append(np.array(n[0].astype(mx.float32)))
    return np.concatenate(out)


def seq_inputs(model, block: int, seqs: list[list[int]]) -> np.ndarray:
    """MLP input of block at every position of token sequences: [M, d]."""
    out = []
    for ids in seqs:
        n = mlp_inputs(model, mx.array([ids]), [block])[block]
        out.append(np.array(n[0].astype(mx.float32)))
    return np.concatenate(out)


CHAT = [
    "What is the largest ocean on Earth?",
    "Give me two ideas for a rainy weekend.",
    "Who wrote Pride and Prejudice?",
    "Explain photosynthesis in one paragraph.",
    "What is the boiling point of water in Fahrenheit?",
    "List four countries in South America.",
    "How do I make a cup of green tea?",
    "Write a haiku about autumn leaves.",
    "What year did the first person walk on the Moon?",
    "Describe a cat to someone who has never seen one.",
    "What is the difference between a virus and a bacterium?",
    "Suggest a name for a small bakery.",
    "How many legs does a spider have?",
    "What is the capital of Canada?",
    "Write a short thank-you note to a teacher.",
    "Why is the sky blue?",
    "Give me a 3-step plan to learn to swim.",
    "What are the primary colors?",
    "Translate 'thank you' into German and Italian.",
    "What does a 24-hour clock show at 3 pm?",
    "Name five animals that live in the Arctic.",
    "How far is the Moon from the Earth, roughly?",
    "Write a limerick about a dog.",
    "What is the tallest mountain in Africa?",
    "Explain what a prime number is.",
    "What is 12 minus 5?",
    "Give a short recipe for scrambled eggs.",
    "Who painted the Mona Lisa?",
    "What is the population of Tokyo, roughly?",
    "List the days of the week in order.",
]


def negative_seqs(model, tok, max_tokens: int = 256) -> list[list[int]]:
    """Prompt + base reply token ids of chat prompts (CHAT, not the
    replycheck prompts) and of other math tasks, as states where the new
    neurons must stay silent. Seeds differ from the side-task seeds."""
    seqs = []
    sets = [
        ("plain", hard_questions(150, 9)),
        ("direct", three(60, 13)),
        ("direct", sub_big(60, 13)),
    ]
    chat = [chat_ids(tok, q) for q in CHAT]
    r = batch_generate(
        model,
        tok,
        chat,
        max_tokens=max_tokens,
        completion_batch_size=len(chat),
        prefill_batch_size=len(chat),
    )
    for ids, text in zip(chat, r.texts, strict=True):
        seqs.append(ids + tok.encode(text, add_special_tokens=False))
    for mode, qa in sets:
        for i in range(0, len(qa), 50):
            for it, (q, _) in zip(
                run(model, tok, mode, max_tokens, qa[i : i + 50]),
                qa[i : i + 50],
                strict=True,
            ):
                reply = tok.encode(it.output, add_special_tokens=False)
                seqs.append((prompt_ids(tok, q, mode) + reply)[: 2 * max_tokens])
    return seqs


class Neurons:
    """Rows of new neurons for one block, built in numpy."""

    def __init__(self) -> None:
        self.gate: list[np.ndarray] = []
        self.up: list[np.ndarray] = []
        self.down: list[np.ndarray] = []

    def clamp(self, z, up, out, const) -> None:
        """Adds out * up(n) * clamp(z(n)); z, up and const (a readout of 1)
        are readout rows [d]."""
        self.gate += [BETA * z, BETA * z - S * const]
        self.up += [up, up]
        self.down += [out / S, -out / S]

    def insert(self, s: Surgeon, layer: int) -> int:
        s.add_neurons(
            layer,
            mx.array(np.stack(self.gate), dtype=mx.float32),
            mx.array(np.stack(self.up), dtype=mx.float32),
            mx.array(np.stack(self.down, axis=1), dtype=mx.float32),
        )
        return len(self.gate)


class Stage:
    """Readouts at the MLP input of one block. Value readouts are fit on the
    hundreds and tens positions of the fit products. The context readout
    ("ctx": 1 at those positions, 0 at all other positions of the product
    sequences, of the stories and of other math replies), the position
    readouts and a constant are fit on all of these states."""

    def __init__(
        self,
        model,
        tok,
        seqs: cd.Seqs,
        block,
        story_lines,
        y,
        neg,
        n_other=400,
        held: dict | None = None,
    ) -> None:
        n = cd.answer_states(model, seqs, block)
        spread = np.linspace(0, len(seqs) - 1, n_other).astype(int)
        other = cd.other_states(model, seqs.subset(spread), block)
        st = np.concatenate(
            [
                story_inputs(model, tok, block, story_lines),
                seq_inputs(model, block, neg),
            ]
        )
        self.used = n[:, [HUNDREDS, TENS]].reshape(-1, n.shape[-1])
        self.all = np.concatenate([self.used, other, st])
        is_ctx = np.r_[np.ones(len(self.used)), np.zeros(len(other) + len(st))]
        self.is_ctx = is_ctx
        self.const = ridge(self.all, np.ones((len(self.all), 1)))[:, 0]
        self.ctx = ridge(self.all, is_ctx[:, None])[:, 0]
        self.ctx /= np.median(self.used @ self.ctx)
        is_h = np.r_[np.tile([1.0, 0.0], n.shape[0]), np.zeros(len(other) + len(st))]
        self.pos = {
            HUNDREDS: ridge(self.all, is_h[:, None])[:, 0],
            TENS: ridge(self.all, (is_ctx - is_h)[:, None])[:, 0],
        }
        self.y = y
        self.held: dict[str, tuple[np.ndarray, dict]] = {}
        for k, q in (held or {}).items():
            hs = cd.answer_states(model, q, block)[:, [HUNDREDS, TENS]]
            self.held[k] = (hs.reshape(-1, hs.shape[-1]), variables(q.pairs))
        g = model.model.layers[block].post_attention_layernorm.weight
        self.g = np.array(g.astype(mx.float32))

    def onehot(
        self, name: str, size: int, everywhere: bool = True
    ) -> tuple[np.ndarray, float]:
        """Readout [d, size] of a variable as one-hot flags, and its largest
        error on the fit states. With everywhere, the readout is also fit to
        0 at all other states (for flags the new neurons write, which exist
        only at the hundreds and tens positions); without it, only the
        hundreds and tens positions are used (for the model's own state)."""
        target = np.eye(size)[np.repeat(self.y[name], 2)]
        if not everywhere:
            w = ridge(self.used, target)
            return w, float(np.abs(self.used @ w - target).max())
        full = np.zeros((len(self.all), size))
        full[: len(target)] = target
        weight = np.r_[
            np.full(len(target), USED_WEIGHT), np.ones(len(self.all) - len(target))
        ]
        w = ridge(self.all, full, weight=weight)
        # scale each value so that its median on the right states is 1
        sc, yy = self.used @ w, np.repeat(self.y[name], 2)
        for v in range(size):
            if (yy == v).any():
                w[:, v] /= np.median(sc[yy == v, v])
        return w, float(np.abs(self.all @ w - full).max())

    def and_gate(self, flags: list[np.ndarray]) -> np.ndarray:
        """Gate of an AND over flag readout rows (in clamp units, where the
        clamp is 1 above S / BETA). With z = sum - (m - 0.5): 0.5 when all
        flags are on, -0.5 when one is off."""
        z = sum(flags) - (len(flags) - 0.5) * self.const
        # clamp units: 1 for z >= 0.1 (sum >= m - 0.4), 0 for z <= -0.2
        return 1.5 * z + 0.25 * self.const


def quiet_directions(
    model, first: int, n: np.ndarray, g: np.ndarray, k: int
) -> np.ndarray:
    """k unit residual directions that the blocks from first on (and the
    output head) read and write least, and along which the model state
    varies little.
    A direction w is read by a matrix W after a norm with gain g as
    W (g * w); each matrix counts with weight 1 / |W|^2."""
    f32 = lambda x: np.array(x.astype(mx.float32), dtype=np.float64)  # noqa: E731
    read = np.zeros((n.shape[1], n.shape[1]))
    blocks = model.model.layers
    for blk in blocks[first:]:
        att, mlp = blk.self_attn, blk.mlp
        for norm, mats in [
            (blk.input_layernorm, [att.q_proj, att.k_proj, att.v_proj]),
            (blk.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj]),
        ]:
            gn = f32(norm.weight)
            for lin in mats:
                w = f32(lin.weight) * gn[None, :]
                read += w.T @ w / (w * w).sum()
        for lin in [att.o_proj, mlp.down_proj]:  # what the block writes
            w = f32(lin.weight)
            read += w @ w.T / (w * w).sum()
    gn = f32(model.model.norm.weight)
    head = f32(model.model.embed_tokens.weight) * gn[None, :]
    read += head.T @ head / (head * head).sum()
    cov = np.cov((n / g).T)
    m = read / np.trace(read) + cov / np.trace(cov)
    _, vec = np.linalg.eigh(m)
    w = vec[:, :k].T
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def reply_fit(model, tok, seqs: list[tuple[int, list[int]]]):
    """Teacher forced on fixed chat replies (prompt length, ids): the top-1
    token at each reply position and the mean NLL of the reply tokens."""
    top, nll = [], []
    for k, ids in seqs:
        logits = model(mx.array([ids]))[0, k - 1 : -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array(ids[k:])
        top.append(np.array(mx.argmax(logits, axis=-1)))
        nll.append(np.array(-mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]))
    return np.concatenate(top), float(np.concatenate(nll).mean())


def final_rms(model, tok, pairs) -> float:
    """rms of the last residual (before the final norm) at the answer positions."""
    x, first = reply_ids(tok, [tuple(p) for p in pairs])
    _, resid, _ = run_blocks(model, x, keep=True)
    h = np.array(resid[-1][:, first - 1 : first - 1 + WIDTH].astype(mx.float32))
    return float(np.sqrt((h**2).mean()))


SEPARATORS = [" ", ",", ".", "\n", "\n\n", ".  ", " **", "**"]


def digit_push(model, tok, margin: float) -> np.ndarray:
    """[10, d] residual vectors (least norm) that raise the logit of one digit
    by margin and leave the logits of the other digits and of separator
    tokens (space, comma, period, newline, bold) unchanged, through the
    final norm and the output head, for a residual of rms 1 (multiply by
    the real rms)."""
    ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    ids += sorted(
        {tok.encode(x, add_special_tokens=False)[0] for x in SEPARATORS} - set(ids)
    )
    e = np.array(model.model.embed_tokens.weight[mx.array(ids)].astype(mx.float32))
    g = np.array(model.model.norm.weight.astype(mx.float32))
    a = (g * e).astype(np.float64)  # logit change = a @ delta
    target = np.zeros((len(ids), 10))
    target[np.arange(10), np.arange(10)] = margin
    return (a.T @ np.linalg.solve(a @ a.T, target)).T


def lookup(
    nn: Neurons,
    st: Stage,
    inputs: list[tuple[np.ndarray, int]],
    write: Callable[..., np.ndarray],
    up: np.ndarray,
) -> None:
    """One AND neuron pair per combination of input values; write(*values)
    gives the output vector of that combination. A single input is ANDed
    with the context flag; with two or more inputs this is not needed, since
    the flag readouts are fit to 0 outside the context."""
    ctx = [st.ctx] if len(inputs) == 1 else []
    for values in itertools.product(*[range(size) for _, size in inputs]):
        flags = [r[:, v] for (r, _), v in zip(inputs, values, strict=True)]
        nn.clamp(st.and_gate([*ctx, *flags]), up, write(*values), st.const)


def readouts(st: Stage, names: list[str], log: dict, tag: str) -> dict:
    out = {}
    stats = {}
    for name in names:
        size = SIZES.get(name, 10)
        out[name], err = st.onehot(name, size)
        log["readout_max_err"][f"{tag} {name}"] = err
        sc, yy = st.used @ out[name], np.repeat(st.y[name], 2)
        right = sc[np.arange(len(sc)), yy]
        wrong = np.where(np.eye(size)[yy] == 1, -9.0, sc).max(1)
        rest = st.all[len(st.used) :] @ out[name]
        stats[name] = (
            round(float(np.percentile(right, 0.5)), 2),
            round(float(np.percentile(wrong, 99.5)), 2),
            round(float(np.abs(rest).max()), 2),
        )
    ctx_in = st.used @ st.ctx
    ctx_out = st.all[len(st.used) :] @ st.ctx
    stats["ctx"] = (
        round(float(np.percentile(ctx_in, 0.5)), 2),
        round(float(np.percentile(ctx_out, 99.9)), 2),
    )
    for k, (hs, hy) in st.held.items():
        acc = {
            n: round(float(((hs @ out[n]).argmax(1) == np.repeat(hy[n], 2)).mean()), 3)
            for n in names
        }
        stats[f"held {k} argmax"] = acc
    log.setdefault("readout_stats", {})[tag] = stats
    print(
        f"{tag} in-context right p0.5 / wrong p99.5 / |other| max:", stats, flush=True
    )
    print(
        f"{tag} readout max err",
        {n: round(log["readout_max_err"][f"{tag} {n}"], 3) for n in names},
        flush=True,
    )
    return out


def erase(nn: Neurons, st: Stage, flags: np.ndarray, out: np.ndarray) -> None:
    """Remove flags after their last read: a clamp on each flag adds -out[v]."""
    for v in range(flags.shape[1]):
        nn.clamp(st.and_gate([st.ctx, flags[:, v]]), st.const, -out[v], st.const)


def debug_neurons(model, tok, nn: Neurons, block: int, lines: list[str]) -> None:
    """Output size of new neurons at story positions, in float32 numpy and in
    bf16 on the GPU (before they are inserted)."""
    n = story_inputs(model, tok, block, lines)
    gate, up, down = np.stack(nn.gate), np.stack(nn.up), np.stack(nn.down, axis=1)
    g = n @ gate.T
    act = g / (1 + np.exp(-np.clip(g, -60, 60))) * (n @ up.T)
    f32 = np.linalg.norm(act @ down.T, axis=1)
    bf = lambda x: mx.array(x).astype(mx.bfloat16)  # noqa: E731
    nb = bf(n)
    gb, ub = nb @ bf(gate).T, nb @ bf(up).T
    ob = (nn_silu(gb) * ub) @ bf(down).T
    b16 = np.linalg.norm(np.array(ob.astype(mx.float32)), axis=1)
    print(
        f"debug block {block}: |gate| max {np.abs(g).max():.1f}; "
        f"output norm f32 p50/p99/max {np.percentile(f32, 50):.3f}/"
        f"{np.percentile(f32, 99):.3f}/{f32.max():.3f}; bf16 p50/p99/max "
        f"{np.percentile(b16, 50):.3f}/{np.percentile(b16, 99):.3f}/{b16.max():.3f}; "
        f"input norm {np.linalg.norm(n, axis=1).mean():.1f}",
        flush=True,
    )


def nn_silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def generate(model, tok, qa: list[tuple[str, int]], batch: int = 100) -> list:
    """Greedy replies in the direct mode (evalmath.run), in batches."""
    out = []
    for i in range(0, len(qa), batch):
        out += run(model, tok, "direct", 64, qa[i : i + batch])
    return out


def side_sets(model, tok) -> dict[str, list]:
    """Right or wrong per question on other tasks, and chat replies: the hard
    set (plain prompt, seed 8), X + Y - Z and five-digit minus four-digit
    (direct prompt, seed 12), and the replycheck prompts."""
    out: dict[str, list] = {}
    hard = hard_questions(600, 8)
    items = []
    for i in range(0, len(hard), 50):
        items += run(model, tok, "plain", 768, hard[i : i + 50])
    out["hard_plain"] = [int(i.correct) for i in items]
    for name, qa in [("three", three(300, 12)), ("sub_big", sub_big(300, 12))]:
        out[name] = [int(i.correct) for i in run(model, tok, "direct", 512, qa)]
    out["reply"] = replies(model, tok, 256)
    return out


PHRASINGS = [
    "Calculate {a} × {b}.",
    "{a} times {b} = ?",
    "Multiply {a} by {b}.",
    "What is the product of {a} and {b}?",
]
WORD = [
    "A box holds {a} pencils. How many pencils are in {b} boxes?",
    "A theater has {a} rows with {b} seats in each row. How many seats are "
    "there in all?",
    "Tom reads {a} pages a day for {b} days. How many pages does he read?",
    "Each ticket costs {a} dollars. How much do {b} tickets cost?",
]


def general_sets(seen: set) -> dict[str, tuple[list, str, int, int]]:
    """Tasks the circuit was not built for, as (qa, mode, max tokens, batch):
    new products with the plain prompt, other phrasings (direct prompt),
    word problems (plain prompt), and GSM8K (plain prompt, first 400)."""

    def pairs(n: int, seed: int) -> list[tuple[int, int]]:
        out = mul2x2(n, seed, exclude=seen)
        seen.update(out)
        return out

    def fill(forms: list[str], ps) -> list[tuple[str, int]]:
        return [
            (forms[i % len(forms)].format(a=x, b=z), x * z)
            for i, (x, z) in enumerate(ps)
        ]

    return {
        "plain": (
            [(f"What is {x} * {z}?", x * z) for x, z in pairs(400, 6)],
            "plain",
            384,
            50,
        ),
        "phrasings": (fill(PHRASINGS, pairs(400, 7)), "direct", 64, 100),
        "word": (fill(WORD, pairs(400, 9)), "plain", 384, 50),
        "gsm8k": (gsm8k(400), "plain", 768, 25),
    }


def run_set(model, tok, spec) -> list:
    qa, mode, max_tokens, batch = spec
    out = []
    for i in range(0, len(qa), batch):
        out += run(model, tok, mode, max_tokens, qa[i : i + batch])
    return out


def sign_p(gained: int, lost: int) -> float:
    """Two-sided sign test (exact McNemar) on the discordant pairs."""
    n, k = gained + lost, min(gained, lost)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--blocks", default="20,21,22,23,24")
    p.add_argument("--alpha", type=float, default=32.0, help="flag size")
    p.add_argument("--margin", type=float, default=15.0)
    p.add_argument("--n-fit", type=int, default=2400, help="direct fit pairs")
    p.add_argument("--n-word", type=int, default=2400, help="word fit pairs")
    p.add_argument("--n-held", type=int, default=400)
    p.add_argument("--one-template", action="store_true", help="old fit set")
    p.add_argument("--trace", action="store_true", help="readouts on held templates")
    p.add_argument("--n-test", type=int, default=1500)
    p.add_argument("--name", default="carrybuild")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--gen", type=int, default=0, help="new products to generate")
    p.add_argument("--gen-seed", type=int, default=4)
    p.add_argument("--side", action="store_true", help="score other tasks too")
    p.add_argument(
        "--general", action="store_true", help="score other forms of the task"
    )
    p.add_argument("--relay", type=int, default=-1, help="relay block; -1: none")
    p.add_argument("--ctx-block", type=int, default=19)
    a = p.parse_args()

    b0, b1, b2, b3, b4 = (int(x) for x in a.blocks.split(","))
    model, tok = load(a.model)
    s = Surgeon(model)
    fit = np.array(mul2x2(a.n_fit + a.n_word, 2))
    test = np.array(mul2x2(a.n_test, 3, exclude={tuple(x) for x in fit}))
    held_pairs = mul2x2(2 * a.n_held, 21, exclude={tuple(x) for x in fit})
    data_log: dict = {}
    if a.one_template:
        seqs = cd.collect(
            model, tok, fit[: a.n_fit], cd.FIT_DIRECT[:1], "direct", 64, 100, data_log
        )
    else:
        seqs = cd.concat(
            [
                cd.collect(
                    model,
                    tok,
                    fit[: a.n_fit],
                    cd.FIT_DIRECT,
                    "direct",
                    64,
                    100,
                    data_log,
                ),
                cd.collect(
                    model, tok, fit[a.n_fit :], cd.FIT_WORD, "plain", 200, 50, data_log
                ),
            ],
            [0, len(cd.FIT_DIRECT)],
        )
    held = {
        "direct": cd.collect(
            model,
            tok,
            held_pairs[: a.n_held],
            cd.HELD_DIRECT,
            "direct",
            64,
            100,
            data_log,
        ),
        "word": cd.collect(
            model,
            tok,
            held_pairs[a.n_held :],
            cd.HELD_WORD,
            "plain",
            200,
            50,
            data_log,
        ),
    }
    print(
        f"fit sequences {len(seqs)} (templates {np.bincount(seqs.template)}), "
        f"held {[len(q) for q in held.values()]}; missing {data_log['missing']}",
        flush=True,
    )
    y = variables(seqs.pairs)
    lines = (ROOT / "data" / "stories.jsonl").read_text().splitlines()
    fit_lines, test_lines = lines[:100], lines[100:200]
    texts = [json.loads(x)["text"] for x in test_lines]
    log: dict = {"config": vars(a), "readout_max_err": {}, "stages": {}}

    base_replies: list[str] = []
    base_seqs: list[tuple[int, list[int]]] = []
    base_fit: list = []

    def check(tag: str) -> None:
        """Score, story loss and chat replies (same as the base model, of the
        replycheck prompts, 96 tokens) with the neurons inserted so far."""
        all_, digits = score(model, tok, [tuple(x) for x in test])
        nll = text_nll(model, tok, texts[:50])
        rep = replies(model, tok, 96)
        if not base_replies:
            base_replies.extend(rep)
            for q, r in zip(PROMPTS, rep, strict=True):
                k = len(chat_ids(tok, q))
                base_seqs.append((k, chat_ids(tok, q) + tok.encode(r)))
            base_fit.extend(reply_fit(model, tok, base_seqs))
        same = sum(x == z for x, z in zip(rep, base_replies, strict=True))
        top, rnll = reply_fit(model, tok, base_seqs)
        agree = float((top == base_fit[0]).mean())
        held_right = {k: cd.answer_right(model, tok, q) for k, q in held.items()}
        held_all = {k: round(float(r.all(1).mean()), 3) for k, r in held_right.items()}
        held_digits = {
            k: [round(float(x), 3) for x in r.mean(0)] for k, r in held_right.items()
        }
        held_digits["by template"] = {
            k: [
                round(float(r[held[k].template == j].all(1).mean()), 2)
                for j in range(held[k].template.max() + 1)
            ]
            for k, r in held_right.items()
        }
        log["stages"][tag] = {
            "all": all_,
            "digits": digits,
            "story_nll_50": nll,
            "reply_same": same,
            "reply_top1_agree": agree,
            "reply_nll": rnll,
            "held_all": held_all,
            "held_digits": held_digits,
        }
        print(
            f"  after {tag}: all {all_:.3f} digits "
            f"{[round(d, 3) for d in digits]} story nll (50) {nll:.4f} "
            f"replies same {same}/{len(rep)}, teacher-forced top-1 agree "
            f"{agree:.4f} reply nll {rnll:.4f} (base {base_fit[1]:.4f}); held "
            f"templates all four digits {held_all}, per digit {held_digits}",
            flush=True,
        )

    check("none")
    neg = negative_seqs(model, tok)
    print(f"negative sequences: {len(neg)}, tokens {sum(map(len, neg))}", flush=True)

    # ---- C (block before B0): the context flag ----------------------------
    st = Stage(
        model, tok, seqs, a.ctx_block, fit_lines, y, neg, held=held if a.trace else None
    )
    quiet = quiet_directions(
        model, a.ctx_block + 1, st.all, st.g, 40 + sum(SIZES.values())
    )
    dirs, at = {}, 0
    for name in [*OPERANDS, *SIZES]:
        size = SIZES.get(name, 10)
        dirs[name] = a.alpha * quiet[at : at + size]
        at += size
    gaps = {}
    sc = st.all @ st.ctx
    lo = float(np.percentile(sc[st.is_ctx == 0], 99.99))
    hi = float(np.percentile(sc[st.is_ctx == 1], 0.01))
    gaps["ctx"] = hi - lo
    mid, half = (lo + hi) / 2, max(hi - lo, 0.02) / 2
    z = (st.ctx - (mid - half / 2) * st.const) * (S / BETA) / half
    nn = Neurons()
    nn.clamp(z, st.const, dirs["ctx"][0], st.const)
    nn.insert(s, a.ctx_block)
    print(f"C block {a.ctx_block}: ctx gap {gaps['ctx']:.3f}", flush=True)
    check("C")

    # ---- B0: operand flags, times the context flag ------------------------
    st = Stage(model, tok, seqs, b0, fit_lines, y, neg, held=held if a.trace else None)
    nn = Neurons()
    for name in OPERANDS:
        r, _ = st.onehot(name, 10, everywhere=False)
        sc, yy = st.used @ r, np.repeat(y[name], 2)
        for i in range(10):
            if not (yy == i).any():
                continue
            lo, hi = sc[yy != i, i].max(), sc[yy == i, i].min()
            gaps[f"{name}={i}"] = float(hi - lo)
            mid, half = (lo + hi) / 2, max(hi - lo, 0.02) / 2
            # 0 below mid - half / 2, 1 above mid + half / 2 (fit states)
            z = (r[:, i] - (mid - half / 2) * st.const) * (S / BETA) / half
            nn.clamp(z, st.ctx, dirs[name][i], st.const)  # up: the context flag
    log["flag_gaps_fit"] = gaps
    k0 = nn.insert(s, b0)
    smallest = sorted(gaps.items(), key=lambda x: x[1])[:4]
    print(f"B0 block {b0}: {k0} neurons; smallest fit gaps {smallest}", flush=True)
    check("B0")

    def one(name: str, v: int) -> np.ndarray:
        return dirs[name][v]

    # ---- B1: digit products ----------------------------------------------
    st = Stage(model, tok, seqs, b1, fit_lines, y, neg, held=held if a.trace else None)
    fl = readouts(st, OPERANDS, log, "B1")
    nn = Neurons()
    families = [
        ("a_t", "b_u", lambda i, j: one("P1u", i * j % 10) + one("P1t", i * j // 10)),
        ("a_u", "b_t", lambda i, j: one("P2u", i * j % 10) + one("P2t", i * j // 10)),
        ("a_u", "b_u", lambda i, j: one("U", i * j // 10)),
        ("a_t", "b_t", lambda i, j: one("Xu", i * j % 10)),
    ]
    for f1, f2, write in families:
        lookup(nn, st, [(fl[f1], 10), (fl[f2], 10)], write, st.const)
    for name in OPERANDS:
        erase(nn, st, fl[name], dirs[name])
    if a.debug:
        g = np.stack(nn.gate)
        u = np.stack(nn.up)
        silu = lambda x: x / (1 + np.exp(-np.clip(x, -60, 60)))  # noqa: E731
        x = st.used
        clampv = (
            (silu(x @ g[0:200:2].T) - silu(x @ g[1:200:2].T)) / S * (x @ u[0:200:2].T)
        )
        yy = np.repeat(y["a_t"] * 10 + y["b_u"], 2)
        right = clampv[np.arange(len(x)), yy]
        wrong = np.where(np.eye(100)[yy] == 1, -9.0, clampv).max(1)
        print(
            f"debug B1 family a_t x b_u at hundreds/tens: right clamp p1/p50 "
            f"{np.percentile(right, 1):.2f}/{np.median(right):.2f}; wrong max "
            f"p50/p99 {np.median(wrong):.2f}/{np.percentile(wrong, 99):.2f}; "
            f"exactly the right one on: {np.mean((right > 0.5) & (wrong < 0.5)):.3f}",
            flush=True,
        )
        for tag, ls in [
            ("fit stories", fit_lines[:3]),
            ("test stories", test_lines[:3]),
        ]:
            n = story_inputs(model, tok, b1, ls)
            parts = {"const": n @ st.const, "ctx": n @ st.ctx}
            parts |= {f: np.abs(n @ fl[f]).max(1) for f in OPERANDS}
            print(
                tag,
                {
                    k: (round(float(v.min()), 2), round(float(v.max()), 2))
                    for k, v in parts.items()
                },
                flush=True,
            )
        debug_neurons(model, tok, nn, b1, test_lines[:3])
    k1 = nn.insert(s, b1)
    print(f"B1 block {b1}: {k1} neurons", flush=True)
    check("B1")

    # ---- B2: tens digit and the carries ----------------------------------
    st = Stage(model, tok, seqs, b2, fit_lines, y, neg, held=held if a.trace else None)
    names2 = ["P1u", "P1t", "P2u", "P2t", "U"]
    fl = readouts(st, names2, log, "B2")
    nn = Neurons()
    lookup(
        nn,
        st,
        [(fl["P1u"], 10), (fl["P2u"], 10), (fl["U"], 9)],
        lambda i, j, u: one("t", (i + j + u) % 10) + one("k", (i + j + u) // 10),
        st.const,
    )
    lookup(
        nn, st, [(fl["P1t"], 9), (fl["P2t"], 9)], lambda i, j: one("K", i + j), st.const
    )
    for name in names2:
        erase(nn, st, fl[name], dirs[name])
    k2 = nn.insert(s, b2)
    print(f"B2 block {b2}: {k2} neurons", flush=True)
    check("B2")

    # ---- B3: hundreds digit ----------------------------------------------
    st = Stage(model, tok, seqs, b3, fit_lines, y, neg, held=held if a.trace else None)
    names3 = ["k", "K", "Xu"]
    fl = readouts(st, names3, log, "B3")
    nn = Neurons()
    lookup(
        nn,
        st,
        [(fl["Xu"], 10), (fl["K"], 17), (fl["k"], 3)],
        lambda x, kk, c: one("h", (x + kk + c) % 10),
        st.const,
    )
    for name in names3:
        erase(nn, st, fl[name], dirs[name])
    k3 = nn.insert(s, b3)
    print(f"B3 block {b3}: {k3} neurons", flush=True)
    check("B3")

    # ---- relay: read t and h, write them fresh into new directions -------
    names4 = ["t", "h"]
    if a.relay >= 0:
        st = Stage(
            model, tok, seqs, a.relay, fit_lines, y, neg, held=held if a.trace else None
        )
        fl = readouts(st, names4, log, "relay")
        nn = Neurons()
        for name in names4:
            lookup(
                nn, st, [(fl[name], 10)], lambda d, n=name: one(n + "2", d), st.const
            )
            erase(nn, st, fl[name], dirs[name])
        kr = nn.insert(s, a.relay)
        print(f"relay block {a.relay}: {kr} neurons", flush=True)
        check("relay")
        y["t2"], y["h2"] = y["t"], y["h"]
        names4 = ["t2", "h2"]

    # ---- B4: write the digits --------------------------------------------
    st = Stage(model, tok, seqs, b4, fit_lines, y, neg, held=held if a.trace else None)
    fl = readouts(st, names4, log, "B4")
    push = digit_push(model, tok, a.margin) * final_rms(model, tok, test[:200])
    nn = Neurons()
    pos_t, pos_h = st.pos[TENS][:, None], st.pos[HUNDREDS][:, None]
    lookup(nn, st, [(pos_t, 1), (fl[names4[0]], 10)], lambda _, d: push[d], st.const)
    lookup(nn, st, [(pos_h, 1), (fl[names4[1]], 10)], lambda _, d: push[d], st.const)
    for name in names4:
        erase(nn, st, fl[name], dirs[name])
    nn.clamp(st.and_gate([st.ctx]), st.const, -dirs["ctx"][0], st.const)
    k4 = nn.insert(s, b4)
    print(f"B4 block {b4}: {k4} neurons", flush=True)
    check("B4")

    edit_all, edit_digits = score(model, tok, [tuple(x) for x in test])
    log["edit"] = {
        "all": edit_all,
        "digits": edit_digits,
        "neurons": [k0, k1, k2, k3, k4],
    }
    print(f"edited test: all {edit_all:.3f} digits {edit_digits}", flush=True)
    nll = text_nll(model, tok, texts)
    if a.gen:
        seen = {tuple(x) for x in fit} | {tuple(x) for x in test}
        pairs = mul2x2(a.gen, a.gen_seed, exclude=seen)
        qa = [(f"What is {x} * {z}?", x * z) for x, z in pairs]
        edit_items = generate(model, tok, qa)
    if a.side:
        edit_side = side_sets(model, tok)
    if a.general:
        seen = {tuple(x) for x in fit} | {tuple(x) for x in test}
        tasks = general_sets(seen)
        edit_general = {}
        for name, spec in tasks.items():
            edit_general[name] = run_set(model, tok, spec)
            right = sum(i.correct for i in edit_general[name])
            print(f"general {name} (edit): {right} of {len(spec[0])}", flush=True)
    s.restore()
    if a.general:
        log["general"] = {}
        for name, spec in tasks.items():
            b_items, e_items = run_set(model, tok, spec), edit_general[name]
            r0 = [int(i.correct) for i in b_items]
            r1 = [int(i.correct) for i in e_items]
            g = sum(b == 0 and e == 1 for b, e in zip(r0, r1, strict=True))
            lo = sum(b == 1 and e == 0 for b, e in zip(r0, r1, strict=True))
            log["general"][name] = {
                "mode": spec[1],
                "base": sum(r0),
                "edit": sum(r1),
                "gained": g,
                "lost": lo,
                "p": sign_p(g, lo),
                "of": len(r0),
                "questions": [q for q, _ in spec[0]],
                "answers": [x for _, x in spec[0]],
                "base_outputs": [i.output for i in b_items],
                "edit_outputs": [i.output for i in e_items],
                "base_right": r0,
                "edit_right": r1,
            }
            print(
                f"general {name} ({spec[1]}): base {sum(r0)} edit {sum(r1)} of "
                f"{len(r0)} (+{g} / -{lo}, p {sign_p(g, lo):.2g})",
                flush=True,
            )
    nll0 = text_nll(model, tok, texts)
    if a.side:
        base_side = side_sets(model, tok)
        log["side"] = {}
        for name in base_side:
            if name == "reply":
                same = sum(
                    x == z
                    for x, z in zip(base_side[name], edit_side[name], strict=True)
                )
                log["side"][name] = {
                    "same": same,
                    "of": len(base_side[name]),
                    "base": base_side[name],
                    "edit": edit_side[name],
                }
                print(f"side reply: same {same} of {len(base_side[name])}", flush=True)
                continue
            r0, r1 = base_side[name], edit_side[name]
            g = sum(b == 0 and e == 1 for b, e in zip(r0, r1, strict=True))
            lo = sum(b == 1 and e == 0 for b, e in zip(r0, r1, strict=True))
            log["side"][name] = {
                "base": sum(r0),
                "edit": sum(r1),
                "gained": g,
                "lost": lo,
                "p": sign_p(g, lo),
                "of": len(r0),
            }
            print(
                f"side {name}: base {sum(r0)} edit {sum(r1)} of {len(r0)} "
                f"(+{g} / -{lo}, p {sign_p(g, lo):.2g})",
                flush=True,
            )
    if a.gen:
        base_items = generate(model, tok, qa)
        r0 = [int(i.correct) for i in base_items]
        r1 = [int(i.correct) for i in edit_items]
        gained = sum(b == 0 and e == 1 for b, e in zip(r0, r1, strict=True))
        lost = sum(b == 1 and e == 0 for b, e in zip(r0, r1, strict=True))
        form = sum("×" in i.output for i in base_items)
        log["generation"] = {
            "n": len(qa),
            "base": sum(r0),
            "edit": sum(r1),
            "gained": gained,
            "lost": lost,
            "p": sign_p(gained, lost),
            "base_x_form": form,
            "base_right": r0,
            "edit_right": r1,
            "pairs": [list(x) for x in pairs],
            "edit_outputs": [i.output for i in edit_items],
            "base_outputs": [i.output for i in base_items],
        }
        print(
            f"generation (direct, {len(qa)} new products): base {sum(r0)} edit "
            f"{sum(r1)} gained {gained} lost {lost} p {sign_p(gained, lost):.2g}; "
            f"base replies with the x form: {form}",
            flush=True,
        )
    log["story_nll"] = {"base": nll0, "edit": nll}
    print(f"story nll base {nll0:.4f} edit {nll:.4f}", flush=True)
    (ROOT / "runs" / f"{a.name}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
