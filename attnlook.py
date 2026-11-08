"""Show where attention heads of one block look while the model writes an answer.

The input is the direct prompt + "The answer is " + the right answer (as in
evalmath.scoring_ids). For each position that predicts an answer digit, the
script takes the attention weights of the chosen heads and reports:
  - the share of attention on digit tokens of the question, on the answer
    digits written so far, and on all other tokens;
  - for --show examples, the top tokens of each head.

Example:
    uv run python attnlook.py --layer 20 --heads 12,13,14,15 --show 2
"""

import argparse

import mlx.core as mx
from mlx_lm import load

from evalmath import DEFAULT_MODEL, default_qa, scoring_ids


def attention_weights(model, ids: list[int], layer: int) -> mx.array:
    """[heads, L, L] attention probabilities of one block (no cache)."""
    x = mx.array([ids])
    h = model.model.embed_tokens(x)
    for blk in model.model.layers[:layer]:
        h = blk(h, "causal", None)
    blk = model.model.layers[layer]
    a = blk.self_attn
    y = blk.input_layernorm(h)
    B, L, _ = y.shape
    q = a.q_norm(a.q_proj(y).reshape(B, L, a.n_heads, -1)).transpose(0, 2, 1, 3)
    k = a.k_norm(a.k_proj(y).reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    q, k = a.rope(q), a.rope(k)
    k = mx.repeat(k, a.n_heads // a.n_kv_heads, axis=1)
    s = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * a.scale
    causal = mx.triu(mx.full((L, L), -mx.inf), k=1)
    return mx.softmax(s + causal, axis=-1)[0]


def is_digit_token(tok, t: int) -> bool:
    text = tok.decode([t]).strip()
    return text.isdigit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--heads", default="12,13,14,15")
    p.add_argument("--show", type=int, default=2, help="examples with top tokens")
    a = p.parse_args()
    heads = [int(x) for x in a.heads.split(",")]

    model, tok = load(a.model)
    totals = {h: [0.0, 0.0, 0.0] for h in heads}
    count = 0
    for n_ex, (q, ans) in enumerate(default_qa()):
        ids, n_prompt = scoring_ids(tok, q, ans)
        w = attention_weights(model, ids, a.layer)
        # The question sits in the user turn; its digits are before n_prompt.
        q_digits = [i for i in range(n_prompt) if is_digit_token(tok, ids[i])]
        for pos in range(n_prompt - 1, len(ids) - 1):
            done = list(range(n_prompt, pos + 1))
            for h in heads:
                row = w[h, pos]
                on_q = row[mx.array(q_digits)].sum().item() if q_digits else 0.0
                on_ans = row[mx.array(done)].sum().item() if done else 0.0
                totals[h][0] += on_q
                totals[h][1] += on_ans
                totals[h][2] += 1.0 - on_q - on_ans
            count += 1
            if n_ex < a.show:
                target = tok.decode([ids[pos + 1]])
                print(f"\n{q}  -> predicting {target!r} at position {pos}")
                for h in heads:
                    row = w[h, pos]
                    top = mx.argsort(-row)[:5].tolist()
                    cells = ", ".join(
                        f"{tok.decode([ids[j]])!r}@{j} {row[j].item():.2f}" for j in top
                    )
                    print(f"  head {h:2d}: {cells}")
    print(f"\nShare of attention over {count} answer-digit positions (math20):")
    print("head  question digits  answer so far  other")
    for h in heads:
        t = [x / count for x in totals[h]]
        print(f"{h:4d}  {t[0]:15.2f}  {t[1]:13.2f}  {t[2]:5.2f}")


if __name__ == "__main__":
    main()
