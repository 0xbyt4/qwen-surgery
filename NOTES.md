# Experiment notes

All runs are on an M5 Pro with stock MLX 0.32.4 and mlx-lm 0.32.2 (`uv`
environment of this repo). No run trains the model: each change is written
directly into the weights or the forward pass.

## Model and test

- Model: Qwen3-0.6B, `mlx-community/Qwen3-0.6B-bf16` (1.19 GB, Apache-2.0).
  28 blocks, hidden size 1024, 16 query heads and 8 key-value heads of 128,
  MLP 3072, tied embeddings.
- Test: `math20.py`, 20 questions with integer answers: 10 arithmetic, 4 word
  problems, 2 equations, 4 others. The answers are Python expressions.
- `evalmath.py`, greedy decoding:
  - `direct`: thinking off, "Answer with only the final number", up to 256
    tokens. The score is the last number of the reply.
  - `think`: thinking on, up to 2048 tokens. The score is the last number
    after `</think>`.
  - Each question also gives `logprob`: log p(answer digits) after the direct
    prompt and the lead "The answer is ". It changes smoothly, so it shows
    small effects.
  - Speed and memory are logged per question.

## 1. Base model

| Mode | Correct | Generated tokens | Generation | Prompt | Time | Peak memory |
|---|---|---|---|---|---|---|
| direct | 20/20 | 957 | 186 tok/s | 2341 tok/s | 5.6 s | 1.26 GB |
| think | 19/20 | 13852 | 178 tok/s | 2245 tok/s | 78.4 s | 1.57 GB |

- Mean logprob of the right answer: -1.870.
- Thinking uses 14.5 times more tokens and is not better here. In question 8
  (23 * 47) the thinking did not end within 2048 tokens.
- With thinking off, the model still writes a short worked answer to the word
  problems; it does not obey "only the final number".
- The direct test is at 20/20, so it can show only losses. A test of
  improvements needs harder questions.

## 2. Remove one block (`sweep_skip.py`, `runs/sweep_skip_direct.json`)

The block is removed from the forward pass: the residual stream goes past it
unchanged.

| Removed block | Correct | logprob |
|---|---|---|
| none | 20/20 | -1.87 |
| 0, 1, 2 | 0/20 | -18.4, -20.5, -14.0 |
| 3 | 18 | -2.22 |
| 4 | 14 | -2.93 |
| 5 | 19 | -2.26 |
| 6 | 6 | -5.32 |
| 7 | 18 | -2.25 |
| 8 | 20 | -2.02 |
| 9 | 15 | -2.37 |
| 10 | 16 | -2.28 |
| 11 | 16 | -4.10 |
| 12 | 18 | -4.46 |
| 13 | 18 | -3.76 |
| 14 | 18 | -2.04 |
| 15 | 17 | -3.09 |
| 16 | 18 | -2.43 |
| 17 | 8 | -4.52 |
| 18 | 11 | -4.54 |
| 19 | 14 | -2.83 |
| 20 | 1 | -9.25 |
| 21 | 4 | -6.29 |
| 22 | 11 | -4.75 |
| 23 | 14 | -3.78 |
| 24 | 12 | -3.65 |
| 25 | 14 | -4.42 |
| 26 | 19 | -1.99 |
| 27 | 5 | -2.27 |

- Blocks 0 to 2 are needed for any output.
- Block 8 can go without a loss on this test; blocks 5 and 26 cost one
  question.
- Block 20 is the most important block after block 2 (1/20). Without it, the
  model still writes the steps, but the arithmetic is wrong: "47 + 38 = 59",
  "5 + 12 = 13". Blocks 17, 18 and 21 also cost much.
- Block 6: the reply becomes one number taken from the question ("38" for
  47 + 38, "5" for the apple problem). The model obeys "only the number" but
  does not compute.
- Block 27: the logprob stays near the base (-2.27), so the model still knows
  the answer. It writes the right answer, then `<|endoftext|>` in place of
  `<|im_end|>`, and does not stop: it writes new "Human:" turns. The score
  takes the last number, so most of these count as wrong. Block 27 controls
  the end of the turn, not the math.
- Speed: the runs without one block show about 212 tok/s against 190 for the
  base, which is more than the 3.6% of one block. The base ran first in the
  same process, so this is not a valid speed result. Compare speeds only with
  alternating runs.
