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

## 3. Attention or MLP of one block off (`sweep_parts.py --part blocks`, `runs/sweep_blocks.json`)

`surgery.py` zeros the input columns of `o_proj` (all heads) or `down_proj`
(all MLP neurons) of one block. Zeroing both is identical to removing the
block (max logit difference 0.0), and `restore()` gives the base logits back
exactly.

| Block | Attention off | MLP off |
|---|---|---|
| 0 | 0/20 (-25.8) | 13 (-2.25) |
| 1 | 7 (-2.02) | 0 (-15.4) |
| 2 | 12 (-2.22) | 0 (-16.8) |
| 3 | 12 (-1.36) | 20 (-2.15) |
| 4 | 19 (-2.04) | 16 (-2.55) |
| 5 | 18 (-1.89) | 20 (-2.01) |
| 6 | 17 (-2.73) | 16 (-2.62) |
| 7 | 16 (-2.26) | 18 (-2.23) |
| 8 | 19 (-1.74) | 20 (-1.93) |
| 9 | 14 (-1.86) | 19 (-2.46) |
| 10 | 18 (-2.73) | 16 (-2.96) |
| 11 | 14 (-4.42) | 19 (-2.47) |
| 12 | 17 (-3.10) | 19 (-2.69) |
| 13 | 19 (-2.51) | 18 (-3.19) |
| 14 | 17 (-2.30) | 20 (-1.90) |
| 15 | 20 (-2.63) | 18 (-3.18) |
| 16 | 19 (-1.57) | 18 (-3.65) |
| 17 | 12 (-3.67) | 15 (-3.10) |
| 18 | 17 (-2.29) | 15 (-5.33) |
| 19 | 16 (-2.50) | 12 (-3.95) |
| 20 | **3 (-8.10)** | 11 (-4.86) |
| 21 | 19 (-1.85) | **5 (-6.19)** |
| 22 | 18 (-2.14) | 10 (-4.14) |
| 23 | 19 (-2.09) | 16 (-3.22) |
| 24 | 17 (-1.58) | 12 (-4.06) |
| 25 | 15 (-2.56) | 18 (-3.63) |
| 26 | 18 (-1.85) | 19 (-2.19) |
| 27 | 16 (-1.86) | 11 (-2.30), 5120 tokens |

Values: correct answers out of 20 (mean logprob of the right answer; base
20/20, -1.87).

- Block 20: the attention is the critical part (3/20). Its MLP costs less
  (11/20). The MLPs of blocks 21, 22, 19 and 24 come next (5, 10, 12, 12).
  A possible reading: attention in block 20 brings the operands together and
  the later MLPs compute the result. This is a hypothesis; the head and
  neuron tests below check parts of it.
- Block 27: the MLP, not the attention, makes the end-of-turn token. Without
  it, every reply runs to the 256-token limit, but the logprob stays near the
  base.
- Early blocks: block 0 needs its attention; blocks 1 and 2 need their MLPs.
- Attention off in block 3: the logprob gets better (-1.36) but the score
  falls to 12/20, with fewer tokens (424 against 957). The model obeys "only
  the final number" more often and then gets the word problems wrong.

## 4. MLP neurons of block 20 (`neurons.py --layer 20`, `runs/neurons20.json`)

- Attribution: the gradient of the summed answer logprob on math20 with
  respect to a scale on each of the 3072 neurons. One backward pass per
  question.
- Held-out set: 40 random questions from `mathgen.py` (seed 1), of the same
  kinds. The base gets 37/40, mean logprob -3.11.
- Control: 3 random sets of each size.

| Zeroed neurons | math20 | math20 logprob | held-out | held-out logprob |
|---|---|---|---|---|
| none | 20/20 | -1.87 | 37/40 | -3.11 |
| top 1 (neuron 2530) | 18 | -2.22 | 37 | -3.32 |
| top 4 | 19 | -2.67 | 32 | -3.61 |
| top 16 | 13 | -4.38 | 27 | -4.94 |
| top 64 | 12 | -5.90 | 24 | -5.79 |
| top 256 | 11 | -6.53 | 21 | -7.05 |
| random 16 (3 sets) | 17 to 19 | -1.80 to -1.86 | 37 to 38 | -3.11 to -3.15 |
| random 64 (3 sets) | 18 to 20 | -1.86 to -2.36 | 36 to 37 | -3.14 to -3.51 |
| random 256 (3 sets) | 19 to 20 | -1.86 to -2.45 | 35 to 38 | -3.39 to -3.83 |
| bottom 16 | 20 | -1.45 | 36 | -2.94 |
| bottom 64 | 18 | -1.31 | 37 | -2.68 |
| bottom 256 | 19 | -2.11 | 33 | -3.19 |

- 16 of 3072 neurons (0.5%) carry much of the math effect of this MLP: the
  held-out score falls from 37 to 27 of 40. Random sets of 16 change nothing.
  The effect holds on questions that were not used to choose the neurons.
- 256 top neurons give about the same math20 score as the whole MLP off (11
  against 11).
- Zeroing the bottom 64 neurons makes the right answer more likely on both
  sets (math20 -1.87 to -1.31, held-out -3.11 to -2.68), but the scores do
  not rise. It is not yet known whether this costs other skills; a test on
  general text is needed.
- Noise: a random single neuron with no logprob change can still move math20
  from 20 to 18, because one changed token changes a greedy reply. Read small
  score changes on 20 questions with care; the logprob is the better measure.
- The first-order estimate ranks well but is too small: for the top 16 it
  predicts a summed change of -18.9, and the measured change is -50.2.
