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

## 5. Attention heads (`sweep_parts.py --part heads`, `runs/sweep_heads.json`, `runs/text_heads20.json`)

One head off at a time in blocks 20, 17, 11 and 9 (serial runs; values are
correct answers / mean logprob, base 20 / -1.87):

- Block 20: no single head breaks the math. The largest effects are heads 15
  (16 / -3.24), 12 (15 / -3.19), 8 (17 / -2.96), 7 (16 / -2.38), 13 and 14
  (-2.32, -2.29). The other heads change the logprob by 0.07 or less.
- Block 17: head 12 (16 / -3.09) and head 11 (-2.59); others small.
- Blocks 11 and 9: no head below -2.46.

Groups of heads in block 20 (`textcheck.py`, no generation; story loss base
2.4606, math20 logprob -1.865, held-out -3.118):

| Heads off | Story loss | math20 | held-out |
|---|---|---|---|
| 12, 15 | 2.4623 | -3.88 | -5.76 |
| 12, 13, 14, 15 | 2.4619 | -5.68 | -9.03 |
| 7, 8, 12, 15 | 2.4800 | -4.90 | -8.33 |
| 7, 8, 12, 13, 14, 15 | 2.4813 | -7.09 | -10.36 |
| the other 9 heads | 2.4950 | -1.98 | -2.59 |
| all 16 (attention off) | 2.5210 | -8.12 | -10.37 |

- Heads 12 to 15 of block 20 (4 of 16) are a math part that general text does
  not need: the story loss changes by 0.001, and the held-out logprob falls
  from -3.12 to -9.03. Heads 12 and 13 share key-value head 6, heads 14 and 15
  share key-value head 7.
- Six heads (7, 8, 12 to 15) give the full effect of the attention on the
  held-out set (-10.36 against -10.37).
- The other 9 heads matter for general text (+0.034) and not for math; without
  them, the held-out logprob even gets better (-3.12 to -2.59).

## 6. MLP neurons of blocks 21 and 22 (`runs/neurons21.json`, `runs/neurons22.json`)

Same method as section 4 (serial runs):

| Zeroed neurons | Block 21 held-out | Block 21 held-out logprob | Block 22 held-out | Block 22 held-out logprob |
|---|---|---|---|---|
| none | 37/40 | -3.11 | 37/40 | -3.11 |
| top 16 | 32 | -4.04 | 32 | -3.89 |
| top 64 | 24 | -5.51 | 28 | -4.49 |
| top 256 | 24 | -6.20 | 32 | -5.02 |
| bottom 16 | 27 | -3.79 | 36 | -3.18 |
| bottom 64 | 23 | -4.66 | 36 | -2.99 |
| bottom 256 | 20 | -5.63 | 35 | -2.96 |
| random 16 / 64 (seed 0) | 36 / 39 | -3.11 / -3.11 | 37 / 37 | -3.13 / -3.14 |

- The math effect is less concentrated than in block 20 (top 16: held-out
  -4.04 and -3.89, against -4.94 in block 20).
- Block 21: the "bottom" neurons, which the gradient says the answers do
  better without, make the answers much worse when zeroed (held-out 37 to 23
  for 64 neurons; the math20 logprob also falls, from -1.87 to -2.67, against
  a predicted gain). The first-order estimate has the wrong sign here. A
  neuron set chosen by gradient must always be checked by a real zeroing.

## 7. Batch evaluation (`evalmath.run(..., batch=True)`, now the default)

- All questions are generated in one `batch_generate` call, and the answer
  logprobs come from one forward pass with right padding. The output head of
  the logprob runs in float32: the bf16 logits are rounded to steps of 0.125
  near 16 to 32.
- Speed (M5 Pro on battery, 3 rounds, order alternated, wall time including
  the logprob pass): math20 6.0 to 6.3 s serial against 1.01 to 1.02 s batch
  (about 6 times faster); held-out 40 questions 10.7 to 11.0 s against 1.34 s
  (about 8 times). Peak memory 1.86 GB serial, 2.46 and 3.56 GB batch.
- The batch run is not bit-identical to the serial run: bf16 sums run in a
  different order, so a greedy reply can take another path at a near tie. On
  the base model, 13 of 20 math20 replies have the same text and 19 of 20 the
  same answer (19/20 against 20/20 correct); held-out 37/40 in both.
- Mean logprob: serial -1.865, batch -1.887, float32 model -1.884 (math20);
  -3.118, -3.112, -3.102 (held-out). The mean is stable to about 0.02. Single
  questions can differ by up to 0.4 between bf16 runs, so compare means.
- Results before this section come from serial runs; compare runs only within
  one mode.

## 8. Hard math set (`mathgen.hard_questions`, `editcheck.py`)

- Six kinds, 20 of each in 120 questions: 3-digit times 2-digit products,
  sums of four 3-digit numbers, two-step word problems, equations with x on
  both sides, mixed expressions, 5-digit minus 4-digit.
- Base, thinking off, batch, up to 512 tokens, seed 2: 94/120 (products 7,
  sums of four 11, word problems 17, equations 20, mixed 20, subtraction 19);
  2 replies cut at 512 tokens. Mean answer logprob -6.24.
- Edits are chosen on seed 2 ("dev") and confirmed on seeds 3 and 4.

## 9. What the block 20 math heads look at (`attnlook.py`, `runs/attnlook20.log`)

The attention weights are computed again from the block input. The output
built from them matches the module output within bf16 rounding (max
difference 0.03 at a scale of 11.75).

- While the model writes an answer digit, heads 12 to 15 and head 8 look at
  the digits of the question: 56%, 81%, 70%, 50% and 73% of their attention
  over the 43 answer-digit positions of math20. Heads 0, 3 and 7 look almost
  only at the first token (`<|im_start|>`) or at the answer lead.
- The heads pick digits by place value. "47 + 38 = 85":
  - for the first digit "8": head 14 takes "4" (tens of 47), head 8 takes "3"
    (tens of 38), heads 15 and 12 take "7" and "8" (the units, which decide
    the carry);
  - for the second digit "5": heads 13 and 14 take "7", head 8 takes "8" (the
    two units digits; 7 + 8 ends in 5).
- "125 - 68 = 57": for the "7", heads 13 and 14 take "5" and head 8 takes
  "8" (units, with a borrow). For the "5", head 15 takes "5", heads 13 and 14
  take "2", head 8 takes "6", head 12 takes "8".
- So these heads route the operand digits of the right column to the
  position that writes the next digit. The later MLPs (blocks 21 and 22) are
  candidates for the digit arithmetic itself; this is not yet tested.

## 10. Scaling the math heads up (`editcheck.py`, `runs/scale_dev_*.json`, `runs/scale_test_s3.json`, `runs/scale_test_s4.json`)

An edit `head:L:H1,...*F` multiplies the o_proj columns of those heads by F.
Seed 2 was used to choose the edits; seeds 3 and 4 are new questions. Hard
set: correct of 120 per seed (sums of four numbers in brackets, of 20).

| Edit | Seed 2 (dev) | Seed 3 | Seed 4 | Total of 360 | Story loss |
|---|---|---|---|---|---|
| none | 94 [11] | 90 [11] | 84 [11] | 268 | 2.4606 |
| heads 12-15 x1.5 | 97 [15] | 99 [15] | 88 [15] | 284 | 2.4665 |
| heads 7, 8, 12-15 x1.5 | 98 [15] | 99 [18] | 87 [16] | 284 | 2.4686 |
| the other 9 heads off | 99 [19] | 94 [16] | 92 [19] | 285 | 2.4950 |
| control: heads 0, 3, 5, 10 x1.5 | | 91 [10] | 85 [11] | (176 of 240; base 174) | 2.4664 |
| control: heads 1, 2, 4, 6, 9, 11 x1.5 | | 85 [9] | 90 [12] | (175 of 240; base 174) | 2.4718 |
| control: block 17 heads 0, 3, 5, 10 x1.5 | | 87 [10] | 81 [12] | (168 of 240; base 174) | 2.4626 |

Other scales on seed 2: heads 12-15 x1.25 92, x2 92, x3 60; heads 7, 8,
12-15 x2 95. The top 16 or 64 neurons of block 20 scaled by 1.5, 2 or 3 gave
78 to 94 (no gain), although the easy-set logprob got better (math20 -1.89
to -1.58 for the top 16 x1.5).

- Heads 12 to 15 of block 20 x1.5 give +16 of 360 hard questions (+4.4
  points), with +9 and +4 on the two new seeds. The same scale on other heads
  gives +2, +1 and -6 on the new seeds.
- The gain is clearest on sums of four numbers: from 11/20 to 15 to 19 in
  every set and every math-head edit, while the controls stay at 9 to 12.
  These sums need the most digits fetched at once.
- The price on general text is small but not zero: the story loss rises by
  0.006 (heads 12-15 x1.5), about as much as for the control scalings.
  Zeroing the other 9 heads gives the same math gain at 5 to 6 times that
  price (+0.034).
- Too much scaling breaks the model: x3 drops to 60/120.
- Each set has only 120 questions and greedy replies are sensitive to small
  changes, so a single-set difference of 3 to 6 is within noise (a control
  edit moved one set by +6). The claim rests on the three sets together and
  on the sums of four numbers.

## 11. The edited model on disk (`export_edit.py`)

- `models/Qwen3-0.6B-mathheads15`: heads 12 to 15 of block 20 scaled by 1.5,
  saved as one safetensors file with the config and tokenizer (git-ignored,
  1.19 GB). `EDIT.json` in the folder records the edit.
- It loads with `mlx_lm.load`. Its logits equal those of the in-memory edit
  (max difference 0.0), and it gives the same hard-set result (seed 3: 99/120,
  same counts per kind).

- Price on products (found later, section 15): this model is worse at
  two-digit by two-digit products. Generation on 500 new products: 288
  correct against 332 for the base model (-8.8 points; two runs each, same
  counts). The hard set did not show it: its products (three-digit by
  two-digit) are right only 3 to 9 of 20 times, so the set cannot see a
  loss there. The edit trades one skill for another; it is not a free gain.

## 12. Where the answer digit forms (`circuit.py`, `runs/lens_add2.log`, `runs/patch_add2.log`)

Data: 200 two-digit additions with a three-digit sum, direct prompt, the
answer after "The answer is ".

Logit lens (the residual after each block through the final norm and head;
share of examples with the right digit on top):

| After block | tens | units |
|---|---|---|
| 0 to 20 | 0.00 to 0.01 | 0.00 to 0.07 |
| 21 | 0.10 | 0.58 |
| 22 | 0.27 | 0.69 |
| 23 | 0.43 | 0.93 |
| 24 | 0.84 | 0.97 |
| 25 | 0.96 | 1.00 |

Patching: pairs that differ only in the units digit of B (so the units digit
of the sum differs). One part output, at the position that writes one digit,
is copied from the source run into the base run; 1 means the answer digit
moves fully to the source.

| Part | units | tens |
|---|---|---|
| block 20 attention | 0.86 | 0.26 |
| block 20 MLP | 0.53 | 0.18 |
| block 21 MLP | 0.80 | 0.20 |
| block 22 MLP | 0.23 | 0.03 |
| blocks 23 to 25 MLP | 0.01 to 0.30 | 0.16 to 0.26 |
| all parts of blocks 0 to 19 | 0.00 to 0.13 | 0.00 to 0.03 |

- At the answer position, nothing before block 20 carries the operand digit:
  block 20 attention brings it in (section 9), and the MLPs of blocks 20 and
  21 carry it on.
- The units digit is ready after block 23; the tens digit, which needs the
  carry, after block 24. The carry is spread over the MLPs of blocks 20 to 25.

## 13. Digit neurons: Fourier waves in block 21, one-hot digits in block 22 (`tuning.py`, `runs/tuning2*.json`)

For each MLP neuron at the position that writes the units digit (1000
additions), eta squared gives the share of its variance explained by a label.

| Block | neurons with eta^2 > 0.3 for: sum units | A units | B units | carry |
|---|---|---|---|---|
| 20 | 0 | 15 | 267 | 3 |
| 21 | 5 | 289 | 768 | 5 |
| 22 | 446 | 19 | 61 | 5 |
| 23 | 1193 | 0 | 9 | 16 |

- Blocks 20 and 21 hold the operands, block 22 and 23 hold the sum digit. The
  sum is formed between them.
- Block 21: the top neurons for the sum digit are pure waves over the digit
  0 to 9. In each, one frequency holds 97 to 99% of the variance of the class
  means, and all five frequencies of a period-10 signal occur: period 10
  (neurons 1496, 2435), 5 (2040), 3.3 (2331), 2.5 (1002) and 2 (569, 2591,
  1073; these fire on even sums). This is a Fourier code of the sum mod 10.
- Block 22: one neuron for each digit. Neuron 613 fires for 1, 816 for 2
  (negative), 779 for 3, 721 for 4, 452 for 5, 1787 for 6, 1441 for 7, 1092
  for 8, 612 for 9 (negative), 1187 for 0 (negative), with activations of 30
  to 50 for its digit and near 0 for the others. Their spectra are flat, as
  expected for a one-hot code. Block 22 turns the waves into digits.
- Causal test, units-digit accuracy on 500 new additions (base 1.000):

| Zeroed | block 21 | block 22 | block 23 | block 20 |
|---|---|---|---|---|
| top 4 for the sum digit | 0.906 | 0.994 | 1.000 | 1.000 |
| top 16 | 0.346 | 1.000 | 1.000 | 1.000 |
| top 64 | 0.216 | 1.000 | 1.000 | 0.998 |
| random 4, 16 or 64 | 1.000 | 1.000 | 1.000 | 1.000 |

- The 16 block 21 wave neurons on all tests (`runs/fourier21.json`, seed 3):

| Zeroed | hard set | sums of four | 5-digit subtraction | math20 logprob | held-out logprob | story loss |
|---|---|---|---|---|---|---|
| none | 90/120 | 11/20 | 18/20 | -1.89 | -3.11 | 2.4606 |
| block 21 top 16 | 19/120 | 0/20 | 0/20 | -3.74 | -6.46 | 2.4609 |
| block 21 random 16 | 91/120 | 11/20 | 18/20 | -1.90 | -3.11 | 2.4615 |

- 16 of the 86,016 MLP neurons of the model (0.02%) carry most of its
  arithmetic, and plain stories do not need them (+0.0003 loss).
- Scaling these 16 neurons up does not help (seed 2, `runs/fourier21_scale_dev.json`):
  x1.25, x1.5 and x2 give 95, 95 and 96 of 120 (base 94); the top 8 x1.5 give
  95. Together with heads 12-15 x1.5 they give 96 and 98, against 97 for the
  heads alone. The wave neurons are needed, but they are not what limits the
  hard set.
- The one-hot neurons of block 22 can be zeroed without loss: the digit is
  also carried by other neurons. The wave neurons of block 21 are a narrow
  point of the circuit.
- Related work (checked on arXiv): Zhou et al. 2024, arXiv 2406.03445 (LLMs
  add with Fourier features; in their models attention does the
  high-frequency, modular part); Kantamneni and Tegmark 2025, arXiv
  2502.00873 (numbers as a helix, the "Clock" algorithm); Nikankin et al.
  2024, arXiv 2410.21272 (arithmetic from a sparse set of heuristic
  neurons). Our results fit these; here the high-frequency waves sit in a
  block 21 MLP, after the attention of block 20 has fetched the digits.

## 14. Writing a digit: block 21 for the units, block 25 for any digit (`steer.py`, `runs/steer*.json`)

Sufficiency test. For each digit d, the mean activation of each neuron over
examples whose answer digit is d comes from 1000 examples (seed 2). On 500
new examples (seed 3) a wrong target digit is chosen at random, and a set of
neurons is set to its means for the target, at the position that writes the
digit. The score is the share of examples whose top digit becomes the target.

Tasks: `add_units` and `add_tens` (units and tens digit of A + B, two-digit
A and B); `mul_units` (units digit of A * B, A two-digit, B one-digit).
Base digit accuracy: 1.000, 0.986 and 1.000.

All neurons of one MLP set (`steer.py --layers 14-27`); blocks not listed
give 0.000 to 0.012:

| Block | add_units | add_tens | mul_units |
|---|---|---|---|
| 20 | 0.048 | 0.140 | 0.002 |
| 21 | 0.892 | 0.444 | 0.002 |
| 22 | 0.134 | 0.004 | 0.022 |
| 23 | 0.000 | 0.012 | 0.060 |
| 24 | 0.000 | 0.126 | 0.004 |
| 25 | 0.134 | 0.666 | 0.596 |
| 26 | 0.002 | 0.004 | 0.116 |

- The units digit of a sum is written in block 21. The tens digit (which
  needs the carry) and the units digit of a product are written later, in
  block 25.
- Block 21, add_units, top k neurons from section 13 (random k: 0.000 to
  0.004):

| k | 4 | 16 | 64 | 256 | all 3072 |
|---|---|---|---|---|---|
| moved to the target | 0.022 | 0.588 | 0.890 | 0.884 | 0.892 |

  The 64 top neurons (2% of the MLP) do all that the whole MLP can do. The
  wave neurons are sufficient as well as necessary.
- Block 21 does not compute the product digit: setting all of its neurons
  moves 0.002, and zeroing its top 64 for the product leaves 0.998.
  Zeroing a whole MLP of blocks 19 to 23 leaves 0.85 to 0.98 on products:
  the product digit is not in one narrow point like the sum digit.
- Block 25 holds one-hot digit neurons (`tuning.py --layer 25`): 92 for 0,
  1452 for 1, 2298 for 2, 1749 for 3 (negative), 2519 for 4 (negative), 753
  for 5, 2664 for 6 (negative), 2383 for 7, 2029 for 8, 333 for 9, each with
  eta^2 0.8 to 0.93. The same neurons code the tens digit of a sum and the
  units digit of a product: 13 of the top 16 are shared, with the same
  digit for each neuron.
- Cross-task test at block 25 (top neurons and means from one task, set on
  another; k 16 / 64): means from add_tens move mul_units 0.418 / 0.436
  (its own means: 0.476 / 0.458), and even add_units 0.326 / 0.342. Means
  from mul_units move add_tens 0.386 / 0.394 (own: 0.356 / 0.372).
- Scaled writes: value = mean over digits + alpha * (target mean - mean
  over digits), with means and neurons from add_tens only:

| Task | k | alpha 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| mul_units | 16 | 0.418 | 0.862 | 0.944 | 0.954 |
| mul_units | 64 | 0.436 | 0.926 | 0.996 | 1.000 |
| add_tens | 16 | 0.356 | 0.820 | 0.972 | 0.958 |
| add_tens | 64 | 0.372 | 0.912 | 1.000 | 1.000 |
| add_units | 16 | 0.326 | 0.870 | 0.962 | 0.940 |
| add_units | 64 | 0.342 | 0.946 | 1.000 | 1.000 |

- Block 25 has a digit output code that does not depend on the task: 64
  neurons, with values taken from additions alone, write any chosen digit
  into a sum or a product in all 500 of 500 cases. With 16 neurons it is 94
  to 97%.
- Block 21 top neurons for the tens digit are weaker (top eta^2 0.65, zeroing
  the top 64 leaves 0.944) and its top 64 for products share 9 neurons with
  the top 64 for sums (neuron 2591, the period-2 "even" wave, is in both).

## 15. Two-digit products: where the errors are, and what does not fix them (`probe.py`, `mulfix.py`, `runs/probe_mul2x2.json`, `runs/mulfix_search.json`)

Task: A * B with A and B two-digit and a four-digit product. In generation
(direct mode, 200 questions) the model is right 133 times; three-digit by
one-digit products and three-digit sums are right 198 times. The model
replies as "79 × 26 = 2014", so the tests below use that format with teacher
forcing. With the "The answer is " lead the model must give the first digit
without writing the sum first, and three-digit sums then fail (thousands
digit right 24%): that format is not usable here.

Per digit, teacher forced, 1000 products (seed 3): thousands 0.971, hundreds
0.844, tens 0.784, units 0.997; all four 0.67.

- Hundreds errors are carry errors: 146 of 156 are +1 or -1 off. Tens
  errors are spread over all offsets.
- The errors are confident: the right digit is second in 97% (hundreds) and
  88% (tens) of cases, but the median log probability gap to the top digit
  is about 3, and only 14 to 24% of errors have a gap below 1.
- A ridge probe per block (fit on 3000 products, seed 2) on the residual at
  the digit position. On model errors, the block 25 probe is right 53%
  (hundreds) and 55% (tens), but on model-right examples it is right only
  85% and 86%: the probe is weak where the model is weak. This is not
  evidence of a right digit hidden inside.
- The block 27 probe beats the model's own output: hundreds 0.882 against
  0.844, tens 0.857 against 0.784. Caveat (section 20): 350 of the 1000
  test products are also in the fit set, so part of this may be memorized. A per-digit bias on the model's logits
  gives only 0.858 and 0.800, and a probe on the operand digits alone gives
  0.246 and 0.141, so the gain is neither calibration nor a simple shortcut.
- But it does not transfer. One probe fit on all digits of these products
  lifts them from 0.668 to 0.731 (all digits right), and lowers three-digit
  by one-digit products from 0.985 to 0.954, three-digit sums from 0.984 to
  0.980 and three-digit by two-digit products from 0.249 to 0.237. A probe
  fit on labels learns the task; it does not show a general flaw in the
  output head.
- Weight-scale search (`mulfix.py --search`, seed 2, 1000 products, all
  four digits right; base 0.649): the block 25 digit code x1.25 to x2
  (16 or 64 neurons) gives -0.030 to +0.010; the attention or MLP of one
  block of 16 to 27 x0.8 or x1.25 gives -0.080 to +0.026. The best dev edits
  on 2000 new products (base 0.672): MLP 17 x1.25 0.678, MLP 18 x1.25 0.667,
  both 0.652, MLP 25 x1.25 0.676, attention 25 x0.8 0.674. All within noise:
  the dev gains were selection noise.
- Making the block 25 digit code stronger does not help, though writing it
  works (section 14): when the model is wrong, the code holds the wrong digit.
- The block 20 math-head edit of section 10 hurts these products: teacher
  forced 0.672 to 0.597 (x1.5), 0.648 (x1.25), 0.653 (heads 8, 12-15 x1.25);
  in generation 332 to 288 of 500 (x1.5).

## 16. The product circuit (`mulpatch.py`, `runs/mulpatch_*.json`)

Patching on 200 pairs A * B and A * B' (B' differs in its units digit).
Both runs get the base product as the forced answer, so the inputs differ
only in B. (A first version forced each run's own product; then the input
token at the scored position differed too, and block 0 attention scored
0.36. That version was dropped.)

Blocks (0 to 19 all score between -0.04 and 0.01):

| Block | attn @hundreds | attn @tens | MLP @hundreds | MLP @tens |
|---|---|---|---|---|
| 20 | 0.27 | 0.61 | 0.16 | 0.23 |
| 21 | 0.00 | -0.01 | 0.05 | 0.32 |
| 22 | -0.01 | 0.21 | 0.07 | 0.34 |
| 23 | 0.04 | 0.18 | 0.12 | 0.46 |
| 24 | 0.01 | 0.00 | 0.16 | 0.59 |
| 25 | 0.38 | 0.12 | 0.39 | 0.56 |
| 26 | 0.24 | 0.17 | 0.39 | 0.33 |
| 27 | 0.15 | 0.25 | 0.38 | 0.21 |

- As for sums, the operands enter through block 20 attention. The tens
  digit then forms over the MLPs of blocks 21 to 25, and the hundreds digit
  late, in blocks 25 to 27.
- Single heads (block 20 heads score at most 0.10 each; the effect is spread
  over heads 6, 8 and 12 to 15): block 22 head 14 (tens 0.28), block 25 head
  15 (hundreds 0.27) and block 26 head 6 (hundreds 0.19, tens 0.21).
- What they look at (attention on 60 products):
  - Block 20 heads 6 and 12 to 15 look at the units digits of A and B in the
    reply, also when the hundreds digit is written; only head 8 looks at the
    tens digits there (B tens 0.30, A tens 0.17). The hundreds digit needs
    A tens x B tens.
  - Block 22 head 14 looks at the current token (the digit just written,
    0.89 to 0.94).
  - Block 25 head 15 and block 26 head 6 look one position back (0.36 to
    0.70) and at the first token, an attention sink: they copy what was
    computed at the previous answer position.
- Scaling these heads only hurts (dev, all four digits, base 0.649): block
  22 head 14 x0.5/1.25/1.5/2 0.621/0.633/0.584/0.426; block 25 head 15
  0.567/0.570/0.507/0.423; block 26 head 6 0.593/0.642/0.621/0.559. The model
  sits at a good point for these scales.
- Block 20 head 8, the only one that looks at the tens digits: x1.25, x1.5
  and x2 give 0.658, 0.658 and 0.651 (+0.009 at most, within noise), x0.5
  gives 0.598: it is needed, but more of it does not help. Head 6 x1.25 gives
  0.588 and x0.5 0.595. The edit "the other 9 heads of block 20 off", which
  lifts sums of four numbers most (section 10), gives 0.575 (-0.074).
- Look-ahead: a probe (as in section 15) reads a later digit at an earlier
  answer position. At the position that writes the thousands digit, the
  hundreds digit is readable 0.54 (block 25) to 0.59 (block 27); the tens
  digit at the hundreds position 0.50. The operands alone give 0.25 and
  0.14. The model computes part of the next digit one step early, and the
  previous-position heads carry it forward. The units digit is readable
  0.69 to 0.79 at earlier positions already after block 20: it is computed
  early, as for sums.

## 17. The math gain is a reply-style choice, and one output row sets it (`tradeoff.py`, `runs/tradeoff_*.json`, `runs/add4_style.json`, `runs/outrow_s3.json`)

Sets (generation, direct mode, greedy): add4 = sums of four three-digit
numbers (200), mul2x2 (300), mul3x2 = three-digit by two-digit products
(200, as in the hard set). A first run with 96 new tokens cut 96 of 200
add4 replies; all runs below use 512 or 768 tokens. Correction (section
20): a few replies are still cut (`outrow_s3.json`: 1, 1 and 4).

Head subsets of block 20 x1.5 (seed 2; base add4 120, mul2x2 190):
12-15 141/167, 12 119/181, 13 127/181, 14 118/190, 15 127/166, 12+13
127/174, 14+15 136/175, 13-15 141/172, 12-14 122/176, 13+14 125/180, 13+14
x2 132/161; 12-15 x1.25 131/174, x1.15 125/175. No subset gains on sums
without a loss on products.

Why the head edit helps sums:
- The model answers a sum of four numbers in one of two styles: one shot
  ("262 + 541 + 753 + 502 = **1,856**") or step by step ("To find the sum
  ... 983 + 969 = 1952 ..."). One-shot replies are right 12 to 18% of the
  time, step-by-step replies about 91%.
- The head edit does not change these rates (one shot 0.18 to 0.19 and 0.12
  to 0.10, step by step 0.93 to 0.91 and 0.90 to 0.91, seeds 2 and 3). It
  changes the style: one-shot replies fall from 87 to 57 and from 84 to 48
  of 200. All of its gain on sums is this shift.
- The style is the first token of the reply: "To" (step by step) or a
  digit. On 200 sums (seed 3) the base model starts with "To" 115 times,
  the edited model 143 times (mean p("To") 0.43 to 0.47).
- Forcing the base model to start with "To find the sum" gives 190 of 200
  (seed 3; unforced 114). For products: "To find the product" gives mul2x2
  234 of 300 (unforced 195) and mul3x2 157 of 200 (unforced 58); "Let's
  break it down step by step." gives 233 and 149.

The edit: scale the output-head row of the token "To" (id 1249). The head
is tied to the input embedding, so the model gets its own copy of it
(`surgery.scale_output_rows`, spec `out:1249*F`); the input embedding stays.
The logit of "To" scales by F (at the reply start, about 26, so x1.1 adds
2.6); no other logit changes.

| Edit (seed 2) | add4 | mul2x2 | mul3x2 |
|---|---|---|---|
| none | 120 | 190 | 76 |
| "To" x1.02 | 164 | 190 | 76 |
| x1.04 | 185 | 190 | 76 |
| x1.06 | 187 | 190 | 76 |
| x1.1 | 189 | 190 | 82 |
| x1.15 | 189 | 194 | 113 |
| x1.2 | 189 | 215 | 153 |
| x1.25 | 189 | 219 | 167 |
| x1.3 | 189 | 221 | 170 |

Products need a larger factor: at their reply start the "To" logit is about
23 to 24 against 28 for the top digit.

Full check on new questions (seed 3, 768 tokens, `editcheck.py`):

| Edit | hard set | mul3x2 | add4 | word2 | mixed | sub_big | story loss |
|---|---|---|---|---|---|---|---|
| none | 90/120 | 8 | 11 | 14 | 19 | 18 | 2.4606 |
| "To" x1.1 | 95 | 8 | 18 | 13 | 18 | 18 | 2.4607 |
| "To" x1.25 | 105 | 16 | 18 | 13 | 18 | 20 | 2.4621 |
| heads 12-15 x1.5 | 99 | 9 | 15 | 16 | 20 | 19 | 2.4665 |

(linear2 is 20 in all rows. The math20 and held-out logprobs score the
answer after the "The answer is " lead, so the "To" edit cannot change
them: -1.887 and -3.112 in all three "none"/"To" rows.)

- Other prompts: of 20 general questions (capital of France, a haiku, how
  to boil an egg, ...), 19 replies are the same token for token. The one
  that changed ("How can I convert Celsius to Fahrenheit?") now starts with
  "To convert" and gives 9/5, where the base reply said 9.5.
  Correction (section 19): this check used the math prompt, which adds
  "Answer with only the final number." to each question, so the replies
  were short. In a plain chat turn, 23 of 30 replies stay the same.
- Cost: the replies are longer. Hard set, seed 3: 87.4 new tokens per
  question on average, 122.0 with "To" x1.25 (+40%).
- Speed: no change. 3 alternating runs of 256 tokens: base 251.9 to 254.2
  tok/s, edited 253.6 to 253.9 tok/s. The tied model reads the same matrix
  as its output head; the input embedding is only a row lookup.
- Saved as `models/Qwen3-0.6B-to125` (git-ignored, 1.50 GB, 0.31 GB more for
  the own head; `config.json` has `tie_word_embeddings: false`). It loads
  with `mlx_lm.load`; logit difference to the in-memory edit 0.0; mul3x2
  seed 2: 167/200, as in memory.
- This changes the reading of section 10: the head edit raised the hard
  set mostly by pushing the model toward step-by-step replies, while it
  lowered the arithmetic itself (products, section 15). The one-row edit
  gets more of the style gain without the arithmetic loss.

## 18. Word problems: a rounding choice, and the row of " down" (`wordcheck.py`, `runs/wordcheck_s6.json`, `runs/outrow2_s3.json`)

The hard-set word problem: buy A packs of B, give away C, split the rest
into D bags "with any extra kept aside"; the answer is the whole part of
(A * B - C) / D. 300 questions per seed (`wordcheck.word2`).

- A flaw in the test set: when A * B < C, the set expects 0. These
  questions are ill-posed (21% on seed 5, 19% on seed 6). They are counted
  apart below ("well-posed"). Correction (section 20): the numbers below
  used "answer is not 0", which also left out 10 (seed 5) and 12 (seed 6)
  well-posed questions whose answer is 0; `wordcheck.py` now uses
  A * B >= C.
- The "To" edit of section 17 does not help here: the replies start with
  the name in the question ("Sam", "Leo"), and they are already step by
  step (seed 5: 189 of 300, with the edit 184).
- Where the errors are (seed 5, base, well-posed): exact shares 74 of 74
  right; shares with a remainder 91 of 156. The wrong ones round to the
  nearest whole number ("23 / 6 = 3.83..., we round to the nearest whole
  number: 4") or give a decimal or fraction (2.5, 1.2, 1/6).
- "round to the nearest" occurs in 76 replies (38 wrong; of the other 38,
  20 are right because the fraction is below 0.5, section 20), "round down" in 18 (all
  right). The choice is the token after " round": " to" or " down". There
  the logit of " to" is above that of " down" by 0.52 (median of 94; 90th
  percentile 1.83), a ratio of 1.025 (90th percentile 1.092).
- Edit: the output row of " down" (id 1495), as in section 17.

| Edit | seed 5 well-posed (220) | seed 5 all (300) | story loss |
|---|---|---|---|
| none | 163 | 189 | 2.4606 |
| " down" x1.05 | 183 | 210 | 2.4608 |
| " down" x1.1 | 194 | 223 | 2.4617 |
| " down" x1.15 | 199 | 228 | 2.4638 |
| "To" x1.25 + " down" x1.1 | 190 | 218 | 2.4633 |

- New seed (6, chosen after the factor): base 179 of 227 well-posed (209 of
  300), " down" x1.1 208 (238), with "To" x1.25 207 (242); a second base run
  gave 179 again. +13 points on well-posed questions.
- Counter-sets, where rounding to the nearest is right (fraction above
  0.5; 100 each: "X.YZ rounded to the nearest whole number", "divide A by
  B and round to the nearest whole number", "km per day to the nearest
  km"): 93, 93 and 99 for the base model and the same for " down" x1.1 and
  x1.15. When the question asks for the nearest number, the model still
  rounds to the nearest. The edit only moves the choice where the question
  leaves it to the model.
- Hard set, seed 3 (`editcheck.py`, 768 tokens): " down" x1.1 91 of 120
  (word2 14 to 15; the set has only 20 word problems, about a quarter of
  them ill-posed), story loss 2.4617. Both rows ("To" x1.25 + " down" x1.1):
  106 of 120 (base 90, "To" alone 105), story loss 2.4633.
- General prompts (25, now with "the opposite of up", "how do I calm down",
  "should I write my ideas down", "when the stock market goes down"): 24
  replies are the same token for token with both rows; the one that changed
  is the Celsius reply of section 17. Correction (section 19): this check
  used the math prompt; in a plain chat turn " down" x1.1 keeps 27 of 30
  replies.
- Saved as `models/Qwen3-0.6B-to125-down110` (both rows; git-ignored). From
  disk: word2 seed 6 207 of 227 well-posed, 242 of 300, as in memory.

## 19. A dropped subtraction, the row of " -", and a better reply check (`mixedcheck.py`, `replycheck.py`, `runs/mixedcheck_s12.json`, `runs/reply_chat*.json`, `runs/tradeoff_minus_s3.json`, `runs/text_minus.json`)

Hard set, seed 8, 600 questions (100 per kind), 768 tokens:

| Model | all | mul3x2 | add4 | word2 | linear2 | mixed | sub_big |
|---|---|---|---|---|---|---|---|
| base | 443 | 39 | 60 | 58 | 100 | 97 | 89 |
| "To" x1.25 + " down" x1.1 | 538 | 85 | 93 | 69 | 100 | 96 | 95 |
| + " -" x1.1 | 541 | 85 | 93 | 70 | 100 | 99 | 94 |

(word2 counts the ill-posed questions of section 18 as wrong.)

- A regression of the "To" edit: on A * B + C - D * A (`mixedcheck.mixed`,
  300 per seed) the base model is right 287 times (seed 11), "To" x1.25
  268 times. " down" x1.1 alone: 287. 24 of its errors give P + C: the
  model drops the subtraction (base: 4 such errors). Paired: 6 questions
  gained, 25 lost, p = 0.0009 (section 20).
- Why: base replies start with "We" and work in steps. "To" replies
  compute the two products and then write "P + C - Q = " in one line (88 of
  300). There they write P + C first. Then the next token decides: " -"
  continues ("= 254 - 95 = 159", right), "\n" stops ("= 1175", wrong). In
  the wrong replies "\n" leads " -" by 0.31 logits (median of 26); in the
  right ones " -" leads by 2.6 (median of 58).
- Edit: the output row of " -" (id 481), on top of the two rows. Seed 11
  (dev): x1.05 287, x1.1 290, x1.2 292.
- Seed 12 (new, `mixedcheck.py`, three sets of 300):

| Edit | mixed | dropped | three | sub_big |
|---|---|---|---|---|
| none | 293 | 0 | 294 | 278 |
| "To" x1.25 + " down" x1.1 | 266 | 24 | 297 | 284 |
| + " -" x1.1 | 288 | 2 | 297 | 283 |
| " -" x1.1 alone | 293 | 0 | 214 | 278 |

- The " -" row alone harms the base model: X + Y - Z drops from 294 to 214.
  The base reply is "585 + 375 = 960, 960 - 773 = 187". With the edit the
  model writes "585 + 375 - 773 = " in one line and then fails ("= 952 -
  773 = 180"). The row acts at two choices: after "X + Y" (" =", two
  steps, or " -", one line) and after "= X+Y" (" -", go on, or "\n",
  stop). It helps at the second and harms at the first. With "To" x1.25
  the one-line form is already there, so only the help shows. A row edit
  cannot tell the two places apart. Use " -" only with the "To" row.
- Other checks of the three rows against the two rows: sums of four
  numbers and products (seed 3) 188/216/165 against 188/218/165; word2
  seed 6 206 against 207 of 227 well-posed; story loss 2.4636 against
  2.4633 (x1.2: 2.4648).
- A flaw in the earlier reply checks (sections 17 and 18): they used
  `evalmath.prompt_ids`, which adds "Answer with only the final number."
  to the question. The replies were short and could not show changes in
  normal text. `replycheck.py` uses a plain chat turn (thinking off, 256
  tokens, 30 prompts, many with hyphens, ranges and lists). Same reply as
  the base model:

| Edit | same of 30 | mean shared start (words) |
|---|---|---|
| "To" x1.25 | 23 | 57.5 |
| " down" x1.1 | 27 | 67.3 |
| "To" x1.25 + " down" x1.1 | 23 | 55.4 |
| + " -" x1.1 | 20 | 51.3 |

  The changes are fluent text, not damage. "To" starts replies with "To
  ..." ("To calm down before an exam, ...") and writes "To-Do List" for
  "to-do list". " down" adds the word ("cools down", "Break down the
  exam"). " -" writes "9am - 5pm" for "9am-5pm" and " - " for an en dash.
- Saved as `models/Qwen3-0.6B-to125-down110-minus110` (three rows;
  git-ignored). From disk: logit difference 0.0; seed 8 hard set 541.

## 20. Audit of these notes (independent read-only check, `runs/` against the text)

An agent with a fresh context checked the numbers in sections 1-19 against
the result files and the code. Most traced numbers match (sections 10, 11,
13, 17, the seed 8, 11 and 12 tables of section 19). The problems found,
and what was done:

- Significance (two-proportion or paired McNemar tests on the saved
  results):
  - "To" x1.25: seed 3 90 to 105 of 120, p = 0.013. Seed 8, two rows: 443
    to 538 of 600, discordant pairs 109 against 14, p about 2e-19. Real.
  - " down" x1.1: word2 seed 6 179 to 208 of 227 well-posed, p about 1e-4
    (counts only). On the hard set seed 3 (20 word problems): 90 to 91,
    p = 0.88.
  - " -" x1.1 on top of the two rows: seed 8 538 to 541, pairs 6 against 3,
    p = 0.51 (no gain); seed 12 mixed 266 to 288, p = 0.0007, but still
    below the base (293 against 288, p = 0.24). It is a repair, not a gain.
  - Heads 12-15 x1.5 (section 10): 268 to 284 of 360, p = 0.16. Not
    significant; the Observations text is corrected.
- The prompt confound: every hard-set and GSM8K prompt says "Answer with
  only the final number." The "To" row makes the model ignore that request
  and work in steps. Without a run on a neutral prompt we cannot say the
  edit adds more than a better prompt. Tested next (`promptcheck.py`,
  `gsm8k.py --mode plain`).
- Lost and missing result files: `wordcheck.py`, `mixedcheck.py` named
  their output by seed only, so the section 19 run overwrote
  `runs/wordcheck_s6.json` of section 18. Both now take `--name`. Some
  numbers have no file at all (section 15 product runs, section 17 forced
  prefixes and token counts, the logit gaps of sections 18 and 19, the
  section 19 seed 8 and 11 data, which are in the session scratchpad only).
  Treat those as single-run results.
- Truncation: `tradeoff.py` did not record cut replies; it does now. On
  seed 8 the base model has 14 replies cut at 768 tokens (11 of them
  mul3x2 repetition loops); the two-row edit removes 9 of them, so part of
  the mul3x2 gain (39 to 85) is loop avoidance.
- word2 definitions: "well-posed" was "answer is not 0", which also drops
  well-posed questions with answer 0. Truly ill-posed (A * B < C): 21% on
  seed 5, 19% on seed 6. Fixed in `wordcheck.py`. In section 19 the
  ill-posed questions are scored against 0, not counted as wrong.
- Probe leakage (section 15): `mul2x2(3000, 2)` and `mul2x2(1000, 3)` share
  350 products, and the "2000 new products" include dev products.
- bf16 noise in the reply check: 3 of the 7 replies that "To" x1.25 changed
  split at a token that is not "To" (after 143 to 172 words). The row edit
  cannot change such a choice directly; the untied head and the batch give
  slightly different bf16 logits. Part of each "same of 30" count is noise.
- Smaller fixes: the alpha 4 steering range is 94 to 100%; the counter-set
  condition is a fraction above 0.5; of the 38 right "round to the nearest"
  replies 20 have a fraction below 0.5, and 18 do not.
- Not yet tested: " down" on questions that need rounding up ("how many
  bags are needed"); more than one template per question kind.

## 21. Row scaling against an additive logit bias (agent run, `runs/agent_bias/`)

Is "To" x1.25 the same as a sampling-time logit bias (logit + b, as many
APIs allow)? Base model, seed 3 tradeoff sets, greedy, 768 tokens. The bias
was put into the model as an output-head bias that is +b at id 1249 only.

- The "To" logit (float32, teacher forced): at the first reply position it
  is large and positive (mean 23.9 on hard questions, 17.2 on chat), so
  x1.25 adds about +6 there. In the middle of replies its median is near 0
  (-0.10 math, 0.23 chat; positive in about half of the positions), so
  x1.25 adds about +0.24 on average and lowers "To" where the logit is
  negative. The row edit gates itself; a bias adds the same b everywhere.

| Edit | add4 | mul2x2 | mul3x2 | of 700 | math tokens | chat same of 30 | chat shared start |
|---|---|---|---|---|---|---|---|
| none | 115 | 196 | 58 | 369 | 57.3 | 30 | 72.1 |
| "To" x1.25 | 190 | 220 | 170 | 580 | 108.1 | 23 | 57.5 |
| bias +4 | 190 | 206 | 111 | 507 | 76.3 | 25 | 59.4 |
| bias +6 | 190 | 217 | 171 | 578 | 112.3 | 20 | 51.1 |
| bias +8 | 190 | 219 | 174 | 583 | 119.8 | 10 | 33.0 |

- On math, bias +6 (the mean extra logit of x1.25 at the reply start)
  matches x1.25 (single greedy run; differences of about 3 per set are
  noise). On chat, x1.25 is better: bias +6 changes 10 replies, x1.25 7.
  With +8 the model puts "To" in the middle of replies ("To shore, to
  shore", "**Toaster**").
- Practical: where weights cannot be changed (an API, llama.cpp
  `--logit-bias`), a bias of about +5 to +6 on "To" gives the math gain,
  with somewhat more drift in chat. The row edit is the better form when
  the weights can be written.
- The first-position argmax counts predict how many replies start with
  "To" (223 predicted, 213 measured on mul2x2), so a factor can be chosen
  from one forward pass, without a generation sweep (not yet used).

## 22. GSM8K: no gain from the row edits (`gsm8k.py`, `runs/gsm8k.json`)

GSM8K test set (1319 questions, all with whole-number answers), "direct"
prompt (with "Answer with only the final number."), greedy, thinking off,
1024 tokens, batches of 24. Paired sign test against the base model:

| Model | right of 1319 | gained / lost | p | replies starting "To" | mean tokens | cut |
|---|---|---|---|---|---|---|
| base | 707 (53.6%) | - | - | 47 | 160.8 | 4 |
| three rows | 695 (52.7%) | 103 / 115 | 0.46 | 664 | 164.0 | 8 |
| "To" x1.25 | 682 (51.7%) | 90 / 115 | 0.09 | 664 | 162.0 | 5 |

- The edit changes the style (47 to 664 "To" starts) but not the length or
  the accuracy: on word problems the base model already works in steps
  (161 tokens on average) in spite of the "only the number" request. The
  "To" gain of sections 17-19 appears only where the base model answers in
  one shot (bare sums and products). It is not a general math gain.
- Two batches hit a GPU out-of-memory error during the run (other jobs on
  the machine) and were run again after a pause.
- Open: the same comparison with the "plain" prompt (no instruction).

## 23. The prompt explains the "To" gain (`gsm8k.py --mode plain`, `promptcheck.py`, `runs/gsm8k_plain.json`, `runs/promptcheck_s8.json`)

The "plain" prompt is the question alone: no "Answer with only the final
number." (thinking off, greedy).

- Hard set, seed 8, 600 questions: the base model with the plain prompt
  gets 537 (mul3x2 99, add4 91, word2 53, linear2 100, mixed 99, sub_big
  95). With the direct prompt: base 443, two rows 538, three rows 541. The
  row edits give the same score as dropping the request for only the
  number: they write that prompt change into the weights. (The edited
  model with the plain prompt was not run; the run was stopped.)
- GSM8K, plain prompt: base 786 of 1319 (59.6%, 236.6 tokens, 12 cut),
  three rows 774 (58.7%, 233.8 tokens, 15 cut). The prompt is worth +79
  questions for the base model; the edit gives nothing on top.
- Conclusion: the output-row edits do not add math skill. They undo a
  prompt that asks for a bare number. They are useful only where the
  prompt cannot be changed.

## 24. Partial agent results (runs stopped when the machine froze; `runs/agent_carry/`, `runs/agent_think/`, `runs/agent_mul3/`)

Read from the saved files and checked; the runs did not finish.

- Carry errors of two-digit products (`mulfix.score`, teacher forced; dev
  seed 2, 1000 products, base 0.649 all digits; test: 1500 new products
  without dev products, base 0.669):
  - Attention temperature (a per-head factor after `q_norm`; scaling
    `q_proj` rows does nothing in Qwen3, because `q_norm` renormalizes each
    head). The attention entropy changes as intended (head 20.12 x0.75:
    entropy 1.74 to 2.13). Single heads on dev: at most +0.015 (20.12 x0.5,
    20.13 x0.5 or x0.75, p 0.06 to 0.13). On the test set: 20.12 x0.75 15
    gained / 15 lost; 20.13 x0.75 30/22 (p = 0.33); 20.12 x0.5 32/26
    (p = 0.51). No gain. Sharper attention (x1.5, x2) hurts.
  - Four heads together (20.12, 20.13, 25.15, 26.6 x0.75): dev +0.023
    (45/22, p = 0.007). Not run on the test set.
  - Running a block twice: blocks 20 to 24 cost 0.03 to 0.07; block 27
    breaks the output (0.061). Swapping neighbour blocks 19 to 26 costs
    0.2 to 0.6. `q_norm` weight of a whole block x0.9 or x1.1: within
    +-0.004.
  - Verdict so far: no tested training-free edit fixes the carry errors.
- Thinking mode (hard set seed 8, the 72 questions both runs share, 4096
  tokens): base 59, three rows 60 (7 gained, 6 lost, p = 1). About 1700
  tokens per question for both. With thinking on, the row edits do nothing.
  The tok/s numbers of the two runs (531 against 246) come from different
  batch sizes and machine load and are not comparable.
- mul3x2 with the three-row model (seed 21, 300, direct prompt): 240 right.
  By reply style: decomposition (333 x 50 + 333 x 8) 224 of 237 right;
  LaTeX long-multiplication tables (`\begin{array}`) 0 of 30; loops or cut
  replies 0 of 7; one-shot 15 of 19. Half of the errors are the table
  layout. With the plain prompt the base model gets 99 of 100 mul3x2
  (section 23), so this failure belongs to the direct prompt.
- Steering vectors (`runs/agent_steerbake/screen.log`): only a first
  screen of how often replies start with "To"; no accuracy yet.

## 25. ExPO: weights from the instruct-base difference (agent run, provisional; `runs/agent_expo/`)

New weights theta = theta_I + alpha * (theta_I - theta_B), with theta_B from
Qwen3-0.6B-Base (downloaded with approval, `models/Qwen3-0.6B-Base`; same
keys and shapes, relative difference about 13-14% per matrix). The instruct
config, tokenizer and stop token are kept. "Timber": before extrapolating,
the small singular values of each delta matrix are shrunk (lam 0.5).

- Screen (dev): teacher-forced products, answer logprob and story loss move
  monotonically with alpha. Toward the base model (alpha -0.3): products
  0.649 to 0.709, story loss 2.446 to 2.352. Amplifying (alpha +0.2):
  products 0.612, story loss 2.671. The chat replies still stop.
- Hard set seed 8, plain prompt, 600 questions (seed 9 was used to choose
  the variants; seed 8 is new):

| Variant | right | gained / lost | p | word2 | sub_big | mean tokens |
|---|---|---|---|---|---|---|
| base | 535 | - | - | 51 | 95 | 157 |
| alpha +0.2 | 541 | 25 / 19 | 0.45 | 67 | 92 | 142 |
| Timber lam 0.5, alpha +0.2 | 552 | 32 / 15 | 0.019 | 72 | 94 | 144 |
| alpha -0.3 | 522 | 18 / 31 | 0.085 | 54 | 79 | 185 |
| base + "To" x1.25 | 535 | 7 / 7 | 1.0 | 49 | 95 | 152 |

- The gain of Timber +0.2 is almost all in word2 (51 to 72; well-posed 43
  to 58 of 76). It is the first edit with computed (not scaled) weights
  that passes a paired test on new questions, but three variants were
  tested on seed 8 and word2 is a flawed template (section 18). GSM8K is
  the deciding test (running).
- alpha < 0 helps the teacher-forced products but hurts generation
  (sub_big 95 to 79, longer replies). The "To" row adds nothing with the
  plain prompt, as in section 23.

## Observations

A running list of things noticed during the work, with the evidence for each.
Items marked "estimate" or "idea" are not measured.

### Inside the model

- Blocks 0 to 2 are needed for any output: block 0 needs its attention,
  blocks 1 and 2 need their MLPs (section 3).
- Block 20 attention is the most important part for arithmetic, and the MLPs
  of blocks 19 to 24 follow (section 3). Within the block 20 MLP, 16 of 3072
  neurons carry much of the effect (section 4).
- The block 20 math neurons are specific to math: zeroing the top 16 raises
  the loss on plain stories by 0.002 nats per token (2.4606 to 2.4630), less
  than a random set of 64 (0.004 to 0.006), while the answer logprob falls
  from -1.87 to -4.38 (`runs/text_check1.json`).
- Block 27 MLP makes the end-of-turn token. Without it the model writes
  `<|endoftext|>` and goes on with invented "Human:" turns.
- Block 3 attention is linked to how the model follows "only the final
  number": without it the replies are shorter and the word problems fail.
- Removing a part can look free on the math test and still cost general
  language: block 8 keeps 20/20 but raises the story loss from 2.461 to 2.566.
  Each surgery needs the text check (`textcheck.py`).
- Thinking mode used 14.5 times more tokens and was not better on this test.
- The first-order (gradient) estimate of a zeroing effect ranks the top
  neurons well, but its size is about 2.5 times too small for 16 neurons, and
  for the "bottom" neurons of block 21 it has the wrong sign (section 6).
- Math in block 20 attention sits in a few heads: heads 12 to 15 (key-value
  heads 6 and 7) are needed for math and not for stories (section 5).
- Heads 12 to 15 of block 20 scaled by 1.5 raised the hard math set by 4.4
  points over three sets (section 10), but the gain is not significant
  (268 to 284 of 360, p = 0.16), it is mostly a reply-style shift (section
  17), and it lowers products (section 15). It is not a skill gain. Scaling the math neurons
  of block 20 did not help, although it raised the answer logprob on the
  easy set: a better logprob on easy questions does not mean better answers
  on hard ones.
- The addition circuit, as far as measured: block 20 attention fetches the
  operand digits (heads 12 to 15 and 8); the MLPs of blocks 20 and 21 hold
  the operand digits; block 21 MLP neurons code the sum digit as waves of
  periods 10, 5, 3.3, 2.5 and 2; block 22 and 23 MLP neurons code it one-hot;
  the digit is the top token after block 23 (sections 9, 12, 13).
- 16 wave neurons of block 21 are a narrow point: without them the hard set
  falls from 90 to 19 of 120, and the story loss does not change (section
  13).
- Two places write answer digits. Block 21 MLP writes the units digit of a
  sum (64 neurons are enough to write any digit). Block 25 MLP holds a
  one-hot digit code that is the same for the tens digit of a sum and the
  units digit of a product; 64 of its neurons, set with values from sums
  alone, write any digit into either task (section 14).
- A weight edit that helps one skill can hurt another that the test set
  does not cover well: the block 20 head edit lifts sums of four numbers
  (+4 of 20) and lowers two-digit products (-8.8 points), and the hard set
  missed the loss (section 15). Each edit needs a check on each kind of task,
  with enough questions where the base model is not near 0 or 100%.
- The two-digit product errors are confident carry errors, and no single
  scale of a block, an MLP or the digit code fixes them (section 15).
- Products use the same entry point as sums (block 20 attention) but need
  more blocks after it: the tens digit forms over MLPs 21 to 25, the
  hundreds digit in 25 to 27. Late heads copy work from the previous answer
  position, and part of the next digit is ready one step early (section 16).
- The largest math lever in this model is the reply style, not the
  arithmetic circuit: with "To" at the reply start the model works step by
  step and is right about 90% of the time on sums of four numbers, against
  12 to 18% for one-shot replies. Scaling one row of the output head (the
  token "To", x1.25) raises the hard set from 90 to 105 of 120 on new
  questions, with almost no cost on stories (+0.0015). In a plain chat
  turn it changes 7 of 30 replies, mostly to a "To ..." start (section 19).
- Two rows can interact. "To" x1.25 brings a reply form ("P + C - Q = " in
  one line) where the model often stops after P + C; the row of " -" x1.1
  repairs that, but alone it harms the base model (section 19).
- Single choices in the reply decide many errors, and a small change in
  one output row can move them: " down" x1.1 after " round" lifts word
  problems with a remainder by 13 points on new questions, and leaves
  questions that ask for the nearest number alone (section 18). The method:
  find the token where right and wrong replies split, measure the logit
  gap there, and scale that token's row just enough.
- An edit can look like a skill gain and be a style change: check the
  replies, not only the score (section 17 against section 10).
- Steering needs a larger push than the class mean: at alpha 1 block 25
  moves 34 to 44% of answers, at alpha 4 it moves 94 to 100% (section 14).
- Block 20 heads 12 to 15 and 8 are digit-routing heads: each one fetches an
  operand digit by place value for the digit being written (section 9).
- bf16 is a noise source for the measurements: batch and serial runs give
  different greedy replies at near ties, and single-question logprobs differ
  by up to 0.4 (section 7).

### Speed and MLX

- Base speed, one stream, bf16, M5 Pro, stock MLX 0.32.4: generation about
  186 to 198 tok/s, prompt about 2300 tok/s, peak memory 1.26 GB (1.57 GB with
  thinking, from the longer KV cache).
- Each generated token reads about 1.19 GB of weights (the tied embedding is
  also the output head), so 190 tok/s is about 226 GB/s of weight reads.
  Generation of one stream is therefore limited by memory bandwidth (estimate
  from the numbers above; the chip limit was not checked).
- Batch evaluation (section 7) made the tests 6 to 8 times faster: one
  weight read per step now serves all questions. A sweep of 57 runs drops
  from about 6 minutes to about 1 minute (estimate from the per-run times).
- Parallel processes work (each needs about 1.3 GB), but they slow each
  other. A batch of 120 questions with up to 512 new tokens holds a large KV
  cache (about 8 GB by estimate: 120 x 600 positions x 28 blocks x 8 heads x
  128 x 2 x 2 bytes); two such processes at once failed with a Metal
  out-of-memory error. Run large batches one at a time, or split them. Speed numbers must come from runs alone, with alternating order: in
  the first sweep the base run (first in the process) showed 190 tok/s and
  later runs 212 tok/s, which is too large a difference for one block.
- The patches in our MLX fork (`mlx-tiny-train`) target head dim 32 and small
  models. Qwen3-0.6B has head dim 128, which stock MLX already runs with its
  fused attention, so the fork gives no expected gain here (not measured).
- Idea: a 4-bit model reads about a quarter of the bytes per token, so
  generation could be much faster (estimate). Surgery on packed 4-bit weights
  is harder: a column cannot be zeroed in place without unpacking the group.
- Weight surgery costs almost nothing: zeroing columns builds one new matrix,
  and `restore()` puts the old one back.
