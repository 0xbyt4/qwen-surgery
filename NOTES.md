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
- Writing a weight can improve a skill without training: heads 12 to 15 of
  block 20 scaled by 1.5 lift the hard math set by 4.4 points over three
  sets, mostly on sums of four numbers (section 10). Scaling the math neurons
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
- Steering needs a larger push than the class mean: at alpha 1 block 25
  moves 34 to 44% of answers, at alpha 4 it moves 96 to 100% (section 14).
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
