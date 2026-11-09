# Data

Both files come from other projects under the MIT license. They are copied
here without changes, so that the scripts run without a download.

| File | Source | License |
|---|---|---|
| `stories.jsonl` | The first 200 stories of the test split of [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) (stories written by language models), one JSON object per line with the story in `"text"` | MIT, as given on the dataset card |
| `gsm8k/test.jsonl` | The GSM8K test set, `grade_school_math/data/test.jsonl` of [openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT, Copyright (c) 2021 OpenAI |

The scripts use `stories.jsonl` only as plain text: for the story-loss check
(`textcheck.py`) and as text where new neurons must stay silent
(`carrybuild.py`). GSM8K is a side-task check (`gsm8k.py`, `carrybuild.py
--general`).
