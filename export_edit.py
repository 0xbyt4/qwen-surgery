"""Write an edited copy of the model to disk, as a normal mlx-lm model folder.

The edit (see surgery.apply_spec) is applied to the loaded model, and all
weights are saved to <out>/model.safetensors. The config and tokenizer files
are copied. The new folder loads with mlx_lm.load like the original, and the
script checks that its logits equal those of the in-memory edit.

Example:
    uv run python export_edit.py --edit "head:20:12,13,14,15*1.5" \
        --out models/Qwen3-0.6B-mathheads15
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

from evalmath import DEFAULT_MODEL, prompt_ids
from surgery import Surgeon, apply_spec

COPY = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--edit", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src, out = Path(a.model), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new folder")

    model, tok = load(str(src))
    apply_spec(Surgeon(model), a.edit)
    weights = dict(tree_flatten(model.parameters()))
    out.mkdir(parents=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights, {"format": "mlx"})
    for name in COPY:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    if not model.args.tie_word_embeddings:
        # The edit untied the output head (surgery.scale_output_rows).
        config = json.loads((out / "config.json").read_text())
        config["tie_word_embeddings"] = False
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "EDIT.json").write_text(
        json.dumps({"base": str(src), "edit": a.edit}, indent=2) + "\n"
    )

    x = mx.array([prompt_ids(tok, "What is 47 + 38?", "direct")])
    want = model(x)
    saved, _ = load(str(out))
    diff = mx.abs(saved(x) - want).max().item()
    print(f"wrote {out} ({len(weights)} tensors); max logit difference {diff}")
    if diff != 0:
        raise SystemExit("the saved model does not match the edit")


if __name__ == "__main__":
    main()
