"""Write changes into the weights of a loaded mlx-lm Qwen3 model, and undo them.

A head or an MLP neuron is removed by zeroing its input columns in the output
projection (o_proj for heads, down_proj for neurons). The block then adds
nothing from that head or neuron to the residual stream. The other parts of
the forward pass (and the KV cache) do not change. Scaling the same columns
by a factor other than 0 makes the part write more (or less) into the
residual stream.
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx import nn


class Surgeon:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.saved: dict[tuple[int, str], mx.array] = {}
        attn = model.model.layers[0].self_attn
        self.n_heads = attn.n_heads
        self.head_dim = attn.o_proj.weight.shape[1] // self.n_heads
        self.n_neurons = model.model.layers[0].mlp.down_proj.weight.shape[1]

    def _linear(self, layer: int, name: str) -> nn.Linear:
        block = self.model.model.layers[layer]
        return block.self_attn.o_proj if name == "o_proj" else block.mlp.down_proj

    def scale_columns(
        self, layer: int, name: str, columns: list[int], factor: float
    ) -> None:
        lin = self._linear(layer, name)
        self.saved.setdefault((layer, name), lin.weight)
        scale = mx.ones(lin.weight.shape[1], dtype=mx.float32)
        scale[mx.array(columns)] = factor
        lin.weight = (lin.weight * scale[None, :]).astype(lin.weight.dtype)

    def zero_columns(self, layer: int, name: str, columns: list[int]) -> None:
        self.scale_columns(layer, name, columns, 0.0)

    def scale_heads(self, layer: int, heads: list[int], factor: float) -> None:
        d = self.head_dim
        cols = [h * d + i for h in heads for i in range(d)]
        self.scale_columns(layer, "o_proj", cols, factor)

    def scale_neurons(self, layer: int, neurons: list[int], factor: float) -> None:
        self.scale_columns(layer, "down_proj", neurons, factor)

    def zero_heads(self, layer: int, heads: list[int]) -> None:
        self.scale_heads(layer, heads, 0.0)

    def zero_neurons(self, layer: int, neurons: list[int]) -> None:
        self.scale_neurons(layer, neurons, 0.0)

    def zero_attention(self, layer: int) -> None:
        self.zero_heads(layer, list(range(self.n_heads)))

    def zero_mlp(self, layer: int) -> None:
        self.zero_neurons(layer, list(range(self.n_neurons)))

    def restore(self) -> None:
        for (layer, name), w in self.saved.items():
            self._linear(layer, name).weight = w
        self.saved.clear()


def apply_spec(s: Surgeon, spec: str) -> None:
    """Apply one edit written as text. Several edits join with "+".

    attn:L, mlp:L, block:L       zero a whole part of block L
    head:L:H1,H2,...             zero heads of block L
    neurons:L:J1,J2,...          zero MLP neurons of block L
    nfile:PATH:KEY               zero the neurons of results[KEY] in a
                                 neurons.py output file
    A head, neurons or nfile edit ending in "*F" scales the parts by F
    instead of zeroing them, for example head:20:12,13*2.
    """
    for part in spec.split("+"):
        _apply_one(s, part)


def _apply_one(s: Surgeon, spec: str) -> None:
    spec, _, factor_text = spec.partition("*")
    factor = float(factor_text) if factor_text else 0.0
    kind, *rest = spec.split(":")
    if kind == "attn":
        s.zero_attention(int(rest[0]))
    elif kind == "mlp":
        s.zero_mlp(int(rest[0]))
    elif kind == "block":
        s.zero_attention(int(rest[0]))
        s.zero_mlp(int(rest[0]))
    elif kind == "head":
        s.scale_heads(int(rest[0]), [int(x) for x in rest[1].split(",")], factor)
    elif kind == "neurons":
        s.scale_neurons(int(rest[0]), [int(x) for x in rest[1].split(",")], factor)
    elif kind == "nfile":
        d = json.loads(Path(rest[0]).read_text())
        neurons = d["results"][rest[1]]["neurons"]
        s.scale_neurons(d["config"]["layer"], neurons, factor)
    else:
        raise ValueError(f"unknown edit: {spec}")
