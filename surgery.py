"""Write changes into the weights of a loaded mlx-lm Qwen3 model, and undo them.

A head or an MLP neuron is removed by zeroing its input columns in the output
projection (o_proj for heads, down_proj for neurons). The block then adds
nothing from that head or neuron to the residual stream. The other parts of
the forward pass (and the KV cache) do not change.
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

    def zero_columns(self, layer: int, name: str, columns: list[int]) -> None:
        lin = self._linear(layer, name)
        self.saved.setdefault((layer, name), lin.weight)
        keep = mx.ones(lin.weight.shape[1], dtype=lin.weight.dtype)
        keep[mx.array(columns)] = 0
        lin.weight = lin.weight * keep[None, :]

    def zero_heads(self, layer: int, heads: list[int]) -> None:
        d = self.head_dim
        self.zero_columns(layer, "o_proj", [h * d + i for h in heads for i in range(d)])

    def zero_neurons(self, layer: int, neurons: list[int]) -> None:
        self.zero_columns(layer, "down_proj", neurons)

    def zero_attention(self, layer: int) -> None:
        self.zero_heads(layer, list(range(self.n_heads)))

    def zero_mlp(self, layer: int) -> None:
        self.zero_neurons(layer, list(range(self.n_neurons)))

    def restore(self) -> None:
        for (layer, name), w in self.saved.items():
            self._linear(layer, name).weight = w
        self.saved.clear()


def apply_spec(s: Surgeon, spec: str) -> None:
    """Apply one edit written as text:

    attn:L, mlp:L, block:L, head:L:H, neurons:L:j1,j2,...
    nfile:PATH:KEY  (the neurons of results[KEY] in a neurons.py output file)
    """
    kind, *rest = spec.split(":")
    if kind == "attn":
        s.zero_attention(int(rest[0]))
    elif kind == "mlp":
        s.zero_mlp(int(rest[0]))
    elif kind == "block":
        s.zero_attention(int(rest[0]))
        s.zero_mlp(int(rest[0]))
    elif kind == "head":
        s.zero_heads(int(rest[0]), [int(rest[1])])
    elif kind == "neurons":
        s.zero_neurons(int(rest[0]), [int(x) for x in rest[1].split(",")])
    elif kind == "nfile":
        d = json.loads(Path(rest[0]).read_text())
        s.zero_neurons(d["config"]["layer"], d["results"][rest[1]]["neurons"])
    else:
        raise ValueError(f"unknown edit: {spec}")
