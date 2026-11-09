"""Charts for the README, drawn from the run logs in runs/.

Each chart reads its numbers from a tracked log, so a new run of the same
script updates the chart. Every chart is written twice, for a light and a
dark page: docs/img/<name>-light.png and docs/img/<name>-dark.png.

Example:
    uv run --with matplotlib python plots.py
"""

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "img"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]  # blue, orange, aqua


THEMES = [
    Theme(
        "light",
        "#fcfcfb",
        "#0b0b0b",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
        ("#2a78d6", "#eb6834", "#1baf7a"),
    ),
    Theme(
        "dark",
        "#1a1a19",
        "#f0efec",
        "#c3c2b7",
        "#898781",
        "#2c2c2a",
        "#383835",
        ("#3987e5", "#d95926", "#199e70"),
    ),
]


@dataclass(frozen=True)
class Paired:
    label: str
    base: int
    edit: int
    n: int
    p: str


def results() -> list[Paired]:
    """Paired base/edit counts of the final circuit (carrybuild --general
    --side --gen)."""
    text = (RUNS / "carrymulti_w12_gen.log").read_text()
    m = re.search(
        r"generation \(direct, 1000 new products\): base (\d+) edit (\d+) "
        r"gained \d+ lost \d+ p (\S+?);",
        text,
    )
    if m is None:
        raise ValueError("no generation line in carrymulti_w12_gen.log")
    out = [
        Paired(
            "New products, prompt forms of the fit", int(m[1]), int(m[2]), 1000, m[3]
        )
    ]
    names = {
        "general phrasings": "Phrasings never used in the fit",
        "general gsm8k": "GSM8K",
        "general plain": "Plain chat prompts",
        "general word": "Word problems never used in the fit",
        "side hard_plain": "Other: mixed hard math set",
        "side three": "Other: X + Y - Z, 3-digit",
        "side sub_big": "Other: 5-digit minus 4-digit",
    }
    for key, label in names.items():
        m = re.search(
            rf"^{key}(?: \(\w+\))?: base (\d+) edit (\d+) of (\d+) "
            r"\(\+\d+ / -\d+, p (\S+)\)",
            text,
            re.M,
        )
        if m is None:
            raise ValueError(f"no line for {key} in carrymulti_w12_gen.log")
        out.append(Paired(label, int(m[1]), int(m[2]), int(m[3]), m[4]))
    return out


def probe() -> tuple[list[int], list[float], list[float]]:
    """Probe accuracy at the position that writes the hundreds digit: mean of
    the four operand digits, and the carry into the hundreds."""
    text = (RUNS / "opprobe_mul2x2.log").read_text()
    part = text.split("position writing the hundreds digit")[1]
    blocks = [int(b) for b in re.search(r"blocks \[([\d, ]+)\]", part)[1].split(",")]
    rows = {}
    for name in ["a_t", "a_u", "b_t", "b_u", "carry_h"]:
        rows[name] = [
            float(v)
            for v in re.search(rf"^\s+{name}\s+([\d. ]+)$", part, re.M)[1].split()
        ]
    ops = [
        sum(rows[k][i] for k in ["a_t", "a_u", "b_t", "b_u"]) / 4
        for i in range(len(blocks))
    ]
    return blocks, ops, rows["carry_h"]


def gate() -> dict[tuple[str, str], dict[int, float]]:
    """Operand readout accuracy on held prompt forms (mean argmax accuracy of
    the four digits), by block, fit and held set."""
    out: dict[tuple[str, str], dict[int, float]] = {}
    line = re.compile(
        r"^block (\d+) fit (\w+)\s+held (\w+)\s*:.*\| operands "
        r"acc/right fires/wrong fires: (.*)$",
        re.M,
    )
    for log in ["carrygate_b15_16.log", "carrygate.log"]:
        for m in line.finditer((RUNS / log).read_text()):
            acc = [float(v) for v in re.findall(r"\w_\w ([\d.]+)/", m[4])]
            out.setdefault((m[2], m[3]), {})[int(m[1])] = sum(acc) / len(acc)
    return out


def alpha() -> tuple[list[int], dict[str, list[float]], dict[str, float]]:
    """Teacher-forced accuracy (all four digits right) by flag scale alpha:
    after the last stage, and for the base model."""
    alphas = sorted(
        int(p.stem.split("_a")[1])
        for p in RUNS.glob("carrymulti_a*.log")
        if re.fullmatch(r"carrymulti_a\d+", p.stem)
    )
    pat = re.compile(
        r"all ([\d.]+) .*all four digits \{'direct': ([\d.]+), 'word': ([\d.]+)\}"
    )
    after: dict[str, list[float]] = {"fit": [], "direct": [], "word": []}
    base: dict[str, float] = {}
    for a in alphas:
        text = (RUNS / f"carrymulti_a{a}.log").read_text()
        b = pat.search(re.search(r"after none:.*$", text, re.M)[0])
        e = pat.search(re.search(r"after B4:.*$", text, re.M)[0])
        base = {"fit": float(b[1]), "direct": float(b[2]), "word": float(b[3])}
        for k, v in zip(after, e.groups(), strict=True):
            after[k].append(float(v))
    return alphas, after, base


def frame(t: Theme, size: tuple[float, float]):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(t.surface)
    ax.set_facecolor(t.surface)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=t.ink2, labelsize=10)
    ax.set_axisbelow(True)
    return fig, ax


def titles(fig, t: Theme, title: str, sub: str) -> None:
    fig.text(0.02, 0.965, title, color=t.ink, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.905, sub, color=t.ink2, fontsize=10.5, va="top")


def legend(fig, t: Theme, handles, y: float) -> None:
    leg = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, y),
        frameon=False,
        fontsize=10,
        ncol=len(handles),
        handlelength=1.6,
    )
    for x in leg.get_texts():
        x.set_color(t.ink)


def save(fig, t: Theme, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{t.name}.png"
    fig.savefig(path, facecolor=t.surface)
    plt.close(fig)
    return path


def plot_results(t: Theme, rows: list[Paired]) -> Path:
    fig, ax = frame(t, (11, 6.4))
    base, edit = t.series[0], t.series[1]
    h, gap = 0.36, 0.03
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (v, c) in enumerate([(r.base, base), (r.edit, edit)]):
            yy = y + (0.5 - j) * (h + gap)
            ax.barh(yy, v / r.n * 100, height=h, color=c)
            ax.text(
                v / r.n * 100 + 0.8,
                yy,
                f"{v}/{r.n}",
                va="center",
                color=t.ink2,
                fontsize=8.5,
            )
        ax.text(110, y, f"{change(r)}, p {r.p}", va="center", color=t.ink2, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.label for r in reversed(rows)], color=t.ink, fontsize=10)
    ax.set_xlim(0, 145)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]])
    ax.grid(axis="x", color=t.grid, linewidth=0.8)
    titles(
        fig,
        t,
        "Answers right, base model and with the carry circuit",
        "Greedy replies, same questions with and without the ~4,200 written neurons; "
        "p from a paired sign test",
    )
    legend(
        fig,
        t,
        [
            Patch(color=base, label="Base model"),
            Patch(color=edit, label="With the carry circuit"),
        ],
        0.86,
    )
    fig.subplots_adjust(left=0.30, right=0.99, top=0.78, bottom=0.07)
    return save(fig, t, "results")


def change(r: Paired) -> str:
    """ "better" or "worse" at p < 0.05, else "no clear change"."""
    if float(r.p) >= 0.05:
        return "no clear change"
    return "better" if r.edit > r.base else "worse"


def plot_summary(t: Theme, rows: list[Paired]) -> Path:
    """The three headline sets as vertical bars, sized for a post."""
    picked = [rows[0], rows[1], rows[4]]
    names = [
        "Prompt forms used in the fit",
        "Phrasings never used in the fit",
        "Word problems never used in the fit",
    ]
    fig, ax = frame(t, (12, 6.75))
    base, edit = t.series[0], t.series[1]
    w, gap = 0.34, 0.02
    for i, (r, name) in enumerate(zip(picked, names, strict=True)):
        for j, (v, c) in enumerate([(r.base, base), (r.edit, edit)]):
            x = i + (j - 0.5) * (w + gap)
            pct = v / r.n * 100
            ax.bar(x, pct, width=w, color=c)
            ax.text(
                x,
                pct + 1.5,
                f"{pct:.0f}%",
                ha="center",
                va="bottom",
                color=t.ink,
                fontsize=15,
                fontweight="bold",
            )
            ax.text(
                x,
                pct + 7.5,
                f"{v}/{r.n}",
                ha="center",
                va="bottom",
                color=t.ink2,
                fontsize=11,
            )
        ax.text(i, -6, name, ha="center", va="top", color=t.ink, fontsize=13)
        ax.text(
            i,
            -12,
            f"{change(r)}, p = {r.p}",
            ha="center",
            va="top",
            color=t.ink2,
            fontsize=11,
        )
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 112)
    ax.set_xticks([])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]], fontsize=11)
    ax.grid(axis="y", color=t.grid, linewidth=0.8)
    fig.text(
        0.06,
        0.95,
        "Hand-written carry neurons in Qwen3-0.6B, no training",
        color=t.ink,
        fontsize=19,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.06,
        0.885,
        "2-digit x 2-digit products answered right (greedy, "
        "same questions with and without the neurons)",
        color=t.ink2,
        fontsize=12,
        va="top",
    )
    leg = fig.legend(
        handles=[
            Patch(color=base, label="Base model"),
            Patch(color=edit, label="With ~4,200 written neurons"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.055, 0.85),
        frameon=False,
        fontsize=12,
        ncol=2,
    )
    for x in leg.get_texts():
        x.set_color(t.ink)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.18)
    return save(fig, t, "summary")


def plot_probe(t: Theme, data) -> Path:
    blocks, ops, carry = data
    fig, ax = frame(t, (9, 5))
    a, b = t.series[0], t.series[1]
    ax.plot(blocks, [v * 100 for v in ops], color=a, lw=2, marker="o", ms=6)
    ax.plot(blocks, [v * 100 for v in carry], color=b, lw=2, marker="o", ms=6)
    ax.text(
        blocks[-1] + 0.3,
        ops[-1] * 100,
        "operand digits",
        color=t.ink,
        va="center",
        fontsize=10,
    )
    ax.text(
        blocks[-1] + 0.3,
        carry[-1] * 100,
        "carry",
        color=t.ink,
        va="center",
        fontsize=10,
    )
    ax.set_xticks(blocks)
    ax.set_xlim(blocks[0] - 0.5, blocks[-1] + 2.6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]])
    ax.set_xlabel("block", color=t.ink2, fontsize=10)
    ax.grid(axis="y", color=t.grid, linewidth=0.8)
    titles(
        fig,
        t,
        "The model holds the operand digits, not the carry",
        "Linear probe accuracy at the position that writes the hundreds digit "
        "of a 2-digit product",
    )
    legend(
        fig,
        t,
        [
            Line2D(
                [], [], color=a, lw=2, marker="o", label="Four operand digits (mean)"
            ),
            Line2D([], [], color=b, lw=2, marker="o", label="Carry into the hundreds"),
        ],
        0.85,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.12)
    return save(fig, t, "probe")


def plot_gate(t: Theme, data) -> Path:
    fig, ax = frame(t, (9, 5))
    color = {"direct": t.series[0], "word": t.series[1]}
    style = {"multi": "-", "single": "--"}
    for (fit, held), acc in sorted(data.items()):
        x = sorted(acc)
        ax.plot(
            x,
            [acc[b] * 100 for b in x],
            color=color[held],
            ls=style[fit],
            lw=2,
            marker="o",
            ms=6,
        )
    blocks = sorted({b for acc in data.values() for b in acc})
    ax.axvline(19.5, color=t.axis, lw=1)
    ax.text(
        19.6,
        3,
        "block 20 attention\nfetches the operands",
        color=t.muted,
        fontsize=9,
        va="bottom",
    )
    ax.set_xticks(blocks)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]])
    ax.set_xlabel("block the readouts are fit at", color=t.ink2, fontsize=10)
    ax.grid(axis="y", color=t.grid, linewidth=0.8)
    titles(
        fig,
        t,
        "Operand readouts hold on new prompt forms only after block 20",
        "Accuracy of the operand digit readouts on held-out prompt forms "
        "(mean of four digits)",
    )
    legend(
        fig,
        t,
        [
            Line2D([], [], color=color["direct"], lw=2, label="Held direct questions"),
            Line2D([], [], color=color["word"], lw=2, label="Held word problems"),
            Line2D([], [], color=t.ink2, lw=2, ls="-", label="Fit on many forms"),
            Line2D([], [], color=t.ink2, lw=2, ls="--", label="Fit on one form"),
        ],
        0.85,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.12)
    return save(fig, t, "gate")


def plot_alpha(t: Theme, data) -> Path:
    alphas, after, base = data
    fig, ax = frame(t, (9, 5))
    names = {
        "fit": "Prompt forms of the fit",
        "direct": "Held direct questions",
        "word": "Held word problems",
    }
    for (k, label), c in zip(names.items(), t.series, strict=True):
        ax.axhline(base[k] * 100, color=c, lw=1.2, ls=":")
        ax.plot(alphas, [v * 100 for v in after[k]], color=c, lw=2, marker="o", ms=6)
        ax.text(
            alphas[-1] + 1,
            after[k][-1] * 100,
            label,
            color=t.ink,
            va="center",
            fontsize=10,
        )
    ax.set_xticks(alphas)
    ax.set_xlim(alphas[0] - 3, alphas[-1] + 19)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]])
    ax.set_xlabel("flag scale alpha", color=t.ink2, fontsize=10)
    ax.grid(axis="y", color=t.grid, linewidth=0.8)
    titles(
        fig,
        t,
        "Flag scale alpha: 32 is best, word problems stay below the base",
        "Products with all four digits right, teacher forced; dotted line = base model",
    )
    legend(
        fig,
        t,
        [
            Line2D([], [], color=t.ink2, lw=2, marker="o", label="With the circuit"),
            Line2D([], [], color=t.ink2, lw=1.2, ls=":", label="Base model"),
        ],
        0.85,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.12)
    return save(fig, t, "alpha")


def main() -> None:
    data = results(), probe(), gate(), alpha()
    for t in THEMES:
        for path in [
            plot_summary(t, data[0]),
            plot_results(t, data[0]),
            plot_probe(t, data[1]),
            plot_gate(t, data[2]),
            plot_alpha(t, data[3]),
        ]:
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
