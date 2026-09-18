"""The two charts that go on the deck.

  plot()                semantic z vs factor-implied beta z, 9 labelled points
  salience_valence_2x2()  the pair, as a quadrant map

DESIGN NOTES, so these do not get "improved" into something worse:

One series, so no legend -- the title names what the points are. All nine points carry
direct labels because identity IS the message here; with 9 points a legend would be a
lookup table. Marks are one colour; the two or three largest disagreements are drawn in
the status red AND annotated in text, so identity never rests on colour alone.

The 45-degree line is the agreement locus: both axes are z-scores of the same nine
names, so a company on the line is ranked identically by text and by market. Distance
from that line, not distance from the origin, is the finding.

Recessive grid, thin marks, text in ink rather than series colour.

    python -m src.reconcile.scatter
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import paths

IN = paths.RECONCILIATION
INDEX = paths.AIA_INDEX
FIGDIR = paths.FIGURES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
SERIES = "#2a78d6"
CRITICAL = "#d03b3b"
GRID = "#e4e3df"


def _style(ax, title, xlabel, ylabel, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=16 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_2,
                fontsize=9.5, va="bottom")


# Placements tried in order around a point: above, below, left, right.
_SLOTS = [(0, 12, "center"), (0, -17, "center"), (-13, -4, "right"), (13, -4, "left")]


def _label_offsets(df: pd.DataFrame, xcol: str, ycol: str, span: float) -> dict:
    """Place each label around its point so neighbours do not overlap.

    Nine points is too few to justify a layout solver and too many to hand-place.
    Points are grouped into clusters of near-neighbours and each member takes the next
    slot in the rotation, so PG and WM -- which sit almost on top of each other near
    the origin, both being the "no AI here" names -- end up above and below rather
    than stacked. Deterministic, so the chart does not move between runs.
    """
    xs, ys = df[xcol], df[ycol]
    near_x, near_y = span * 0.18, span * 0.16

    neighbours = {a: {b for b in df.index
                      if b != a and abs(xs[a] - xs[b]) < near_x
                      and abs(ys[a] - ys[b]) < near_y}
                  for a in df.index}

    out, used = {}, {}
    # Highest first, so the top of a cluster keeps the natural label-above position.
    for t in sorted(df.index, key=lambda i: -ys[i]):
        taken = {used[n] for n in neighbours[t] if n in used}
        slot = next((i for i in range(len(_SLOTS)) if i not in taken), 0)
        used[t] = slot
        out[t] = _SLOTS[slot]
    return out


def plot(df: pd.DataFrame | None = None, n_flag: int = 3, out: Path | None = None) -> Path:
    """Labelled scatter: semantic score vs factor-implied beta, both as z-scores."""
    df = pd.read_parquet(IN) if df is None else df
    out = out or FIGDIR / "reconciliation_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    flagged = set(df["gap"].abs().nlargest(n_flag).index)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)

    lim = max(df["semantic_z"].abs().max(), df["beta_z"].abs().max()) * 1.25
    ax.plot([-lim, lim], [-lim, lim], color=GRID, linewidth=1.5, zorder=1)
    ax.text(lim * 0.97, lim * 0.90, "text and market agree", color=INK_2, fontsize=8.5,
            ha="right", style="italic")
    ax.axhline(0, color=GRID, linewidth=1, zorder=1)
    ax.axvline(0, color=GRID, linewidth=1, zorder=1)

    slots = _label_offsets(df, "semantic_z", "beta_z", lim)
    for t, r in df.iterrows():
        hit = t in flagged
        x, y = r["semantic_z"], r["beta_z"]
        ax.scatter(x, y, s=130 if hit else 95,
                   color=CRITICAL if hit else SERIES,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ldx, ldy, lha = slots[t]
        ax.annotate(t, (x, y), xytext=(ldx, ldy), textcoords="offset points", ha=lha,
                    color=INK, fontsize=10,
                    fontweight="bold" if hit else "normal", zorder=4)
        if hit:
            # Text, not colour alone, carries the "this one diverges" meaning.
            side = "text > market" if r["gap"] > 0 else "market > text"
            # Near an edge a centred annotation overflows the axes; anchor it inward.
            ha, dx = ("center", 0)
            if x > lim * 0.55:
                ha, dx = "right", 14
            elif x < -lim * 0.55:
                ha, dx = "left", -14
            ax.annotate(f"{side}  {r['gap']:+.1f}sd", (x, y),
                        xytext=(dx, 14 if ldy < 0 else -20),
                        textcoords="offset points", ha=ha,
                        color=CRITICAL, fontsize=8.5, zorder=4)

    _style(ax, "Semantic AI association vs factor-implied AI beta",
           "AI-association index  (cross-sectional z)",
           "AI beta  (cross-sectional z)",
           f"Nine names. Distance from the diagonal is the disagreement; "
           f"the {n_flag} largest are marked.")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def salience_valence_2x2(idx: pd.DataFrame | None = None, quarters: int = 4,
                         out: Path | None = None) -> Path:
    """The pair as a quadrant map -- what AIA collapses into one number.

    AIA = salience x valence, so AIA near zero means either "no AI coverage" or
    "balanced AI coverage". Those are different companies and this chart separates
    them: the x axis is how much, the y axis is which direction.
    """
    idx = pd.read_parquet(INDEX) if idx is None else idx
    out = out or FIGDIR / "salience_valence_2x2.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    periods = sorted(idx["period"].unique())[-quarters:]
    d = (idx[idx["period"].isin(periods)].groupby("ticker")
         .agg(salience=("salience_shrunk", "mean"), valence=("valence_shrunk", "mean")))

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)
    xmid = d["salience"].median()
    ax.axvline(xmid, color=GRID, linewidth=1.5, zorder=1)
    ax.axhline(0, color=GRID, linewidth=1.5, zorder=1)

    xlo, xhi = -d["salience"].max() * 0.06, d["salience"].max() * 1.22
    ylo, yhi = d["valence"].min() - 0.25, d["valence"].max() + 0.30
    for x, y, ha, va, txt in [
        (xhi, yhi, "right", "top", "engaged, positive\nAI as opportunity"),
        (xlo, yhi, "left", "top", "quiet, positive"),
        (xhi, ylo, "right", "bottom", "engaged, negative\nAI as a threat"),
        (xlo, ylo, "left", "bottom", "quiet, negative"),
    ]:
        ax.text(x, y, txt, color=INK_2, fontsize=8.5, ha=ha, va=va, style="italic")

    slots = _label_offsets(d, "salience", "valence", d["salience"].max() or 1.0)
    for t, r in d.iterrows():
        ldx, ldy, lha = slots[t]
        ax.scatter(r["salience"], r["valence"], s=110, color=SERIES,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(t, (r["salience"], r["valence"]), xytext=(ldx, ldy),
                    textcoords="offset points", ha=lha, color=INK, fontsize=10, zorder=4)

    _style(ax, "How much a company discusses AI, and in which direction",
           "Salience  (AI share of the company's own disclosure)",
           "Valence  (direction of AI's impact on the company)",
           f"SEC filings, {periods[0]}..{periods[-1]}. Vertical line is the peer median.")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> int:
    plot()
    salience_valence_2x2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
