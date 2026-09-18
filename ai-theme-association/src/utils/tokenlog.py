"""Token and cost accounting for every LLM call.

The brief asks for approximate token usage and which model was used per task.
Instrument this from the first call — reconstructing spend from the Console
afterwards is imprecise and loses the per-task attribution that is the actual ask.

Usage:

    from src.utils.tokenlog import logged

    resp = logged("classify:bulk", client.messages.create)(
        model="claude-haiku-4-5", max_tokens=512, messages=[...]
    )
"""

from __future__ import annotations

import csv
import functools
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "docs" / "token_log.csv"

# USD per 1M tokens (input, output). Verify against current Anthropic pricing
# before quoting totals in the write-up.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}

CACHE_READ_MULTIPLIER = 0.10  # cache reads bill at ~10% of the input rate
BATCH_MULTIPLIER = 0.50       # Batch API is 50% off

_HEADER = ["ts", "task", "model", "input", "output", "cache_read", "cache_write", "batch", "est_usd"]


def record(task: str, model: str, usage, *, batch: bool = False) -> float:
    """Append one call to the log and return its estimated USD cost."""
    if model not in PRICES:
        raise KeyError(f"No price for {model!r}; add it to PRICES before logging.")
    p_in, p_out = PRICES[model]

    n_in = getattr(usage, "input_tokens", 0) or 0
    n_out = getattr(usage, "output_tokens", 0) or 0
    n_cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    n_cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    # input_tokens excludes cached reads, so bill them separately rather than subtracting.
    cost = (n_in * p_in + n_cache_read * p_in * CACHE_READ_MULTIPLIER + n_cache_write * p_in + n_out * p_out) / 1e6
    if batch:
        cost *= BATCH_MULTIPLIER

    LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(_HEADER)
        w.writerow([int(time.time()), task, model, n_in, n_out, n_cache_read, n_cache_write, batch, round(cost, 6)])
    return cost


def logged(task: str, fn, *, batch: bool = False):
    """Wrap an Anthropic client method so every call is logged."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        resp = fn(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            record(task, kwargs.get("model", "unknown"), usage, batch=batch)
        return resp
    return wrapper


def summary() -> str:
    """Per-task, per-model totals. Paste into the appendix."""
    if not LOG.exists():
        return "no calls logged"
    import collections
    agg: dict[tuple[str, str], list[float]] = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    with LOG.open() as fh:
        for row in csv.DictReader(fh):
            k = (row["task"], row["model"])
            agg[k][0] += 1
            agg[k][1] += int(row["input"])
            agg[k][2] += int(row["output"])
            agg[k][3] += float(row["est_usd"])
    lines = [f"{'task':28} {'model':20} {'calls':>6} {'in':>10} {'out':>9} {'usd':>8}"]
    for (task, model), (n, ti, to, c) in sorted(agg.items()):
        lines.append(f"{task:28} {model:20} {n:6d} {ti:10d} {to:9d} {c:8.3f}")
    lines.append(f"{'TOTAL':49} {sum(v[0] for v in agg.values()):6d} "
                 f"{sum(v[1] for v in agg.values()):10d} {sum(v[2] for v in agg.values()):9d} "
                 f"{sum(v[3] for v in agg.values()):8.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
