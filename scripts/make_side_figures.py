"""Draw the side figures: context and secondary results, candidates for the write-up.

    python scripts/export_side_data.py --runs runs   # once, -> reports/*.csv
    python scripts/make_side_figures.py              # -> reports/figures/side/*.svg

Reads ``reports/agent_episodes.csv``, ``reports/agent_tool_calls.csv`` and
``reports/feature_signal.csv`` (written by ``scripts/export_side_data.py``), and draws in the
same style as ``scripts/make_figures.py``, whose helpers it reuses. Everything here is
descriptive and after the fact: no figure is a test, and none feeds a configuration.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from make_figures import GRID, INK, INK2, MUTED, SERIES, SURFACE, svg, text

OUT = Path("reports/figures/side")
EPISODES = Path("reports/agent_episodes.csv")
CALLS = Path("reports/agent_tool_calls.csv")
SIGNAL = Path("reports/feature_signal.csv")

GAME_LABEL = {"N-dev": "N-dev (Nipah)", "E": "E (EGFR transfer)"}
Z90 = 1.6449  # two-sided 90%, the interval every figure of the project uses


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_interval(values: list[float], ci: float = 0.9) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap interval of a mean (unpaired, fixed rng)."""
    rng = random.Random(0)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(10_000))
    tail = (1 - ci) / 2
    return (
        sum(values) / n,
        means[int(tail * len(means))],
        means[int((1 - tail) * len(means)) - 1],
    )


def bar(x, y, w, h, fill, rx=3):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}"/>'
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1, extra=""):
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}" {extra}/>'
    )


def dot_interval(sx, y, mean, low, high, colour):
    """One dot-and-whisker row, the form the main figures use."""
    return "".join(
        [
            line(sx(low), y, sx(high), y, colour, 2, 'stroke-linecap="round"'),
            line(sx(low), y - 6, sx(low), y + 6, colour, 2),
            line(sx(high), y - 6, sx(high), y + 6, colour, 2),
            f'<circle cx="{sx(mean):.1f}" cy="{y:.1f}" r="5" fill="{colour}" '
            f'stroke="{SURFACE}" stroke-width="2"/>',
        ]
    )


def legend(items, x, y):
    parts = []
    for label, colour in items:
        parts.append(bar(x, y - 9, 10, 10, colour, 2))
        parts.append(text(x + 16, y, label, 12, INK2))
        x += 16 + 7.2 * len(label) + 22
    return "".join(parts)


def save(path: Path, width: int, height: int, parts: list[str], title: str, desc: str) -> Path:
    path.write_text(svg(width, height, "".join(parts), title, desc), encoding="utf-8")
    return path


# --- 1. the opening move -----------------------------------------------------------------


def opening_figure(episodes, signal, path, width=880, min_n=10):
    auc = {(r["game"], r["feature"]): float(r["auc"]) for r in signal}
    groups = []
    for game in ("N-dev", "E"):
        by_column = defaultdict(list)
        for r in episodes:
            if r["game"] == game and r["first_sort_by"]:
                by_column[r["first_sort_by"]].append(float(r["hits"]))
        kept = sorted(
            ((c, v) for c, v in by_column.items() if len(v) >= min_n), key=lambda cv: -len(cv[1])
        )
        dropped = sum(len(v) for c, v in by_column.items() if len(v) < min_n)
        groups.append((game, [(c, len(v), *mean_interval(v)) for c, v in kept], dropped))

    left, right, top, row_h, gap = 300, 120, 84, 32, 26
    height = top + sum(len(items) * row_h + gap for _, items, _ in groups) + 56
    hi = max(item[4] for _, items, _ in groups for item in items) * 1.08
    x0, x1 = left, width - right

    def sx(v):
        return x0 + v / hi * (x1 - x0)

    parts = [
        text(24, 30, "The opening move: what the agent sorts on first", 16, INK, weight="600"),
        text(
            24,
            50,
            "Mean hits of the episode by the column of its first sorted query_pool call, "
            "all agent episodes, 90% bootstrap interval",
            12.5,
            INK2,
        ),
    ]
    for value in range(0, int(hi) + 1, 5):
        parts.append(line(sx(value), top - 8, sx(value), height - 50))
        parts.append(text(sx(value), height - 34, value, 11, MUTED, "middle"))
    parts.append(text(width - 24, top - 14, "feature AUC", 11.5, MUTED, "end"))
    y = top + 4
    for index, (game, items, dropped) in enumerate(groups):
        colour = SERIES[index]
        plural = "episode" if dropped == 1 else "episodes"
        note = f"  ({dropped} {plural} with a rarer opening left out)" if dropped else ""
        parts.append(text(24, y + 4, GAME_LABEL[game] + note, 12.5, colour, weight="600"))
        for column, n, mean, low, high in items:
            cy = y + 22
            parts.append(text(42, cy + 4, f"{column}  (n={n})", 12.5, INK2))
            parts.append(dot_interval(sx, cy, mean, low, high, colour))
            parts.append(text(sx(high) + 10, cy + 4, f"{mean:.1f}", 12, INK, weight="600"))
            if (game, column) in auc:
                parts.append(
                    text(width - 24, cy + 4, f"{auc[(game, column)]:.2f}", 12, INK2, "end")
                )
            y += row_h
        y += gap
    parts.append(
        text(
            24,
            height - 12,
            "Exploratory: models, conditions and trial blocks are mixed in every row, so a "
            "gap here is a correlation, not an effect.",
            11.5,
            MUTED,
        )
    )
    parts.append(text(x1, height - 34, "hits", 11.5, MUTED, "end"))
    return save(
        path,
        width,
        height,
        parts,
        "The opening move",
        "Mean hits by the agent's first sort column, on N-dev and on E.",
    )


# --- 2. cost against hits ----------------------------------------------------------------


def cost_figure(episodes, path, width=900):
    rows = [
        (model, condition)
        for model in ("deepseek", "mimo", "sonnet")
        for condition in ("A", "B1", "B2", "C")
        if any(r["model"] == model and r["condition"] == condition for r in episodes)
    ]
    cost, hits, count = {}, {}, {}
    for model, condition in rows:
        picked = [r for r in episodes if r["model"] == model and r["condition"] == condition]
        count[(model, condition)] = len(picked)
        cost[(model, condition)] = 100 * sum(float(r["cost_usd"]) for r in picked) / len(picked)
        for game in ("N-dev", "E"):
            values = [float(r["hits"]) for r in picked if r["game"] == game]
            if values:
                hits[(model, condition, game)] = sum(values) / len(values)
    total = sum(float(r["cost_usd"]) for r in episodes)

    top, row_h, bottom = 104, 30, 52
    label_w = 190
    c0, c1 = label_w, label_w + 250  # cost panel
    h0, h1 = c1 + 70, width - 40  # hits panel
    height = top + len(rows) * row_h + bottom
    cmax = max(cost.values()) * 1.15
    hmin = 10.0
    hmax = max(hits.values()) * 1.08

    def cx(v):
        return c0 + v / cmax * (c1 - c0)

    def hx(v):
        return h0 + (v - hmin) / (hmax - hmin) * (h1 - h0)

    parts = [
        text(24, 30, "Cheap models are enough here", 16, INK, weight="600"),
        text(
            24,
            50,
            f"Cost and hits per episode, by model and condition: {len(episodes)} agent "
            f"episodes cost ${total:.2f} in all",
            12.5,
            INK2,
        ),
        text(c0, top - 18, "US cents per episode", 12, INK2, weight="600"),
        text(h0, top - 18, "mean hits", 12, INK2, weight="600"),
        legend([("N-dev", SERIES[0]), ("E", SERIES[1])], h0 + 90, top - 18),
    ]
    for value in range(int(hmin), int(hmax) + 1, 5):
        parts.append(line(hx(value), top - 6, hx(value), height - bottom + 4))
        parts.append(text(hx(value), height - bottom + 20, value, 11, MUTED, "middle"))
    previous = None
    for i, (model, condition) in enumerate(rows):
        y = top + i * row_h
        if previous and previous != model:
            parts.append(line(24, y - 4, width - 24, y - 4, GRID))
        previous = model
        parts.append(
            text(
                label_w - 12,
                y + 15,
                f"{model} {condition}  (n={count[(model, condition)]})",
                12.5,
                INK2,
                "end",
            )
        )
        value = cost[(model, condition)]
        parts.append(bar(c0, y + 5, cx(value) - c0, 14, SERIES[2]))
        parts.append(text(cx(value) + 8, y + 16, f"{value:.1f}", 12, INK, weight="600"))
        for index, game in enumerate(("N-dev", "E")):
            if (model, condition, game) in hits:
                cx_hit = hx(hits[(model, condition, game)])
                parts.append(
                    f'<circle cx="{cx_hit:.1f}" cy="{y + 7 + 10 * index:.1f}" '
                    f'r="5.5" fill="{SERIES[index]}" stroke="{SURFACE}" stroke-width="2"/>'
                )
    parts.append(
        text(
            24,
            height - 12,
            "At the prices in config/prices.yaml when each block was played. Hits axis starts "
            "at 10. Sonnet was probed on N-dev only (trials 0-9).",
            11.5,
            MUTED,
        )
    )
    return save(
        path,
        width,
        height,
        parts,
        "Cost against hits",
        "Cost per episode and mean hits per game for each model and condition.",
    )


# --- 3. what the agent does each round ---------------------------------------------------

TOOL_GROUPS = [
    (
        "look at the pool",
        ("query_pool", "column_summary", "describe_campaign", "get_history", "campaign_summary"),
    ),  # fmt: skip
    (
        "fit or ask a model",
        ("fit_and_rank", "fit_custom_model", "propose_batch", "estimate_hit_rate", "stop_advice"),
    ),  # fmt: skip
    ("neighbours of hits", ("similar_designs",)),
    ("buy", ("submit_batch",)),
]


def rounds_figure(calls, episodes, path, game="E", width=880, height=420):
    group_of = {tool: i for i, (_, tools) in enumerate(TOOL_GROUPS) for tool in tools}
    conditions = ("A", "B1")
    n_episodes = Counter(r["condition"] for r in episodes if r["game"] == game)
    totals = defaultdict(float)
    for r in calls:
        if r["game"] == game and r["condition"] in conditions and r["tool"] in group_of:
            totals[(r["condition"], int(r["round"]), group_of[r["tool"]])] += int(r["calls"])
    per = {k: v / n_episodes[k[0]] for k, v in totals.items()}
    stack = {
        (c, rnd): sum(per.get((c, rnd, g), 0) for g in range(len(TOOL_GROUPS)))
        for c in conditions
        for rnd in range(1, 5)
    }
    left, right, top, bottom = 56, 40, 104, 70
    x0, x1, y0, y1 = left, width - right, top, height - bottom
    hi = max(stack.values()) * 1.12
    slot = (x1 - x0) / 4
    bw = 54

    def py(v):
        return y1 - v / hi * (y1 - y0)

    parts = [
        text(24, 30, "What the agent does each round on E", 16, INK, weight="600"),
        text(
            24,
            50,
            "Tool calls per episode by round, A (campaign tools only) against B1 (with the "
            "advisor tools), both models",
            12.5,
            INK2,
        ),
        legend([(label, SERIES[i]) for i, (label, _) in enumerate(TOOL_GROUPS)], 24, 78),
    ]
    for value in range(0, int(hi) + 1, 2):
        parts.append(line(x0, py(value), x1, py(value)))
        parts.append(text(x0 - 10, py(value) + 4, value, 11, MUTED, "end"))
    for rnd in range(1, 5):
        centre = x0 + (rnd - 0.5) * slot
        parts.append(text(centre, y1 + 44, f"round {rnd}", 12, INK2, "middle", "600"))
        for j, condition in enumerate(conditions):
            bx = centre + (j - 1) * (bw + 8) + 4
            level = 0.0
            for g in range(len(TOOL_GROUPS)):
                value = per.get((condition, rnd, g), 0)
                if value <= 0:
                    continue
                parts.append(
                    bar(bx, py(level + value), bw, py(level) - py(level + value), SERIES[g], 0)
                )
                level += value
            parts.append(
                text(bx + bw / 2, py(level) - 6, f"{level:.1f}", 11.5, INK, "middle", "600")
            )
            parts.append(text(bx + bw / 2, y1 + 18, condition, 12, INK2, "middle"))
    parts.append(
        text(
            24,
            height - 10,
            f"Means over {n_episodes['A']} A and {n_episodes['B1']} B1 episodes (trials "
            "100-119, deepseek and mimo). A round ends when a batch is submitted.",
            11.5,
            MUTED,
        )
    )
    return save(
        path,
        width,
        height,
        parts,
        "Tool calls by round on E",
        "Stacked tool calls per episode in each round, for conditions A and B1 on E.",
    )


# --- 4. how informative the predicted scores are -----------------------------------------


def auc_interval(auc: float, positives: int, negatives: int) -> tuple[float, float]:
    """Hanley and McNeil (1982) standard error of an AUC, as a 90% normal interval."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    variance = (
        auc * (1 - auc) + (positives - 1) * (q1 - auc * auc) + (negatives - 1) * (q2 - auc * auc)
    ) / (positives * negatives)
    se = math.sqrt(max(variance, 0.0))
    return auc - Z90 * se, auc + Z90 * se


def signal_figure(signal, path, width=880):
    groups = []
    for game in ("N-dev", "E"):
        items = []
        for r in sorted((r for r in signal if r["game"] == game), key=lambda r: -float(r["auc"])):
            auc = float(r["auc"])
            hits, n = int(r["hits"]), int(r["with_value"])
            items.append((r["feature"], r["higher_is_better"] == "True", auc,
                          *auc_interval(auc, hits, n - hits), hits, n))  # fmt: skip
        groups.append((game, items))
    left, right, top, row_h, gap = 330, 90, 84, 30, 24
    height = top + sum(len(items) * row_h + gap for _, items in groups) + 56
    lo, hi = 0.35, 0.8
    x0, x1 = left, width - right

    def sx(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    parts = [
        text(
            24,
            30,
            "The predicted scores carry little signal, and almost none on E",
            16,
            INK,
            weight="600",
        ),  # fmt: skip
        text(
            24,
            50,
            "AUC of each visible feature for telling hits from non-hits, in its declared good "
            "direction, 90% interval (Hanley-McNeil)",
            12.5,
            INK2,
        ),
    ]
    for tick in (0.4, 0.5, 0.6, 0.7, 0.8):
        parts.append(line(sx(tick), top - 8, sx(tick), height - 50))
        parts.append(text(sx(tick), height - 34, f"{tick:.1f}", 11, MUTED, "middle"))
    parts.append(line(sx(0.5), top - 12, sx(0.5), height - 50, INK2, 1.5))
    parts.append(text(sx(0.5) + 6, top - 14, "no signal", 11.5, MUTED))
    y = top + 4
    for index, (game, items) in enumerate(groups):
        colour = SERIES[index]
        designs = next(int(r["designs"]) for r in signal if r["game"] == game)
        counts = sorted({item[5] for item in items})
        hits = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
        parts.append(
            text(
                24,
                y + 4,
                f"{GAME_LABEL[game]}: {designs} designs, {hits} hits",
                12.5,
                colour,
                weight="600",
            )  # fmt: skip
        )
        for feature, higher, auc, low, high, _, _ in items:
            cy = y + 22
            label = feature if higher else f"{feature}  (lower is better)"
            parts.append(text(42, cy + 4, label, 12.5, INK2))
            parts.append(dot_interval(sx, cy, auc, low, high, colour))
            parts.append(text(sx(high) + 10, cy + 4, f"{auc:.2f}", 12, INK, weight="600"))
            y += row_h
        y += gap
    parts.append(
        text(
            24,
            height - 12,
            "N-dev is the development split the games use (the held-out split is not read). "
            "Designs missing a value are left out of that row.",
            11.5,
            MUTED,
        )
    )
    return save(
        path,
        width,
        height,
        parts,
        "How informative the predicted scores are",
        "AUC of each visible feature for hits, on N-dev and on E.",
    )


# --- 5. agents follow defaults -----------------------------------------------------------

POLICY_GROUPS = [
    ("learner_sim (the default)", SERIES[0]),
    ("learner", SERIES[1]),
    ("a purchased variant", SERIES[2]),
    ("random or fixed", SERIES[3]),
]


def policy_group(choice: str) -> int:
    if choice in ("", "learner_sim"):
        return 0  # an unset policy runs the default
    if choice == "learner":
        return 1
    if choice.endswith("_purchased"):
        return 2
    return 3


def defaults_figure(calls, path, width=880):
    rows = []
    for model in ("deepseek", "mimo", "sonnet"):
        counts = [0] * 4
        for r in calls:
            # a failed call ran no policy, so it chose nothing
            if r["tool"] == "propose_batch" and r["model"] == model and r["ok"] == "True":
                counts[policy_group(r["choice"])] += int(r["calls"])
        if sum(counts):
            rows.append((f"{model}: propose_batch policy", counts))
    scaling = [0, 0, 0, 0]
    for r in calls:
        if r["tool"] == "fit_custom_model" and r["ok"] == "True":
            scaling[2 if r["choice"] == "purchased" else 0] += int(r["calls"])
    left, right, top, row_h = 290, 70, 104, 46
    height = top + (len(rows) + 1) * row_h + 70
    x0, x1 = left, width - right
    parts = [
        text(24, 30, "Agents take the default", 16, INK, weight="600"),
        text(
            24,
            50,
            "Share of successful calls by the choice made, all agent episodes; n is their number",
            12.5,
            INK2,
        ),
        legend(POLICY_GROUPS, 24, 80),
    ]

    def stacked(y, label, counts, names=None):
        out = [text(left - 12, y + 16, label, 12.5, INK2, "end")]
        total = sum(counts)
        x = x0
        for g, value in enumerate(counts):
            if not value:
                continue
            w = value / total * (x1 - x0)
            out.append(bar(x, y + 2, w, 20, POLICY_GROUPS[g][1], 0))
            if w > 34:
                out.append(text(x + w / 2, y + 16, f"{100 * value / total:.0f}%", 11.5, SURFACE,
                                "middle", "600"))  # fmt: skip
            x += w
        out.append(text(x1 + 8, y + 16, f"n={total}", 11.5, MUTED))
        return out

    for i, (label, counts) in enumerate(rows):
        parts.extend(stacked(top + i * row_h, label, counts))
    y = top + len(rows) * row_h + 10
    parts.append(line(24, y - 6, width - 24, y - 6))
    parts.extend(stacked(y, "condition C: fit_custom_model scaling", scaling))
    parts.append(
        text(
            24,
            height - 30,
            "In condition C the blue share is the default pool scaling and the green share "
            "the purchased scaling, an explicit argument.",
            11.5,
            MUTED,
        )
    )
    parts.append(
        text(
            24,
            height - 12,
            "Purchased scaling is the best scripted policy on N-dev and the worst on E: "
            "agents rarely try it, but always using it would not be right either.",
            11.5,
            MUTED,
        )
    )
    return save(
        path,
        width,
        height,
        parts,
        "Agents take the default",
        "Share of propose_batch policies and fit_custom_model scalings chosen, per model.",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, calls, signal = read(EPISODES), read(CALLS), read(SIGNAL)
    written = [
        opening_figure(episodes, signal, OUT / "opening-move.svg"),
        cost_figure(episodes, OUT / "cost-vs-hits.svg"),
        rounds_figure(calls, episodes, OUT / "e-tools-by-round.svg"),
        signal_figure(signal, OUT / "feature-signal.svg"),
        defaults_figure(calls, OUT / "defaults.svg"),
    ]
    for path in written:
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
