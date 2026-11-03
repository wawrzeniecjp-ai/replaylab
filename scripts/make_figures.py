"""Draw the write-up's figures from the committed CSVs.

    python scripts/make_figures.py        # -> reports/figures/*.svg

Reads ``reports/agent_study.csv`` (written by ``scripts/export_study.py``) and
``reports/ladder.csv`` (written by ``scripts/run_ladder.py``), and uses the project's own
``paired_bootstrap``, so a figure and ``replaylab-eval compare`` cannot drift apart.

The figures are static SVG, since they are read in a README where scripts do not run: every
series is directly labelled rather than relying on a hover layer, and every figure's numbers
also appear as a table in the README. They carry an explicit light surface so they read the
same under either GitHub theme.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from replaylab.eval.seeds import episode_seed
from replaylab.eval.stats import paired_bootstrap

OUT = Path("reports/figures")
STUDY = Path("reports/agent_study.csv")
LADDER = Path("reports/ladder.csv")

# the validated categorical slots (docs: dataviz reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a84"
GRID = "#e6e5e1"
FONT = "font-family=\"ui-sans-serif,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif\""


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, body, size=13, fill=INK, anchor="start", weight="400"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{esc(str(body))}</text>'
    )


def svg(width, height, body, title, desc):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="t d">'
        f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>'
        f"{body}</svg>\n"
    )


# --- data -------------------------------------------------------------------------------


def study_rows():
    with STUDY.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hits_by_seed(rows, game, run, condition, metric="hits"):
    # keyed by episode seed, as ``compare`` keys them, so the resampling draws are the same
    return {
        episode_seed(game, int(r["trial"])): float(r[metric])
        for r in rows
        if r["game"] == game and r["run"] == run and r["condition"] == condition
    }


def _bootstrap(a, b, ci):
    result = paired_bootstrap(a, b, n_resamples=10_000, seed=0, ci=ci)
    return result.mean_diff, result.ci_low, result.ci_high


def paired(rows, game, run, a, b, ci=0.9, metric="hits"):
    """(mean difference, low, high) of a - b, the same way ``compare`` computes it."""
    return _bootstrap(
        hits_by_seed(rows, game, run, a, metric),
        hits_by_seed(rows, game, run, b, metric),
        ci,
    )


def pooled(rows, cells, a, b, ci=0.9):
    """The same, pooled over several (game, run) cells; pairs are keyed by (run, seed)."""
    left, right = {}, {}
    for game, run in cells:
        left.update({(run, k): v for k, v in hits_by_seed(rows, game, run, a).items()})
        right.update({(run, k): v for k, v in hits_by_seed(rows, game, run, b).items()})
    return _bootstrap(left, right, ci)


def ladder_means():
    means = defaultdict(list)
    with LADDER.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            means[(row["game"], row["policy"])].append(float(row["hits"]))
    return {k: sum(v) / len(v) for k, v in means.items()}


def round_curve(rows, game, run, condition):
    keys = ("r1", "r2", "r3", "r4")
    picked = [r for r in rows if r["game"] == game and r["run"] == run
              and r["condition"] == condition]  # fmt: skip
    return [sum(float(r[k]) for r in picked) / len(picked) for k in keys]


# --- figure 1: the two-sided result -----------------------------------------------------


def interval_figure(
    cells, path, title, subtitle, note, width=880, axis_label="hits, 90% paired interval"
):
    """Horizontal dot-and-interval: the form for a signed effect with uncertainty."""
    left, right, top = 252, 76, 76
    row_h, gap = 34, 18
    groups = []
    for group, items in cells:
        groups.append((group, items))
    height = top + sum(len(i) * row_h + gap for _, i in groups) + 46
    lo = min(c[2] for _, items in groups for c in items)
    hi = max(c[3] for _, items in groups for c in items)
    span = max(abs(lo), abs(hi)) * 1.18
    x0, x1 = left, width - right

    def sx(v):
        return x0 + (v + span) / (2 * span) * (x1 - x0)

    parts = [
        text(24, 30, title, 16, INK, weight="600"),
        text(24, 50, subtitle, 12.5, INK2),
    ]
    for value in range(int(-span // 5 * 5), int(span) + 6, 5):
        if abs(value) > span:
            continue
        parts.append(
            f'<line x1="{sx(value):.1f}" y1="{top - 8}" x2="{sx(value):.1f}" '
            f'y2="{height - 42}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(text(sx(value), height - 24, value, 11, MUTED, "middle"))
    parts.append(
        f'<line x1="{sx(0):.1f}" y1="{top - 12}" x2="{sx(0):.1f}" y2="{height - 42}" '
        f'stroke="{INK2}" stroke-width="1.5"/>'
    )
    y = top + 6
    for index, (group, items) in enumerate(groups):
        colour = SERIES[index % len(SERIES)]
        parts.append(text(24, y + 4, group, 12.5, colour, weight="600"))
        for label, mean, low, high in items:
            parts.append(text(42, y + 22, label, 12.5, INK2))
            parts.append(
                f'<line x1="{sx(low):.1f}" y1="{y + 17:.1f}" x2="{sx(high):.1f}" '
                f'y2="{y + 17:.1f}" stroke="{colour}" stroke-width="2" '
                f'stroke-linecap="round"/>'
            )
            for edge in (low, high):
                parts.append(
                    f'<line x1="{sx(edge):.1f}" y1="{y + 11:.1f}" x2="{sx(edge):.1f}" '
                    f'y2="{y + 23:.1f}" stroke="{colour}" stroke-width="2"/>'
                )
            parts.append(
                f'<circle cx="{sx(mean):.1f}" cy="{y + 17:.1f}" r="5" fill="{colour}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
            side = "start" if mean >= 0 else "end"
            offset = 12 if mean >= 0 else -12
            parts.append(
                text(
                    sx(high) + offset if mean >= 0 else sx(low) + offset,
                    y + 21,
                    f"{mean:+.2f}",
                    12,
                    INK,
                    side,
                    weight="600",
                )  # fmt: skip
            )
            y += row_h
        y += gap
    parts.append(text(24, height - 6, note, 11.5, MUTED))
    parts.append(text(width - right, height - 6, axis_label, 11.5, MUTED, "end"))
    path.write_text(
        svg(width, height, "".join(parts), title, f"{subtitle}. {note}"), encoding="utf-8"
    )
    return path


# --- figure 2: the ladder inversion -----------------------------------------------------


def ladder_figure(path, width=760):
    means = ladder_means()
    policies = [
        ("random", "random"),
        ("fixed:esmfold_plddt", "best fixed column"),
        ("learner", "learner"),
        ("learner_sim", "learner_sim"),
        ("learner_sim_purchased", "learner_sim_purchased"),
    ]
    games = [("N-dev", "N-dev (Nipah)"), ("E", "E (EGFR transfer)")]
    left, right, top, bottom = 196, 108, 92, 44
    row_h = 52
    height = top + len(policies) * row_h + bottom
    hi = max(means.get((g, p), 0) for g, _ in games for p, _ in policies) * 1.12
    x0, x1 = left, width - right

    parts = [
        text(24, 30, "The same ladder, two pools: the ranking inverts", 16, INK, weight="600"),
        text(
            24, 50, "Mean hits of 60 purchases over 100 seeds, from reports/ladder.csv", 12.5, INK2
        ),  # fmt: skip
    ]
    for index, (_, label) in enumerate(games):
        colour = SERIES[index]
        lx = 24 + index * 210
        parts.append(f'<rect x="{lx}" y="{62}" width="10" height="10" rx="2" fill="{colour}"/>')
        parts.append(text(lx + 16, 71, label, 12, INK2))
    for row, (policy, label) in enumerate(policies):
        y = top + row * row_h
        parts.append(text(left - 12, y + 24, label, 12.5, INK2, "end"))
        for index, (game, _) in enumerate(games):
            value = means.get((game, policy))
            if value is None:
                continue
            bar_y = y + 4 + index * 17
            w = (value / hi) * (x1 - x0)
            parts.append(
                f'<rect x="{x0}" y="{bar_y}" width="{max(w, 2):.1f}" height="13" rx="4" '
                f'fill="{SERIES[index]}"/>'
            )
            parts.append(text(x0 + w + 8, bar_y + 11, f"{value:.1f}", 12, INK, weight="600"))
        parts.append(
            f'<line x1="{x0}" y1="{y + row_h - 6}" x2="{x1}" y2="{y + row_h - 6}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
    parts.append(
        text(
            24,
            height - 14,
            "On N-dev the purchased-row scaler is best by far; on E it is the worst and "
            "random beats both learners.",
            11.5,
            MUTED,
        )  # fmt: skip
    )
    path.write_text(
        svg(
            width,
            height,
            "".join(parts),
            "The same ladder, two pools: the ranking inverts",
            "Mean hits per scripted policy on N-dev and on E, 100 seeds.",
        ),  # fmt: skip
        encoding="utf-8",
    )
    return path


# --- figure 3: the cold start on E ------------------------------------------------------


def curves_figure(rows, path, width=900, height=400):
    series = [
        ("B1  campaign + advisor tools", round_curve(rows, "E", "egfr-ds", "B1"), SERIES[0]),
        ("A  campaign tools only", round_curve(rows, "E", "egfr-ds", "A"), SERIES[1]),
        (
            "B2  advisor minus neighbour search",
            round_curve(rows, "E", "egfr-ds", "B2"),
            SERIES[2],
        ),  # fmt: skip
        (
            "scripted:learner_sim  the bar",
            round_curve(rows, "E", "egfr-ds", "scripted:learner_sim"),
            SERIES[3],
        ),  # fmt: skip
    ]
    left, right, top, bottom = 56, 320, 86, 52
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    hi = max(max(s[1]) for s in series) * 1.1

    def px(i):
        return x0 + i / 3 * (x1 - x0)

    def py(v):
        return y1 - v / hi * (y1 - y0)

    parts = [
        text(24, 30, "Where the gain comes from: rounds 2 to 4 on E", 16, INK, weight="600"),
        text(
            24,
            50,
            "Cumulative hits at each round boundary, mean over trials 100-119, deepseek",
            12.5,
            INK2,
        ),  # fmt: skip
    ]
    for value in range(0, int(hi) + 1, 5):
        parts.append(
            f'<line x1="{x0}" y1="{py(value):.1f}" x2="{x1}" y2="{py(value):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(text(x0 - 10, py(value) + 4, value, 11, MUTED, "end"))
    for i in range(4):
        parts.append(text(px(i), y1 + 22, f"round {i + 1}", 11.5, MUTED, "middle"))
    for _label, values, colour in series:
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for i, v in enumerate(values):
            parts.append(
                f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" fill="{colour}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
    # direct labels, nudged apart so near-equal endpoints do not overprint
    ordered = sorted(series, key=lambda s: s[1][-1], reverse=True)
    placed: list[float] = []
    for label, values, colour in ordered:
        y = py(values[-1]) + 4
        while any(abs(y - taken) < 19 for taken in placed):
            y += 19 - min(abs(y - taken) for taken in placed)
        placed.append(y)
        parts.append(
            f'<line x1="{px(3) + 6:.1f}" y1="{py(values[-1]):.1f}" x2="{px(3) + 20:.1f}" '
            f'y2="{y - 4:.1f}" stroke="{colour}" stroke-width="1.5" opacity="0.5"/>'
        )
        parts.append(text(px(3) + 26, y, f"{values[-1]:.1f}", 12, INK, weight="600"))
        parts.append(text(px(3) + 62, y, label, 12, INK2))
    parts.append(
        text(
            24,
            height - 14,
            "The scripted players cannot fit until round 4: the model gate needs 2 hits and "
            "8 purchases, and on E it opens once.",
            11.5,
            MUTED,
        )  # fmt: skip
    )
    path.write_text(
        svg(
            width,
            height,
            "".join(parts),
            "Cumulative hits by round on E",
            "Cumulative hits at each round boundary for B1, A, B2 and the scripted bar.",
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = study_rows()
    written = []

    nd_cells = (("N-dev", "ds2"), ("N-dev", "mimo2"))
    e_cells = (("E", "egfr-ds"), ("E", "egfr-mimo"))
    written.append(
        interval_figure(
            [
                (
                    "N-dev (Nipah), trials 0-19",
                    [
                        ("deepseek", *paired(rows, "N-dev", "ds2", "B1", "A")),
                        ("mimo", *paired(rows, "N-dev", "mimo2", "B1", "A")),
                    ],
                ),
                (
                    "E (EGFR transfer), trials 100-119",
                    [
                        ("deepseek", *paired(rows, "E", "egfr-ds", "B1", "A")),
                        ("mimo", *paired(rows, "E", "egfr-mimo", "B1", "A")),
                    ],
                ),
                (
                    "Pooled",
                    [
                        ("N-dev, both models", *pooled(rows, nd_cells, "B1", "A")),
                        ("E, both models", *pooled(rows, e_cells, "B1", "A")),
                        ("all four cells", *pooled(rows, nd_cells + e_cells, "B1", "A")),
                    ],
                ),
            ],
            OUT / "advisor-effect.svg",
            "The advisor tools add hits on both pools, for both models",
            "B1 (campaign + advisor tools) minus A (campaign tools only), paired by seed",
            "20 pairs per cell. Same prompt and limits in both arms; only the tool list differs.",
        )
    )

    bar_nd = "scripted:learner_sim_purchased"
    nd = paired(rows, "N-dev", "confirm1", "B1", bar_nd)
    e_ds = paired(rows, "E", "egfr-ds", "B1", "scripted:learner_sim")
    e_mimo = paired(rows, "E", "egfr-mimo", "B1", "scripted:learner_sim")
    written.append(
        interval_figure(
            [
                (
                    "N-dev (Nipah), vs scripted:learner_sim_purchased",
                    [
                        ("deepseek B1, 0-19", *paired(rows, "N-dev", "ds2", "B1", bar_nd)),
                        ("deepseek B1, 100-119", *nd),
                        ("mimo B1, 0-19", *paired(rows, "N-dev", "mimo2", "B1", bar_nd)),
                    ],
                ),  # fmt: skip
                (
                    "E (EGFR transfer), vs scripted:learner_sim",
                    [("deepseek B1, 100-119", *e_ds), ("mimo B1, 100-119", *e_mimo)],
                ),  # fmt: skip
            ],
            OUT / "two-sided-result.svg",
            "Against the strongest baseline: behind on Nipah, ahead on EGFR",
            "Agent minus the strongest baseline of each pool, paired by seed, 20 trials a row",
            "Bars: docs/design-notes.md sections 9, 12 and 15. Labels give the trial block.",
        )
    )

    ablation = []
    for model, run in (("deepseek", "egfr-ds"), ("mimo", "egfr-mimo")):
        cells = [
            ("B1 - A   (all advisor tools)", *paired(rows, "E", run, "B1", "A")),
            ("B1 - B2  (neighbour search alone)", *paired(rows, "E", run, "B1", "B2")),
            ("B2 - A   (advisor without it)", *paired(rows, "E", run, "B2", "A")),
        ]
        ablation.append((f"E, {model}", cells))
    written.append(
        interval_figure(
            ablation,
            OUT / "e-ablation.svg",
            "On E, the advisor's whole contribution is the neighbour search",
            "Paired differences between conditions, 20 trials per cell",
            "B2 is B1 without similar_designs. Remove it and the advisor adds nothing "
            "measurable over A.",
        )
    )
    # the authorship ablation: the same contrast on both pools, on hits and on diversity
    tool_cells = e_cells + (("N-dev", "auth-ds"), ("N-dev", "auth-mimo"))
    labels = ("EGFR, deepseek", "EGFR, mimo", "Nipah, deepseek", "Nipah, mimo")
    written.append(
        interval_figure(
            [
                (
                    "hits -- above zero in all four cells",
                    [
                        (label, *paired(rows, game, run, "B1", "B2"))
                        for label, (game, run) in zip(labels, tool_cells, strict=True)
                    ],
                ),
                (
                    "distinct families -- never below zero",
                    [
                        (label, *paired(rows, game, run, "B1", "B2", metric="distinct_families"))
                        for label, (game, run) in zip(labels, tool_cells, strict=True)
                    ],
                ),
            ],
            OUT / "similarity-tool.svg",
            "The similarity tool adds hits on both pools, and costs no diversity on either",
            "B1 minus B2 (B1 without similar_designs), paired by seed",
            "EGFR 20 pairs a row, Nipah 40. Nipah's families are multi-author, so its hit gain "
            "cannot be an authorship effect.",
            axis_label="90% paired interval",
        )
    )
    written.append(ladder_figure(OUT / "ladder-inversion.svg"))
    written.append(curves_figure(rows, OUT / "e-round-curves.svg"))
    for path in written:
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
