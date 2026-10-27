# Walkthrough: the data and the numbers, one REPL step at a time

This is a hands-on tour for a Python REPL: first the preliminary analyses (what
the tables are, which features are allowed in, how clusters and folds were cut),
then evaluating and comparing the strategies on the training Nipah data. Every
block is copy-pasteable, runs against the real tables under `data/processed/`,
and says what you should see.

Conventions:

* Start **one** REPL in the repository root; the blocks build on each other.
  Either `uv run python` or `.venv/bin/python` works.
* `print(...)` shows what matters — bare expressions are left for you to poke at.
* Timings are measured on this machine (WSL2), so yours will be in the same
  ballpark.
* Nothing here touches the held-out fold (see "steps that are easy to miss").

```bash
# in a shell, from the repository root
uv run python
```

## 0. Setup

```python
from pathlib import Path
import json

import numpy as np
import pandas as pd

from replaylab.config import load_feature_config, load_games
from replaylab.datasets.proteinbase import build_all, compare_to_spec
from replaylab.datasets.split_tables import (
    EXPECTED_CLUSTER_SUMMARY,
    build_cluster_table,
    build_fold_table,
    compare_cluster_summary,
    heldout_ids,
)
from replaylab.datasets.feature_review import (
    render_markdown,
    review_non_numeric_metrics,
    review_numeric_metrics,
)
from replaylab.datasets.author_checks import (
    designs_per_author,
    nearest_neighbor_table,
    summarize_author_checks,
)
from replaylab.eval import ladder as L

ROOT = Path.cwd()  # run the REPL from the repository root
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
NIPAH = "nipah-glycoprotein-g"
EGFR = "egfr"

designs = pd.read_parquet(PROCESSED / "designs.parquet")
outcomes = pd.read_parquet(PROCESSED / "outcomes.parquet")
long = pd.read_parquet(PROCESSED / "evaluations_long.parquet")

print(designs.shape, outcomes.shape, long.shape)
# (5253, 15) (2630, 8) (205620, 10)
```

The three tables, and what each one is for:

| table | one row per | carries |
| --- | --- | --- |
| `designs` | designed sequence | id, sequence, author/method metadata, `length`, `design_class`, and the **visible metrics** |
| `outcomes` | (design, target) pair | the aggregated verdict: `hit`, `conflict`, `strength`, `kd_molar` |
| `evaluations_long` | one raw evaluation record | the 205k rows everything else is derived from |

```python
print(designs.columns.tolist())
# ['design_id', 'sequence', 'meta_name', 'meta_author', 'meta_method', 'length',
#  'seq_nonstandard', 'design_class', 'boltz2_iptm', 'boltz2_ipsae',
#  'boltz2_min_ipsae', 'boltz2_plddt', 'shape_complimentarity_boltz2_binder_ss',
#  'esmfold_plddt', 'proteinmpnn_score']

print(outcomes.groupby("target")["hit"].agg(n="size", hit_rate="mean").round(3).to_string())
```

```
                           n  hit_rate
target
...
egfr                     826     0.162
hnmt                      30     0.067
human-serum-albumin      103     0.000
nipah-glycoprotein-g    1030     0.100
...
```

Two pools carry the experiments: **nipah-glycoprotein-g (1030 designs, 10.0%
hits)** and **egfr (826, 16.2%)**. The ~20 small targets (12–103 designs) are
the stop-games and the single-shot checks.

---

# Stage 1 — the preliminary analyses

## 1.1 Rebuild the tables and check them against the spec (1.5 s)

```python
tables = build_all(ROOT / "data" / "raw" / "proteinbase.csv")
for line in compare_to_spec(tables.report):
    print(line)
```

```
OK  n_designs: expected 5253, got 5253
OK  n_binding_records: expected 9326, got 9326
OK  targets.nipah-glycoprotein-g.n_hits: expected 103, got 103
OK  targets.egfr.n_hits: expected 134, got 134
...  (15 lines, all OK)
```

This is what `python data/build_tables.py --check` does (`--strict` would exit 1
on a `DIFF`). It does **not** rewrite anything: `write_tables(tables, PROCESSED)`
is the explicit "freeze it again" call, and `--force`-style edits to
`data/processed/` are how you would invalidate everything downstream.

Prove that what is on disk is what a rebuild produces:

```python
committed = json.loads((PROCESSED / "build_report.json").read_text())
print("report matches the one on disk:", committed == tables.report)
# report matches the one on disk: True
```

## 1.2 Which metrics are allowed in (the feature review)

The visible features in `config/features.yaml` were chosen from a coverage and
leakage review of every computational metric: a metric must have **coverage
≥ 0.9**, and it is excluded if having a value correlates with the label
(hit-rate gap > 0.05, with at least 30 designs in both groups).

```python
reviews = review_numeric_metrics(long, outcomes, NIPAH)
cols = [
    "metric",
    "n_without",
    "coverage",
    "hit_rate_with",
    "hit_rate_without",
    "missingness_auc",
    "decision",
    "reason",
]
review_frame = pd.DataFrame([r.__dict__ for r in reviews])[cols]
print(review_frame.sort_values(["decision", "metric"]).round(3).to_string(index=False))
```

```
                            metric  n_without  coverage  hit_rate_with  hit_rate_without  missingness_auc decision             reason
            aligned-lengthafdb50        825     0.199          0.151             0.087            0.557  exclude  coverage below minimum
            aligned-lengthcath50        823     0.201          0.150             0.087            0.556  exclude  coverage below minimum
...                            ...        ...       ...            ...               ...              ...      ...                   ...
                      boltz2_ipsae          2     0.998          0.098             1.000            0.490  include                  ok
                     esmfold_plddt          8     0.992          0.101             0.000            0.504  include                  ok
                 proteinmpnn_score          8     0.992          0.101             0.000            0.504  include                  ok
...
```

```python
print("excluded:", [r.metric for r in reviews if r.decision == "exclude"])
```

```
excluded: ['aligned-lengthafdb50', 'aligned-lengthcath50', 'aligned-lengthpdb100', 'evalueafdb50',
 'evaluecath50', 'evaluepdb100', 'rmsdafdb50', 'rmsdcath50', 'rmsdpdb100', 'seqidentityafdb50',
 'seqidentitycath50', 'seqidentitypdb100', 'tm-scoreafdb50', 'tm-scorecath50', 'tm-scorepdb100']
```

15 exclusions, all for the same reason: the `*afdb50/cath50/pdb100` family sits
at ~0.20 coverage. Note the trap in the table: `boltz2_ipsae` shows
`hit_rate_without = 1.000` (a gap of 0.9!) — but only **2** designs lack a
value, below `min_group = 30`, so the leakage rule does not fire. That is the
rule working, not a bug.

Non-numeric metrics (labels, JSON blobs) are inventoried separately, and
`render_markdown` reproduces the report the script writes:

```python
non_numeric = review_non_numeric_metrics(long, outcomes, NIPAH)
print(pd.DataFrame(non_numeric).head(6).to_string(index=False))

week0 = ROOT / "notes" / "week0" / "feature_review.md"
if week0.exists():
    both = "\n".join(
        [
            render_markdown(
                t,
                review_numeric_metrics(long, outcomes, t),
                review_non_numeric_metrics(long, outcomes, t),
            )
            for t in (NIPAH, EGFR)
        ]
    )
    print("reproduces notes/week0/feature_review.md:", both == week0.read_text())
# reproduces notes/week0/feature_review.md: True
```

(The file holds both targets -- nipah and egfr -- which is why the check
renders both sections.)

`python data/feature_review.py` regenerates that file; it is evidence only —
*you* decide what goes into `config/features.yaml`.

## 1.3 Is authorship baked into "similarity"? (0.1 s)

Nearest neighbours by 3-mer cosine similarity are often the same author, which
is why the docs warn against over-buying one family and why clustering exists.

```python
per_author = designs_per_author(designs, outcomes, NIPAH)
print(per_author.head(5).to_string())
```

```
author
avt              10
briannaughton    10
chowe            10
david-bichara    10
driegerdesign    10
```

```python
nn = nearest_neighbor_table(designs, outcomes, NIPAH, k=3)  # 0.1 s
author_summary = summarize_author_checks(nn, per_author)
for key in (
    "n_designs",
    "n_authors",
    "max_designs_per_author",
    "same_author_rate_all",
    "same_author_rate_hit_pairs",
):
    value = author_summary[key]
    print(f"{key}: {value if not isinstance(value, float) else round(value, 3)}")
```

```
n_designs: 1030
n_authors: 206
max_designs_per_author: 10
same_author_rate_all: 0.447
same_author_rate_hit_pairs: 0.516
```

45% of nearest-neighbour pairs are same-author — and among *hits* it is 52%.
Authorship is a strong, hidden structure in this pool.

The same check restricted to the development folds:

```python
folds_csv = pd.read_csv(SPLITS / "folds.csv")
dev_ids = folds_csv[(folds_csv["target"] == NIPAH) & (folds_csv["fold"] != 2)]["design_id"]
dev_summary = summarize_author_checks(
    nearest_neighbor_table(designs, outcomes, NIPAH, ids=dev_ids),
    designs_per_author(designs, outcomes, NIPAH, ids=dev_ids),
)
print(dev_summary["n_designs"], round(dev_summary["same_author_rate_hit_pairs"], 3))
```

## 1.4 Clusters (0.2 s)

Sequences within cosine distance 0.4 (binary 3-mer vectors) form a cluster —
the connected components of that pair graph. A cluster is a *family*.

```python
cluster_table, summaries = build_cluster_table(
    designs[["design_id", "sequence"]], outcomes[["design_id", "target", "hit"]]
)
for line in compare_cluster_summary(summaries):
    print(line)
```

```
OK   nipah-glycoprotein-g.n_clusters: expected 631, got 631
OK   nipah-glycoprotein-g.largest_cluster: expected 125, got 125
```

```python
print(
    cluster_table.groupby("target")
    .agg(
        designs=("design_id", "size"),
        clusters=("cluster_id", "nunique"),
        largest=("cluster_size", "max"),
    )
    .to_string()
)
```

```
                     designs  clusters  largest
target
egfr                     826       499      145
nipah-glycoprotein-g    1030       631      125
```

The committed file agrees cluster-for-cluster:

```python
committed_clusters = pd.read_csv(SPLITS / "clusters.csv")
merged = committed_clusters.merge(
    cluster_table, on=["design_id", "target"], suffixes=("_disk", "_re")
)
print("cluster ids agree:", (merged["cluster_id_disk"] == merged["cluster_id_re"]).all())
# cluster ids agree: True
```

> The expected 631 (not 632) is deliberate: one pair sits *exactly* on the 0.4
> threshold (138/230 = 0.6 similarity) and strict float rounding dropped that
> edge in the exploratory run. `EXPECTED_CLUSTER_SUMMARY` carries the number and
> the comment.

## 1.5 Folds — the train/test split (<0.1 s)

```python
nipah_clusters = cluster_table[cluster_table["target"] == NIPAH]
folds_df, selection = build_fold_table(nipah_clusters, outcomes, NIPAH)
print("chosen seed:", selection.seed, "| sizes:", selection.sizes, "| hits:", selection.hits)

balance = folds_df.merge(
    outcomes.loc[outcomes["target"] == NIPAH, ["design_id", "hit"]], on="design_id"
)
print(balance.groupby("fold")["hit"].agg(n="size", hits="sum", rate="mean").round(3).to_string())
print("held-out designs:", len(heldout_ids(folds_df)))
```

```
chosen seed: 0 | sizes: (344, 343, 343) | hits: (35, 34, 34)
        n  hits   rate
fold
0     344    35  0.102
1     343    34  0.099
2     343    34  0.099
held-out designs: 343
```

How to read that: `StratifiedGroupKFold` with **cluster = group** (a family
never straddles a fold) and **hit = label** (equal hit rates), taking the first
seed that lands every fold within ±2 designs and ±2 hits. **Fold 2 is the
held-out fold** — `split: dev` is folds 0 and 1, `split: heldout` is fold 2, and
`split: all` ignores folds (the stop games, `E`, `E-long`, `legacy`).

Confirm it matches the committed split:

```python
committed_folds = pd.read_csv(SPLITS / "folds.csv")
print(
    "folds match data/splits/folds.csv:",
    committed_folds[["design_id", "fold"]].equals(folds_df[["design_id", "fold"]]),
)
```

## 1.6 Freeze and gate

```bash
# in a shell
make data          # build_tables --check (15 x OK) + build_clusters --check
pytest -m data     # 4 tests against the real tables
```

`data/splits/*` and `config/frozen_v1.yaml` are frozen: `--check` reports,
`--force` (and only deliberately) rewrites.

---

# Stage 2 — evaluating and comparing the strategies

## 2.1 One game, one policy, one seed

```python
games = load_games(ROOT / "config" / "games.yaml")
feature_configs = load_feature_config(ROOT / "config" / "features.yaml")
frozen = L.load_frozen(ROOT / "config" / "frozen_v1.yaml")

game = games["N-dev"]
lg = L.LadderGame.build(
    game,
    tables_dir=PROCESSED,
    feature_config=feature_configs[game.target],
    folds_path=SPLITS / "folds.csv",
    clusters_path=SPLITS / "clusters.csv",
)
print(len(lg.policies(0)), "policies:", [p.name for p in lg.policies(0)])
```

```
14 policies: ['random', 'fixed:boltz2_ipsae', 'fixed:boltz2_iptm', 'fixed:boltz2_min_ipsae',
 'fixed:boltz2_plddt', 'fixed:shape_complimentarity_boltz2_binder_ss', 'fixed:esmfold_plddt',
 'fixed:proteinmpnn_score', 'learner', 'learner+stop', 'learner_sim', 'learner_sim+stop',
 'learner_purchased', 'learner_sim_purchased']
```

(the fixed ones follow `config/features.yaml` order, `first_column` first)

```python
policy = next(p for p in lg.policies(0) if p.name == "learner_sim")
card = lg.run(policy, seed=0)
print("hits", card.hits, "purchases", card.purchases, "families", card.distinct_families)
print("cumulative hits after each batch:", card.hits_at_boundaries)
```

`run` builds a fresh campaign every call — nothing is kept between runs — so a
`ScoreCard` is a pure function of (game, policy, seed). The `Preprocessor` on
`lg` is shared on purpose (value-based, seed-independent); the policies you get
from `lg.policies(seed)` are rebuilt per seed (alias-keyed — see the glossary).

## 2.2 Every policy, many seeds (the ladder)

```python
import time

started = time.time()
frame, traces = L.run_ladder(
    games,
    feature_configs,
    tables_dir=PROCESSED,
    seeds=100,
    game_ids=["N-dev"],
    folds_path=SPLITS / "folds.csv",
    clusters_path=SPLITS / "clusters.csv",
    frozen=frozen,
)
print(f"{len(frame)} rows in {time.time() - started:.0f}s")
print(L.game_table(frame, "N-dev"))
```

1,400 rows; **about 7 minutes** on this machine (the whole ladder -- 11 games,
11,400 rows -- took 26 minutes). Every game at once is
`python scripts/run_ladder.py`, which writes `reports/ladder.{csv,md}`.

```
| policy | mean hits | sd | min | max | mean purchases | mean distinct families |
| --- | --- | --- | --- | --- | --- | --- |
| learner | 18.18 | 0.99 | 16 | 21 | 60.0 | 14.1 |
| learner_sim | 14.52 | 3.28 | 12 | 27 | 60.0 | 14.3 |
| learner_sim_purchased | 26.43 | 2.46 | 17 | 31 | 60.0 | 25.4 |
| random | 6.03 | 2.02 | 2 | 12 | 60.0 | 5.9 |
... (14 rows, identical to reports/ladder.md)
```

## 2.3 Read the results you already have

You rarely need to re-run: `reports/ladder.csv` holds every (game, policy,
seed) row.

```python
ladder = pd.read_csv(ROOT / "reports" / "ladder.csv")
variants = ["learner", "learner_purchased", "learner_sim", "learner_sim_purchased"]
table = (
    ladder[ladder["policy"].isin(variants)]
    .groupby(["game", "policy"])["hits"]
    .mean()
    .round(2)
    .unstack()
)
print(table[variants].to_string())
```

```
                       learner  learner_purchased  learner_sim  learner_sim_purchased
game
E                           7.44               4.51        10.39                   4.45
N-dev                      18.18              15.75        14.52                  26.43
N-dev-delay2               17.29              13.71        16.43                  25.09
...
```

(the exact rows come from the committed file; the point is the **flip**: on
`N-dev` the purchased scaler wins by 11.9, on `E` it loses by 5.9.)

One game as the formatted markdown table the report uses:

```python
print(L.game_table(ladder, "E"))
```

## 2.4 Compare paired, not means-vs-means

Every policy plays the *same* seeds on the *same* pool, so the honest
comparison is the per-seed difference:

```python
n_dev = ladder[ladder["game"] == "N-dev"].pivot(index="seed", columns="policy", values="hits")
diff = n_dev["learner_sim_purchased"] - n_dev["learner"]
print(
    "mean diff",
    round(diff.mean(), 2),
    "| sd of diff",
    round(diff.std(), 2),
    "| wins in",
    int((diff > 0).sum()),
    "of",
    len(diff),
    "seeds",
)
print(diff.describe().round(2).to_string())
```

```
mean diff 8.25 | sd of diff 2.71 | wins in 96 of 100 seeds
count    100.00
mean       8.25
std        2.71
min       -1.00
25%        7.00
50%        9.00
75%       10.00
max       14.00
```

Why this matters: `learner` has sd 0.99 and `learner_sim_purchased` sd 2.46 as
*seeds*, but the paired difference is much tighter — the same random draws drive
both policies in a given round, so the difference is mostly the policy, not the
seed. And remember the scope: intervals over seeds measure **policy randomness
on this one pool**, never variation across targets (the `reports/e_long.md`
header says exactly this).

## 2.5 The six legacy numbers

```bash
python scripts/reproduce_legacy.py            # exit 0 = all six inside tolerance
python scripts/reproduce_legacy.py --seeds 20 # a quick look (~10x faster)
```

This script re-implements the *exploratory* learner (it lives only there) and
checks the numbers the spec quotes: `legacy learner 25.93` vs expected 25.9,
`legacy learner_sim 38.49` vs 38.9, plus the purchased-vs-pool-wide scaling
comparison (38.49 vs 32.22; 25.93 vs 27.00). A `DIFF` is a finding, not
necessarily a bug — the script prints where to look.

## 2.6 Look inside a single decision

The MCP tool layer is also the best inspection API: it is what an agent sees,
and it is plain Python.

```python
from replaylab.interfaces.runtime import build_runtime
from replaylab.interfaces.tools import Toolbox

rt = build_runtime(
    "N-dev",
    0,
    ":memory:",
    games_path=ROOT / "config" / "games.yaml",
    features_path=ROOT / "config" / "features.yaml",
    tables_dir=PROCESSED,
    folds_path=SPLITS / "folds.csv",
    clusters_path=SPLITS / "clusters.csv",
)
tb = Toolbox(rt.tool_context())

status = tb.describe_campaign()
print(status.target, status.rounds_left, status.budget, status.purchased, status.hits_found)
print([c.name for c in status.columns])
```

```
nipah-glycoprotein-g 4 60 0 0
['length', 'design_class', 'boltz2_iptm', 'boltz2_ipsae', 'boltz2_min_ipsae',
 'boltz2_plddt', 'shape_complimentarity_boltz2_binder_ss', 'esmfold_plddt',
 'proteinmpnn_score']
```

Play a few rounds and watch the model gate open:

```python
for _ in range(3):
    proposal = tb.propose_batch(k=15, policy="learner_sim")
    tb.submit_batch(proposal.design_ids)

print("hits:", tb.campaign_summary().hits)
rate = tb.estimate_hit_rate()
print("hit rate", round(rate.mean, 3), "interval", [round(x, 3) for x in rate.interval90])
print("stop rules:", [(r.name, r.fired) for r in tb.stop_advice().rules])
```

```
hits: 10
hit rate 0.234 interval [0.14, 0.341]
stop rules: [('low_yield', False), ('decay', True)]
```

`decay` firing there is normal -- it fires by chance in most runs (that is what
the false-stop table in `reports/ladder.md` measures), which is why `low_yield`
is the only rule `WithStop` listens to by default.

Now the proposal itself — why these 15 designs:

```python
proposal = tb.propose_batch(k=15, policy="learner_sim")
print(
    "predicted hit rate:",
    proposal.predicted_hit_rate,
    "| basis:",
    proposal.calibration.basis,
    "| calibrated:",
    proposal.calibration.calibrated,
)
print(proposal.reasons[0].model_dump())
```

```
predicted hit rate: 0.167018 | basis: model | calibrated: False
{'design_id': 'd0143', 'score': 0.207495,
 'top_features': [{'feature': 'shape_complimentarity_boltz2_binder_ss',
                   'contribution': 0.358226}, ...],
 'nearest_hit': 'd0374', 'similarity': 0.479059}
```

`basis` is `model` once at least 2 hits and 8 purchases are delivered, otherwise
`posterior_mean` (the fallback); `calibrated` only flips to `True` after three
realized model predictions (the calibration adjustment). `top_features` are the
orientation-applied contributions, so a positive number means "better, more
hits".

```python
ranked = tb.fit_and_rank(top_k=5)
print([(w.feature, w.weight) for w in ranked.top_weights])
print([(r.design_id, r.score) for r in ranked.ranked])

history = tb.get_history()
print(len(history.designs), "delivered;", history.designs[0].model_dump())
```

`build_runtime` refuses held-out games, and a real store path instead of
`:memory:` resumes where a killed process stopped.

## 2.7 The reports already on disk

| file | what it is |
| --- | --- |
| `reports/ladder.md` | one table per game + the stop-rule false-positive table |
| `reports/ladder.csv` | 11,400 rows, one per (game, policy, seed) |
| `reports/e_long.md` / `.csv` | the pre-registered longer-horizon E run (exploratory, excluded from the default ladder) |
| `notes/week0/` | the evidence from stage 1 (gitignored): `feature_review.md`, `author_checks.json`, `llm_smoke.txt` |

`L.summarize(frame, traces)` regenerates `reports/ladder.md` from a frame you
built yourself; `L.write_reports(frame, csv, md, traces)` writes both files.

---

## Steps that are easy to miss

1. **Read `docs/design-notes.md` before believing a number.** Sections 2–8
   explain the clusters (631 not 632), the decisions behind the ambiguous
   corners, the frozen configuration, the legacy reproduction, the ladder, the
   two decisions from 2026-10-13, and the pre-registered E-long run.
2. **Run the gate before trusting any change you make while exploring:**
   `make test` (634 passed), `make lint`, `pytest -m data` (4),
   `pytest -m integration` (4 — real MCP subprocesses).
3. **Never read fold 2.** `load_outcomes` raises `HeldOutAccessError`,
   `run_ladder` refuses held-out games, `replaylab serve` exits 2. It is read
   once, by the final run. Everything in this walkthrough is dev or `all`.
4. **Seeds are not interchangeable with policy objects.** Aliases depend on the
   seed, so anything alias-keyed (`SimilarityIndex`) belongs to exactly one
   seed: build policies per seed (`lg.policies(seed)`, `make_policy(...)`) and
   take randomness only from `make_rng(seed, slot)`. Same seed → same batch,
   always — re-running one seed is the cheapest regression check there is.
5. **Know which scaling variant you are looking at.** `learner_sim` (pool-wide)
   and `learner_sim_purchased` (delivered rows) are both tracked everywhere
   until the gap is explained; the pre-registered scripted bar is
   `learner_sim_purchased` (26.43 on N-dev).
6. **Frozen means frozen:** `data/splits/*` and `config/frozen_v1.yaml` are
   checked, not edited (`--force` only when you mean it, then re-run
   `make data` and `pytest -m data`).
7. **Timings measured here:** table rebuild 1.5 s, feature review <0.5 s,
   author checks 0.1 s, clustering 0.2 s, folds <0.1 s, N-dev ladder at 100
   seeds ~7 min, full ladder ~26 min, E-long report ~6 min, data tests ~10 s.
8. **Not covered here, but part of the picture:** the LLM round-trip smoke test
   (`python scripts/smoke_llm.py` needs an OpenAI-compatible endpoint — the
   recorded run used a local Qwen3-1.7B), and the agent-facing session
   (`replaylab serve` over MCP, or `replaylab run-scripted --game N-dev
   --policy learner_sim_purchased` for a scripted end-to-end play).
