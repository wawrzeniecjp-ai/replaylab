# Pre-registration: is `similar_designs` exploiting authorship?

Written **before any N-dev episode of condition B2 exists** (checked: no run directory holds a
B2 record for game N-dev, and no record of any condition exists for N-dev trials >= 140).
Same discipline as `docs/prereg-orientation.md`: the question, the arms, the block, the
measurements, the gates and the decision rule are fixed here, and nothing below may change
once the first episode is played.

## 1. The question

Design notes section 17 established that on **E** the advisor's entire measurable value is one
tool: `B1 - B2` (removing `similar_designs`) costs **+7.15** hits [1.65, 12.55] on deepseek and
**+3.85** [0.55, 7.3] on mimo, and B2 falls back to the level of condition A.

`docs/caveats.md` entry 7 then asked *what that tool exploits*, and found a confound. Hits are
clustered, and on E the clusters are largely one author's design series: a design's nearest
3-mer neighbour is the same author **74.3%** of the time against a 4.7% chance baseline (79.8%
among hit pairs), and 22 of E's 28 families with >= 3 designs are single-author. So "fetch the
neighbours of a hit" may be exploiting a shared backbone, a shared pipeline, or simply the
fact that some authors are far more successful than others (`crg-ferruz-lab` binds at 38.1%
against `biolm`'s 12.5%). E cannot separate these.

**N-dev can.** Counting authors per family on the nipah pool:

| | E | N-dev target |
| --- | --- | --- |
| hit-holding families with >= 3 designs | 11 | 6 |
| ... of those, single-author | 6 | **0** |
| mean authors per such family | 3.0 | **11.2** |

Every hit-holding family of any size on nipah spans multiple authors, 11.2 on average, so
following a hit's neighbours there **necessarily crosses author boundaries**. Whatever
`similar_designs` buys on N-dev, it cannot be a proxy for authorship.

**The question:** does removing `similar_designs` cost the agent hits on N-dev, where
authorship is ruled out by the pool's structure?

A single-author replay was considered first and rejected as unplayable: nipah's most prolific
author has 10 designs (budget is 60), and E's three authors with >= 20 designs have degenerate
family structure -- `biolm` has one family holding 79 of its 96 designs (within-family lift
1.2x, so similarity cannot discriminate), `crg-ferruz-lab` has 76 families of which most are
singletons (23 of its 32 hits have no neighbour at all). Within one author this data offers
either one giant family or none, so the contrast has to come from a multi-author pool whose
*families* are multi-author. That is N-dev.

## 2. The arms

Two conditions already in `config/conditions.yaml`, unchanged:

* **B1** -- campaign + advisor tools (14 tools).
* **B2** -- B1 with `similar_designs` excluded (13 tools).

Both arms share one prompt, one model entry, one limits block and one conditions file, so they
differ in exactly one tool. The manipulation is structural: B2 cannot call the tool.

**This runs on the frozen post-orientation-fix toolkit**, config hash **`0fecb4a7e74a`**
(deepseek) and **`c2c8dac7e136`** (mimo) -- verified by `--dry-run` before writing this. Those
are the hashes registered for the Anthropic final test (`docs/prereg-anthropic.md` section 6),
so B1's absolute hit level here is **not** comparable with the earlier N-dev blocks, which were
played on `b457e246517b` before the orientation fix. Only the within-block `B1 - B2` contrast
is interpretable, which is what this note measures.

**This experiment may not alter the final test.** It is a different game behind no lock; the
AB-* games keep their registered conditions (A, B1, B2), block, caps and hashes whatever comes
out of this.

## 3. The block, the game and the models

* **Game N-dev** (nipah development pool, 687 designs, 4 rounds x 15 purchases, budget 60).
* **Trials 140-179**, a fresh block. N-dev trials 0-139 are used: 0-19 by design notes section
  9, 20-39 by the tuning log (section 11), 40-49 by condition C's development block (section
  13), 100-139 reserved for and partly used by the confirmatory run (section 12). Design notes
  section 12 records that a new N-dev idea starts at 140. This note uses 140-179, so the next N-dev
  idea starts at 180.
* **Both models**, `deepseek` and `mimo`, in separate run directories (`runs/auth-ds`,
  `runs/auth-mimo`). deepseek is the headline, mimo the replication, judged by the same rule
  and reported separately -- as in section 9.
* **40 trials per arm per model**: 80 episodes per model, 160 in all. Registered at 40 rather
  than 20 before any episode existed, for the reason in section 4: at 20 trials almost any
  null falls in the inconclusive bucket, so the experiment could fail to answer the question
  it exists to answer. The extra cost is about $2 in all.
* Both arms are played on the same seeds, so every comparison is paired per trial.

## 4. What is measured

All intervals: `replaylab-eval compare`, 90% two-sided, 10,000 resamples, resampling seed 0,
`--trial-range 140-179`.

**Primary -- does the tool help where authorship is ruled out?** `B1 - B2` on hits, per model.

* **Helps:** the 90% interval lies **wholly above zero**.
* **No measurable effect:** the interval lies wholly below zero, or contains zero while being
  narrow enough to exclude an E-sized effect (its upper bound below +3.85, the smaller of E's
  two estimates).
* **Inconclusive:** any other interval containing zero. Reported as inconclusive, **not** as
  "no effect" -- see the power note below.
* **Headline:** the deepseek verdict; mimo is the replication. If the two disagree, both are
  reported and the disagreement is the finding.

**Power, stated in advance.** The paired per-episode sd of `B1 - A` on N-dev is 9.51 (deepseek)
and 11.78 (mimo) (`docs/prereg-anthropic.md`, amendment of 2026-11-01). The 90% half-width that
implies:

| | n = 20 | n = 40 (registered) |
| --- | --- | --- |
| deepseek, sd 9.51 | 3.50 | **2.47** |
| mimo, sd 11.78 | 4.33 | **3.06** |

At n = 20 the interval would resolve an effect the size of E's deepseek result (+7.15) but not
one the size of E's mimo result (+3.85), so most nulls would be uninterpretable and the
"no measurable effect" branch above would be effectively unreachable. At n = 40 it is reachable:
an interval containing zero with an upper bound below +3.85 becomes a finding rather than a
shrug. That is why this note registers 40.

**What the larger n does and does not buy.** On a fixed pool the episode seed varies only the
alias map and the scripted exploration draws (`docs/caveats.md` entry 6), so extra trials
reduce the **LLM-sampling** component of variance -- which is almost all of it on agent arms,
per-episode sd 9.5 to 11.8 against 0.7 to 2.8 on scripted arms -- and nothing else. n = 40
sharpens the estimate of the effect *on this pool with this model*; it does not make the result
generalise across targets, and the write-up must not read it that way.

**No extension.** If the primary interval still contains zero and is wider than the
"no measurable effect" branch allows, the result is reported as inconclusive and nothing
further is played. Adding trials after seeing the interval would be optional stopping. A later
test needs its own pre-registration and a fresh block, which for N-dev starts at 180.

**Secondary (reported, no threshold).**

* `B1 - B2` on **`distinct_families`** (`--metric distinct_families`): the diversity cost of
  the tool, measured on the pool where its hit value is being tested. Entry 7 of the caveats
  found that on E the agent's hit lead comes with *fewer* distinct families than random
  selection; this says whether the neighbour tool is what drives that.
* `B1` and `B2` against the four baseline policies played on the same seeds
  (`random`, `learner`, `learner_sim`, `learner_sim_purchased`), so any policy beating either
  arm is visible beside the verdict.

## 5. Validity gates, checked before any hit count is read

1. **Completeness.** 80 episodes per model, 0 unrecovered failures, every episode's
   `stop_reason` in {`finished`, `agent_stopped`}.
2. **One configuration per model.** Every record's `config_hash` equals the registered value
   above, and `git diff <git_sha> -- src prompts config` is empty for the recorded sha.
3. **Manipulation check.** In B1, `similar_designs` is called in **at least 32 of 40 episodes**
   per model. On N-dev trials 0-19 it was called in 20/20, at 3.5 calls per episode (deepseek)
   and 7.0 (mimo), so this gate should pass easily -- but if B1 largely ignores the tool on
   this block, the contrast is vacuous and must be reported as such rather than as a null.
   In B2 the call count must be exactly 0 (structural; a non-zero count is a bug).
4. **Spend.** Under the cap, with `spend.json` consistent with the episode costs.
5. **Argument validity.** `flags` and `argument_validity` inspected for tool-use problems
   before hits, as in section 9's gates.

Gate failures are reported and, where recoverable (a provider error), the missing episodes are
replayed; an unrecoverable failure voids the affected cell rather than being worked around.

## 6. Cost

Measured per-episode costs on N-dev, trials 0-19: deepseek $0.0293 (B1) and $0.0283 (A); mimo
$0.0258 (B1) and $0.0320 (A). B2 on E ran dearer than B1 ($0.0225 against $0.0140), plausibly
because the agent compensates with more pool queries, so budget at the high end: **$0.035 per
episode**, 80 episodes, plus the pilot's 1.25 margin gives a projection of about **$3.50 per
model**. Expected actual spend is nearer $2.40 per model at the measured rates. Caps are set at
**$4.00 per model** ($8.00 in all), which covers the projection the run gate checks against
plus headroom for replayed failures.

The four baseline policies on the same trials cost nothing (no model calls).

## 7. The commands

Per model, in order. Nothing here touches a held-out or final-test game.

The baseline rows first (free, and the gates expect them in the same records file):

    uv run replaylab-eval scripted --run-dir runs/auth-ds --games N-dev \
        --policies random,learner,learner_sim,learner_sim_purchased \
        --trials 40 --first-trial 140

    uv run replaylab-eval scripted --run-dir runs/auth-mimo --games N-dev \
        --policies random,learner,learner_sim,learner_sim_purchased \
        --trials 40 --first-trial 140

Then the pilot, which writes the cost projection the run gate requires:

    uv run replaylab-eval pilot --run-dir runs/auth-ds --model deepseek --games N-dev \
        --conditions B1,B2 --trials 40 --first-trial 140 --n 10 --max-usd 4.0

    uv run replaylab-eval pilot --run-dir runs/auth-mimo --model mimo --games N-dev \
        --conditions B1,B2 --trials 40 --first-trial 140 --n 10 --max-usd 4.0

Then the remaining episodes (the pilot's 10 are already recorded; `run` plays only what is
missing, so an interrupted run is resumed by repeating the same command):

    uv run replaylab-eval run --run-dir runs/auth-ds --model deepseek --games N-dev \
        --conditions B1,B2 --trials 40 --first-trial 140 --max-usd 4.0

    uv run replaylab-eval run --run-dir runs/auth-mimo --model mimo --games N-dev \
        --conditions B1,B2 --trials 40 --first-trial 140 --max-usd 4.0

Then the gates, and only after they pass, the comparisons:

    uv run replaylab-eval compare --records runs/auth-ds/episodes.jsonl --game N-dev \
        --a B1 --b B2 --ci 0.9 --resamples 10000 --seed 0 --trial-range 140-179

    uv run replaylab-eval compare --records runs/auth-mimo/episodes.jsonl --game N-dev \
        --a B1 --b B2 --ci 0.9 --resamples 10000 --seed 0 --trial-range 140-179

The secondary on diversity, and the baseline context rows (same two files, each policy in
turn):

    uv run replaylab-eval compare --records runs/auth-ds/episodes.jsonl --game N-dev \
        --a B1 --b B2 --metric distinct_families --ci 0.9 --resamples 10000 --seed 0 \
        --trial-range 140-179

    uv run replaylab-eval compare --records runs/auth-ds/episodes.jsonl --game N-dev \
        --a B1 --b scripted:learner_sim_purchased --ci 0.9 --resamples 10000 --seed 0 \
        --trial-range 140-179

## 8. Disclosure

What is known at the time of writing: N-dev condition B1 on trials 0-19 (20.5 hits deepseek,
24.25 mimo) and on 100-119 (19.15 deepseek), all on the pre-fix configuration; every E result
including `B1 - B2`; and the author and family structure of both pools, as quoted in section 1
and in caveats entry 7. **Condition B2 has never been played on N-dev**, on any block, and no
episode of any condition exists for N-dev trials >= 140. The per-episode costs in section 6
come from the played trials 0-19.

## 9. What the result will mean

* **B1 - B2 above zero on N-dev:** the neighbour tool pays where similarity cannot proxy
  authorship, so E's finding is about pool structure in general rather than about authorship in
  particular. Caveats entry 7's confound stands for E's absolute numbers but no longer
  threatens the mechanism.
* **B1 - B2 at or below zero on N-dev, gates and manipulation check passed:** the tool's value
  is pool-specific, and the authorship confound becomes the leading explanation of E's result.
  Entry 7 would be strengthened, and the write-up would have to say that the advisor's measured
  value rests on a property of one competition's pool.
* **Inconclusive:** reported as such. The honest reading is that 20 paired episodes cannot
  resolve an effect of the size E's mimo cell showed, and the question stays open.

In no case does this change the Anthropic final test, whose configuration is already locked.
