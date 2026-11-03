# Caveats for the write-up

A running list, added to as things come up, of the conditions under which this project's
results hold and the ways they could mislead. Each entry says what the caveat is, why it
matters, and what the write-up should say. Nothing here is a finding; findings live in
`docs/design-notes.md`.

## 1. The base hit rate decides how much room the method has (2026-10-30)

**What.** Every result here -- the N-dev no-go, the E exceeds, and anything the Anthropic
final test produces -- is measured on a pool with one particular base hit rate, and the rate
is a property of the campaign that produced the pool: its design strategies, its target, how
hard the target is, and how aggressively the pool was pre-filtered before anyone was asked to
choose. Two pools of the same size can sit at very different rates.

**Why it matters.** The rate plausibly decides whether the toolkit helps at all, in opposite
ways at the two ends:

* **Too low**, and the tools have nothing to work with. A model needs hits to fit on (the
  scripted gate asks for 2 hits and 8 purchases, and the agent's `fit_and_rank` has the same
  problem), and `similar_designs` needs hits to be similar *to*. With almost no hits in the
  purchased rows, the advisor degenerates: there is nothing to learn from, and a policy can
  do little better than sort by a fixed column. This is the mechanism we already saw on E,
  where the gate opened in only one round of four.
* **Too high**, and the problem stops being a problem. If a large share of the pool binds, a
  random batch already succeeds, the spread between the best and worst policy collapses, and
  no method can distinguish itself.
* **In between** -- a reasonable but not high rate, perhaps around 20% -- is where selection
  has the most room, and where a decision engine should be worth the most.

So "does an agent add anything?" is not a single question with a single answer. It is a
question whose answer is expected to depend on the rate, and our pools do not span the range
widely enough to map that dependence.

**What the write-up should say.** State the base rate of every pool reported, next to the
result. Frame the N-dev / E split as what it is: two points in a space, not a general law.
Say explicitly that the rate at which a decision engine earns its keep is an open question
this project does not answer, and that a low-rate pool is the regime in which the advisor
tools are most likely to be useless and a high-rate pool the regime in which nothing can
show an advantage. Do not present any single pool's result as "the" answer.

## 2. The player is told which target it is working on, and the targets are public
(2026-10-30)

**What.** `describe_campaign` returns the game's target, so every episode names it. Checked
in the recorded transcripts: `egfr` appears in the E episodes. So the published E result, and
every N-dev result, was obtained with the target named.

**Why it matters.** These are public targets with published competition results, so a model
may be recalling rather than predicting. It cannot be ruled out by withholding the name
after the fact, and the existing results cannot be re-run without the name without becoming
a different experiment.

**What the write-up should say.** Name the exposure. The mitigations: it applies identically
to every condition, so a *paired* comparison between conditions is still interpretable even
if the absolute level is inflated; the designs themselves are unpublished in the Anthropic
pool (0 of 1,440 sequences appear in Proteinbase); and recall of a target's general
difficulty does not tell a model which of 90 unseen designs binds. What it does *not*
support is any claim about absolute hit rates being evidence of de novo reasoning.

## 3. A result's resolution depends on the pool size and the budget share (2026-10-30)

**What.** How much a game can possibly show is bounded by the gap between what random
buying gets and what a perfect oracle gets. On N-dev and E (about 1,100 to 1,340 designs, a
4% to 5% budget share) that gap is roughly 55 hits. On a 90-design single-target pool with a
20% budget share it is about 14. A margin in absolute hits therefore means very different
things in the two settings, and the pre-registered 2-hit margin does not transfer between
them unchanged.

**Why it matters.** A small pool also forces a bigger budget share, which shrinks the gap
further: by the last round much of the pool has been bought, so there is less left to choose
badly from. And the batch size interacts with the scripted fit gate -- too small a batch and
the gate never opens, which we know from E systematically favours the agent over the scripted
ladder. The shape of a game is therefore not a neutral administrative choice.

**What the write-up should say.** Report pool size, budget share, and the random-to-oracle
gap alongside every result. Express a practical margin as a **hit rate within the purchased
batch**, not in absolute hits and not as a share of the oracle gap: the first does not
transfer between pool sizes, and the second turns out to be sub-hit on a small pool (2/55 of
a gap of 13 is 0.47 hits), which is no standard at all. N-dev's 2 hits of 60 purchases is
3.33 percentage points, and that is the standard the Anthropic games carry over
(`docs/prereg-anthropic.md` section 5). When a game's batch size keeps the scripted gate
shut, say so, and treat the comparison as a cold-start result rather than a general one.

## 4. Feature orientation is not described to the player, and it has misled it
(2026-10-30)

**What.** The player sees **raw** feature values everywhere (`query_pool`, `column_summary`,
D7), but the model fits on **oriented** features, where every column has been flipped so that
higher means more likely to bind. Nothing in the agent-facing layer states which columns are
flipped: `higher_is_better` appears nowhere in `interfaces/`, in any tool's output schema, or
in `prompts/system.md`. Two consequences:

* `fit_and_rank` and `fit_custom_model` report `top_weights` / `weights` in the oriented
  space. Demonstrated on a pool built so that **low raw** `f2` is the hit: the reported weight
  on `f2` is **+0.399**. A reader who takes that at face value against the raw values the
  same toolkit shows would conclude the opposite of the truth.
* `query_pool` sorts by raw value with `descending=True` as the **default**, so the default
  sort on a lower-is-better column returns the worst designs first. Its description does warn
  ("Metric values are raw, so a lower ProteinMPNN score is better"), but it names that one
  column in prose rather than deriving the list from `config/features.yaml`, so it is silent
  for any other inverted column and for any new pool.

**Why it matters -- it is not hypothetical.** Across all 350 recorded agent episodes, of the
607 `query_pool` calls that sorted on `proteinmpnn_score` (the one lower-is-better column in
the Proteinbase configs):

| | calls |
| --- | --- |
| ascending -- best designs first | 490 |
| descending with no bounding filter -- **worst designs first** | 113 |
| descending but filtered to good values | 4 |

The 113 affect 74 of 350 episodes, and they are concentrated in **E: 62 of 160 episodes
(39%)** against N-dev's 12 of 190 (6%) -- consistent with E having only two features, one of
them inverted, so the inverted one carries half the signal. Those queries fed purchases: 86
batches were submitted directly after one, and on average **22.9%** of such a batch was drawn
from the worst-first rows (11 batches more than half, one entirely).

**Fixed, for everything played from here on** (commit on `final-test-plumbing`, before any
Anthropic episode): `describe_campaign` and `column_summary` report each column's
`higher_is_better`; the fit tools report weights in the raw-value direction, so a positive
weight means a higher raw value goes with binding; `query_pool`'s description states the
hazard without naming a column; the system prompt says the scores do not all point the same
way. D7 is untouched -- the values shown are still raw. **Nothing was replayed:** the N-dev
and E results stand as results about the configuration they were played on, and the fix
changes the prompt and tool-description hashes, so the Anthropic games carry a different
`config_hash` (`docs/prereg-anthropic.md` section 7).

**What the write-up should say.** That the toolkit, not the model, is at fault here, and in
which direction it cuts: buying designs selected for the worst score **hurts** the agent, so
E's result (section 15, agent exceeds on all four cells) was obtained *despite* this and
cannot be inflated by it. It also means there is unmeasured headroom in the agent conditions.
What is **not** established is how many hits it cost: the overlap shows the rows were bought
from, not that every such purchase was a mistake, since a returned row carries all its
features and the agent may have chosen within the set on another column. A counterfactual run
would be needed, and it would need its own fresh trial block.


## 5. A warning in a tool description does not reliably change what the agent does
(2026-10-30)

**What.** `docs/prereg-orientation.md` played 80 episodes on E across two arms whose only
difference was how clearly the toolkit described feature orientation. In **both** arms about
20% of the `query_pool` calls that sorted on the lower-is-better column asked for the worst
designs -- including the arm whose description said, in plain words, *"Metric values are raw,
so a lower ProteinMPNN score is better"*, naming the exact column in play. Paired, the
difference between the arms was +0.25 calls per episode [-0.25, +0.80] on both models: no
detectable change.

**Why it matters.** The natural fix for an agent misusing a tool is to explain the tool
better, and the natural evidence is that the explanation is now correct and complete. This
block is a counterexample: the explanation was already correct and specific, and the behaviour
did not move. Guidance that a model must read, remember and apply at the right moment competes
with everything else in a long context.

**What the write-up should say.** That the project tried the documentation fix and measured
no behavioural change, so the orientation hazard is better removed than described -- by making
the hazardous call impossible or self-correcting (for example, refusing a descending sort on
an inverted column unless it is asked for explicitly, or returning the direction in the rows
themselves) rather than by warning about it. And that this is one block on one pool: it shows
a warning *can* fail to land, not that documentation never works.

## 6. Pairing by seed does not cancel the agent's own randomness (2026-11-01)

**What.** Every comparison in this project is paired on the episode seed, and the seed depends
only on game and trial, so two conditions meet the same pool, the same hidden outcomes and the
same alias order. The claim that was being made on top of that -- that pairing therefore
cancels luck of the draw -- is true for scripted arms and **false for agent arms**. Measured on
the played blocks:

| comparison | r between arms | sd of the paired difference | sd if treated as independent | change |
| --- | --- | --- | --- | --- |
| `scripted:learner` vs `scripted:learner_sim` (E) | +0.50 | 0.73 | 1.02 | -29% |
| `scripted:random` vs `scripted:learner_sim` (E) | +0.37 | 2.25 | 2.54 | -11% |
| B1 vs `scripted:learner_sim_purchased` (N-dev) | +0.23 | 5.26 | 5.79 | -9% |
| B1 vs `scripted:learner_sim` (E, deepseek) | +0.04 | 9.34 | 9.36 | -0.3% |
| B1 vs A (E, mimo) | +0.01 | 8.65 | 8.69 | -0.4% |
| B1 vs A (N-dev, mimo) | -0.22 | 11.78 | 10.72 | **+10%** |
| B1 vs A (E, deepseek) | -0.42 | 15.30 | 12.86 | **+19%** |

**Why.** What the seed controls is small. On a `split: all` game the pool and the outcomes are
identical across trials, so the seed varies only the alias map -- which design is `d0001`, and
hence the order the pool is presented in -- plus the generator that scripted policies use for
their 20% exploration (`policies/core.make_rng`, a fresh generator per round, so a policy that
draws a different number of values in one round still gets the identical stream in the next).
No fit consumes randomness: the learner is lbfgs with no `random_state`, `fit_custom_model` is
lbfgs or saga with `random_state=0`, and its validation is leave-one-out, which is exhaustive
rather than sampled. The one shuffling RNG in the project, `StratifiedGroupKFold` in
`datasets/splits.py`, ran once offline to build the committed folds.

So for two scripted policies the shared alias order is most of the variance and pairing earns
its keep. For an agent it is almost none of it: the per-seed sd is 0.7 to 2.8 hits on scripted
arms and 4.4 to 9.3 on agent arms, and the difference is LLM sampling, for which no seed is set
(none of the providers used here offers a reliable one). That noise is independent across arms,
so differencing two agent arms adds it twice while cancelling nothing.

**What the write-up should say.** That pairing is justified by a shared *exogenous* factor --
same pool, same outcomes, same presentation order -- and not by the claim that it removes the
agent's variance, which it does not. The paired bootstrap stays the right interval for the mean
paired difference either way; where the correlation is negative it is simply **wider** than an
unpaired analysis, which is the direction that cannot manufacture a result. No published number
changes; the sentence claiming that "luck of the draw cancels out" does.

## 7. What `similar_designs` exploits, and what it costs
(2026-11-03)

**What.** Section 17 of the design notes established that on E the advisor's entire measurable
value is `similar_designs` (B1 - B2 = +7.15 ds, +3.85 mimo, and B2 falls back to A's level).
That is a statement about which tool, not about what the tool exploits. The pools answer the
second question, and the answer qualifies the finding in three ways.

**Narrowed 2026-11-03.** The authorship and diversity qualifications below were written from E
alone and read as properties of `similar_designs`. Neither holds up. The N-dev ablation (design
notes section 18) shows the tool pays where authorship cannot explain it, and running the
matched `B1 - B2` contrast on both pools shows it costs no diversity on either. The diversity
deficit against `random` is real but belongs to hit-chasing in general. Each paragraph now says
which pool and which contrast it is about.

**Hits are clustered, so a neighbour of a hit is a good bet by construction.** From
`data/splits/clusters.csv` and `outcomes.parquet`:

| | E | N-dev target |
| --- | --- | --- |
| pool hit rate | 0.162 | 0.100 |
| hit rate within a family holding >= 1 hit | **0.427** | **0.348** |
| families with >= 1 hit / all families | 60 / 499 | 70 / 631 |
| mean designs in a hit-holding family | 5.2 | 4.2 |

So "fetch the neighbours of a hit and buy them" draws from a 2.6x enriched subpool on E (3.5x
on nipah). The agent must still find the first hit and choose to follow it up -- the buy
attribution in section 16 shows it doing exactly that, 24-30 neighbour-sourced purchases per
episode at a 0.67 hit rate against 0.24-0.26 for the rest -- but the size of the payoff is a
property of how the pool was assembled, not something the agent learned about binding.

**The similarity measure is entangled with authorship, far more on E than phase 0 suggested.**
Phase 0 (design notes section 1) measured nipah only: nearest 3-mer neighbour is the same
author 44.7% of the time, 51.6% among hit pairs. Run on E for the write-up, 2026-11-03, with
`datasets/author_checks.nearest_neighbor_table`:

| | E | nipah |
| --- | --- | --- |
| same-author rate, both authors known | **0.743** (n=564) | 0.447 (n=1030) |
| chance baseline, sum of squared author shares | 0.047 | 0.007 |
| same-author rate among hit pairs, both known | **0.798** (n=84) | 0.516 (n=64) |
| designs with no author | 192 of 826 | 0 of 1030 |
| largest two authors | 96 + 84 designs | 10 + 10 |

(`same_author` requires both authors present and equal, and `same_author_rate_all` is taken
over pairs where both are known, so missing authors never inflate it. The hit-pair rate in
`summarize_author_checks` is over *all* hit pairs, so E's 0.713 there is a floor; 0.798 is the
both-known figure.) A 16x lift over chance on E means a family is largely one author's design
series. **On E this cannot be separated with this data,** and it is not a leak the safeguards
can close: no tool exposes `meta_author`, the entanglement is inside the 3-mer measure itself.

**But authorship is not the mechanism.** Tested directly on nipah, whose hit-holding families
span 11.2 authors on average and 0 of 6 of which are single-author, so an authorship effect
cannot operate there (design notes section 18, pre-registered, trials 140-179, 40 trials per
arm): `B1 - B2` on hits is **+5.30 [3.15, 7.40]** (deepseek, registered) and +4.88 [2.15, 7.47]
(mimo, descriptive -- its cell failed a completeness gate). The tool is worth about 5 hits in 60
purchases where riding one author's series is impossible. So the correlation above does **not**
license "the gain is a pipeline artefact": the gain survives without it. What stays unresolved
is narrower -- on E specifically, how much of *that* pool's +7.15 / +3.85 rides authorship is
not separable, so E's own effect size carries the confound even though the mechanism does not
depend on it.

**On E, the hit lead comes with less family diversity than random selection.** Distinct
families among the 60 purchases, E, from `reports/agent_study.csv`:

| condition | hits | distinct families | hits per family |
| --- | --- | --- | --- |
| B1 deepseek | 22.80 | 6.70 | 3.40 |
| B1 mimo | 24.25 | 5.75 | 4.22 |
| A deepseek | 16.10 | 5.20 | 3.10 |
| `scripted:random` | 9.90 | **7.45** | 1.33 |

`random` finds more distinct binding families than any agent cell while finding less than half
the hits. This is not a defect in the agent: hits is the only scored objective and it rewards
concentration. But if the purpose of a first campaign is to find as many *distinct* binding
families as possible, to be narrowed in a follow-up, the objective rewards the wrong behaviour
and these rankings could invert.

**But `similar_designs` is not what costs the diversity.** The table above compares agent cells
against `random`; the matched contrast is `B1 - B2`, and removing the tool never *raises* the
family count:

| `distinct_families` | E deepseek | E mimo | N-dev deepseek | N-dev mimo |
| --- | --- | --- | --- | --- |
| `B1 - B2` (the tool) | +1.30 [-0.50, 3.15] | +0.10 [-1.25, 1.50] | **+3.58 [1.32, 5.83]** | +4.28 [1.93, 6.53] |
| `B1 - random` | -0.75 [-2.35, 0.90] | -1.70 [-3.25, -0.20] | +12.10 [10.25, 14.05] | +15.82 [13.72, 17.85] |
| `A - random` | **-2.25 [-3.45, -1.10]** | -1.45 [-2.55, -0.40] | not played | not played |

Strongly positive on nipah, straddling zero on E. And E's deficit against `random` is **worst in
condition A**, which has no similarity tool at all (5.2 families against B1's 6.7). So the
concentration comes from chasing hits with whatever tools are to hand -- which is what a
hits-only objective rewards -- and the neighbour lookup is not its cause.

**Correction (same day).** An earlier revision of this entry read E's `B1 - random` against
nipah's `B1 - B2` and called the difference a sign reversal. Those are different contrasts; the
matched one is null on E, so there was no reversal to report. The error was caught by plotting
the matched contrast on both pools, which is why the figure exists.

Why the two are easy to confuse: hits per family is 3.10 to 4.22 on E against 1.13 to 1.23 on
nipah (`random`: 1.33 and 1.04). Where hits are spread thinly over small families, concentration
has nowhere to go and more hits almost mechanically means more families; on E, hit-holding
families are large enough that an agent can spend a batch inside one. So the **diversity
objective** worry stays conditional on pool structure -- real where hit-holding families are
large relative to the batch -- but it is a worry about scoring hits, not about this tool.

**What the write-up should say.** That the tool's value rests on **pool structure** -- hits
being clustered, so a neighbour of a hit is a good bet by construction -- and that this is a
property of how these pools were assembled rather than something the agent learned about
binding. "The agent learns sequence-activity relationships" remains **not** supported: nothing
here distinguishes exploiting a neighbourhood from understanding one. What the write-up should
*not* say any more is that the gain is an authorship artefact or that this tool costs diversity;
the first is contradicted on nipah, the second by the matched contrast on both pools. Where E's
own numbers are quoted, the authorship entanglement still belongs beside them as a limit on that
effect size, and the diversity deficit against `random` belongs wherever agents are compared
with `random` on E -- attributed to the objective, not to a tool.

**What is settled and what is not.** Settled: authorship is not the mechanism, and the diversity
deficit is not caused by `similar_designs` (both above). Open: whether a
**diversity-scored game** reorders the players where it bites. `distinct_families` is already
computed per episode by `engine/scoring.py` and carried in every record, so scoring it needs no
new measurement, only a new objective; running the existing baseline policies under it costs no
agent spend and would show whether they reorder. Any agent claim needs its own pre-registration
and a fresh trial block: E is used through trial 159 (100-119 by design notes sections 15 to 17,
120-139 left unplayed, 140-159 by `docs/prereg-orientation.md` section 3) so a new E idea starts
at 160, and N-dev is used through 179 (140-179 by section 18) so a new N-dev idea starts at 180.
A structural similarity measure (TM-score, Foldseek) would separate shared backbone from shared
pipeline on E, but Proteinbase's structure-comparison columns cannot supply it: ~20% coverage
and a missingness leak (entry 4's rule; design notes section 1).
