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

**What the write-up should say.** That the toolkit, not the model, is at fault here, and in
which direction it cuts: buying designs selected for the worst score **hurts** the agent, so
E's result (section 15, agent exceeds on all four cells) was obtained *despite* this and
cannot be inflated by it. It also means there is unmeasured headroom in the agent conditions.
What is **not** established is how many hits it cost: the overlap shows the rows were bought
from, not that every such purchase was a mistake, since a returned row carries all its
features and the agent may have chosen within the set on another column. A counterfactual run
would be needed, and it would need its own fresh trial block.
