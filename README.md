# replaylab

**Analysis tools, exposed over MCP, to help agents run protein design campaigns**

An agent toolkit, an evaluation harness and a set of baseline policies, scored by replaying
real lab outcomes.

*Abstract.* Labs that take experiments through an API, such as Adaptyv, expect agents to run
design–test campaigns: each round, choose which computationally designed binders to buy for
testing, on a fixed budget. We ask whether giving such an agent specialised **advisor tools**
(fit a model on its own results, find designs similar to its hits, get a batch proposal from a
baseline policy, estimate hit rates) makes it find more binders than the same model with only
basic **campaign tools** (inspect the pool, buy, wait). We replay public Proteinbase
outcomes, so that every purchase can be scored without a lab, and compare the two tool sets on
the same seeds, on two targets (Nipah-G and EGFR), with two inexpensive models.
**The advisor tools add hits in all four model × target cells, with intervals above zero in
three of them: +5.1 hits per 60 purchases pooled over the four (90% interval about
[3.0, 7.2]; pooled post-hoc, not the pre-registered outcome).** As a secondary comparison,
against hand-coded baseline policies the picture is two-sided: the agent falls short of the
strongest baseline on Nipah-G and beats every baseline by a wide margin on EGFR, where the
baselines stall at a cold start. Section 2.5 explains how
the primary outcome was chosen.

![B1 (campaign + advisor tools) minus A (campaign tools only), paired by seed: +2.45 and +6.80 on Nipah, +6.70 and +4.50 on EGFR, +5.11 pooled over all four cells, with 90% paired intervals](reports/figures/advisor-effect.svg)

## 1. Introduction

[Adaptyv API: Giving everyone access to a real lab](https://www.adaptyvbio.com/blog/adaptyv-api)
names agents as its primary users:

> "agents can submit experimental campaigns, retrieve results, and reason over
> them to plan the next round of designs"

Such an API already lets an agent read a campaign: submit designs, fetch results, look at what
came back. What it leaves to the model is the analysis. Turning a table of scores and a few
dozen results into the next batch is a statistical task, and a model that reasons from its
built-in knowledge alone has to do that arithmetic in its head. We hypothesised that a small
set of analysis tools (fit a model on the campaign's own results, follow up on its own hits,
estimate hit rates) would let it make better-informed choices, and therefore find more
binders, more cheaply than a bigger model or longer reasoning would.

Testing that needs three things, which this project provides:

* an **environment** that plays a campaign faithfully, with outcomes hidden until a design is
  bought;
* a **toolkit** exposed to the agent over MCP, split into campaign tools and advisor tools;
* a set of **baseline policies**, hand-coded selection rules that see exactly what the agent
  sees, so "is the agent any good?" has a reference point that does not depend on the agent.

## 2. Methods

### 2.1 Campaigns and games

A **game** is one simulated campaign built from a public dataset of designed binders and their
recorded lab outcomes. It fixes the target, the pool of candidate designs, the budget, the
number of rounds and the reporting delay (all in `config/games.yaml`). The outcomes stay hidden:
the agent sees one only for a design it has bought and that has been delivered.

Every player, agent or baseline, gets:

* a **pool** of computer-generated protein designs;
* a handful of **computational features** for each design: fast predicted scores that rank
  candidates plausibly but not reliably;
* a **budget** and a fixed number of **rounds**.

Each round the agent buys a **batch** of designs, and the lab reports which ones are **hits**:
designs that actually bind the target. Depending on the game, results come back at once or
after a delay; a design that has been bought but not yet reported is *pending* and cannot be
bought again. The goal is to **find as many hits as possible before the budget runs out**, and
to recognise when stopping early is the better use of the money.

Two games carry the results. **Nipah-G** (`N-dev`, 687 designs, about 10% hits, seven score
columns) is the development pool. **EGFR** (`E`, 826 designs, about 16% hits, two score
columns) is the transfer set. Both run for 4 rounds of 15 purchases, 60 in all, and deliver
their results at the end of each round.

### 2.2 Tools and conditions

Each turn the agent sees the campaign state (budget and rounds left, what it has bought, what
is still pending) and either calls a tool or submits a batch. We offer two families of
tools:

* **campaign tools**: inspect the pool and its columns, buy, wait, review the history, stop;
* **advisor tools**: obtain a batch proposal from a baseline policy, fit a logistic model on
  the results so far, find the designs most similar to a hit (by a sequence-similarity index),
  and estimate hit rates.

Eight campaign tools and six advisor tools make fourteen. A fifteenth, used by one
experimental condition only, lets the agent configure its own model.

The **conditions** differ only in which of those tools the agent gets; they share the same
prompt, model and limits:

| condition | tools |
| --- | --- |
| **A** | campaign tools only |
| **B1** | campaign + advisor tools (the headline agent) |
| **B2** | B1 without `similar_designs`, the similarity lookup |
| **C** | B1 with `fit_and_rank` replaced by `fit_custom_model`, which lets the agent set up its own logistic regression |

### 2.3 Baseline policies

The baselines are hand-coded selection rules. They see exactly what the agent sees and choose a
batch each round; in the run records they appear as conditions named `scripted:<policy>`. We
consider six:

1. **`random`**: a uniform random draw from the designs not yet bought.
2. **`fixed:<column>`**: the top of the pool by one score column, the same every round (one
   policy per column).
3. **`learner`**: fits a logistic regression on its own delivered results and buys the
   designs with the highest predicted hit probability, drawing 20% of each batch at random to
   keep exploring. It may fit only once it has **at least 2 hits among at least 8 delivered
   purchases** (the *model gate*). The 2-hit half is close to forced: below it only one class
   is present and a binary logistic regression has nothing to separate, so the fit is undefined
   rather than merely noisy. The 8-purchase half is a judgement call, meant to stop the policy
   chasing a model fitted on a handful of rows. Until the gate opens it falls back to
   `fixed:<first_column>`: the top of the pool by one score column named per target before the
   experiments (`boltz2_ipsae` for Nipah-G, `esmfold_plddt` for EGFR; `config/features.yaml`).
4. **`learner_sim`**: as `learner`, but it ranks each design by the average of two ranks, its
   predicted probability and its sequence similarity to the hits found so far.
5. **`learner_purchased`** and
6. **`learner_sim_purchased`**: as the two above, but they standardise the features (centre
   and scale them) on the designs bought so far rather than on the whole pool.

Because every player reads the same pool and the same hidden outcomes on the same seeds, the
comparison is fair by construction.

### 2.4 Evaluation design

We followed a fixed sequence, writing down each stage's question and analysis plan before
seeing its data:

1. **Preliminary analysis.** Which score columns can be trusted, and what sequence similarity
   actually measures.
   * *Coverage and missingness leaks.* A column was kept if at least 90% of designs had a
     value for it, and dropped if the hit rate among designs *with* a value differed from the
     hit rate among designs *without* one by more than 5 points. That second rule catches a
     **missingness leak**: a column whose *gaps* predict the answer. A metric only gets
     computed for designs that reached some stage of a pipeline, so "this field is empty" can
     itself carry information about the design — and a model can learn from the pattern of
     blanks without the values telling it anything. The structure-comparison columns failed
     badly: designs with a value had a 1.6% hit rate against 17.4% for designs without one, so
     merely *having* been scored was a strong negative signal. They were excluded.
   * *What similarity measures.* In part, authorship. On Nipah-G, a design's nearest
     neighbour by 3-mer similarity is the same author **44.7%** of the time, against a 0.7%
     chance baseline, and 51.6% among pairs of hits. So similarity partly reveals *who* made a
     sequence, not only how good it is. Section 3.4 returns to this, with the much stronger
     figures for EGFR.
2. **Frozen setup**, committed before any agent episode:
   * *Split.* We cut Nipah-G into three folds with a stratified group k-fold
     (`StratifiedGroupKFold`), which groups designs into sequence families by 3-mer similarity,
     puts every family in exactly one fold, and balances hits across folds. Two folds form the
     development pool `N-dev`; we hold the third back for a single final run.
   * *Features.* The visible score columns per target, in `config/features.yaml`: for Nipah-G
     the Boltz-2 ipTM, ipSAE, min-ipSAE and pLDDT, the Boltz-2 binder shape complementarity,
     the ESMFold pLDDT and the ProteinMPNN score; for EGFR the ESMFold pLDDT and the
     ProteinMPNN score. Sequence length and design class are visible in every game.
   * *Model settings* of the baseline learners, in `config/frozen_v1.yaml`: L2 logistic
     regression with C = 0.1, the lbfgs solver, no class weights, 20% exploration, the 2-hit /
     8-purchase gate, and a similarity weight of 0.5. We chose them on `N-dev` only.
   * *Decision rule* for agent against baseline: a 90% paired interval judged against a 2-hit
     margin, with validity gates (`docs/design-notes.md` section 9).
3. **Development pool (Nipah-G).** Agents against the baseline policies, with two flash models
   (`deepseek-flash` and `mimo-v2.6-flash`) and two rounds of prompt and tool tuning.
4. **Transfer set (EGFR).** Configuration frozen, a fresh block of never-played episodes.
5. **Ablations.** Remove all advisor tools (condition A) or a single one (`similar_designs`,
   condition B2) to see what carries the result. We repeated the single-tool ablation on
   Nipah-G, whose families are multi-author, to test whether that tool's value is an authorship
   artefact (section 3.4).
6. **Final test on the Anthropic dataset: built, pre-registered, deliberately not yet played.**
   Four single-target games (Nipah-G, EGFR, BBF-14, BHRF1; 90 designs each, sharing no
   sequence with the development pools), conditions A, B1 and B2, both models, 20 trials, a
   pooled headline, every input file pinned by hash, and each game behind a run-once lock. Its
   primary outcome is the tool contrast B1 − A. See
   [`docs/prereg-anthropic.md`](docs/prereg-anthropic.md).

We pair every comparison by seed and summarise it with a 90% paired bootstrap interval over
20 trials per cell. Pairing gives both arms the same pool, outcomes and presentation order; it
does not remove the model's own sampling noise, which no provider lets us seed
([`docs/caveats.md`](docs/caveats.md) entry 6).

### 2.5 Choice of the primary outcome

For transparency: we registered the experiments on Nipah-G and EGFR with a different primary
outcome, comparing the agent's performance with the strongest baseline policy of each pool. We
declared B1 − A (the net effect of adding advisor tools to the setup) before its data existed
in every case, but as a secondary, descriptive comparison with no pass/fail rule attached, and
computed the pooled figure over the four cells afterwards.
As the results came in it became clear that "the strongest baseline" is known only in
hindsight: the ranking of the baselines reverses between the two pools (section 3.3), so a
practitioner could not have picked the winner in advance. The tool contrast compares the same
model with and without the advisor tools and needs no such oracle, and it is the question the
project set out to answer. We therefore report it first. The change adds hindsight to the
presentation, not to the numbers: apart from the pooled figure, every value below comes from
comparisons we declared before playing their episodes. The unplayed final test registers
B1 − A as its primary outcome (`docs/prereg-anthropic.md`, amendment of 2026-11-01), so it
will confirm or refute this result on fresh data.

### 2.6 Keeping the answers sealed

Lab outcomes are the answer sheet: an agent that reads them early is not being evaluated, it
is cheating. To prevent that, we built several independent safeguards.

* **Access to outcome data protected structurally in the code.** Only one function reads the outcome
  table (`load_outcomes`), and the outcomes it returns are held privately by the replay
  backend, which releases one only when its design has been bought and its delivery round has
  come. Everything a tool or a baseline policy works from is a read-only snapshot of the
  campaign that contains delivered outcomes only. A test parses every module's imports and
  fails the build if a package imports one it is not allowed to (for example, if the advisor
  code imported the backend that holds the outcomes). Every tool response is a declared schema
  that cannot carry sequences, real identifiers or undelivered outcomes, and dedicated tests
  check that.
* **Pseudonymous ids.** Agents see aliases such as `d0001`, reshuffled by every episode's
  seed, so ids cannot be matched across runs or to outside knowledge.
* **Configuration fingerprints.** Every run records a `config_hash` over the model, prices,
  prompts, conditions, tool descriptions and limits, plus the git commit. The hash cannot be
  turned back into a configuration, but any candidate configuration can be checked against
  it, so two results with the same hash provably come from the same configuration.
* **Append-only experiment logs.** One JSON line per finished episode, never edited. Episodes
  that fail (provider error, truncated reply, spending cap) are not recorded and are played
  again, so a run cannot be curated after the fact.
* **Frozen artifacts.** We commit the splits, feature lists and model settings and check them
  on every run; three separate guards refuse the held-out fold until the one final run.
* **Pre-registration.** We write down and commit each experiment's question, trials, decision
  rule and validity gates before its episodes exist, and check the gates before reading any
  score.

## 3. Results

**In brief.** The advisor tools help the agent on both pools (Nipah-G and EGFR) and with both
models. Measured against the baseline policies, the agent **fails** on Nipah-G, where the
strongest baseline (`learner_sim_purchased`) is strong, and **clearly passes** on EGFR, where
every baseline is weak. On EGFR, roughly half of the agent's lead comes from its own judgment
and half from the advisor tools, and the advisor half is almost entirely the similarity lookup.
Section
4 lists the limits these numbers come with.

### 3.1 Do the advisor tools help? (B1 − A)

| pool | deepseek-flash | mimo-v2.6-flash |
| --- | --- | --- |
| Nipah-G (trials 0–19) | +2.45 [−0.95, +5.85] | **+6.80** [2.6, 11.0] |
| EGFR (trials 100–119) | **+6.70** [1.25, 12.2] | **+4.50** [1.5, 7.6] |

Hits per 60 purchases, B1 minus A on the same seeds, 90% paired intervals. Three of the four
intervals sit above zero; deepseek on Nipah points the same way but is not conclusive on its
own. Pooling all four cells gives **+5.1 hits**, with an interval of roughly [3.0, 7.2]
(computed after the fact; see section 2.5).

### 3.2 Against the baseline policies

![Agent minus the strongest baseline of each pool, paired by seed: on Nipah -6.20 and -6.90 for deepseek and -2.45 for mimo, on EGFR +12.25 for deepseek and +13.70 for mimo, with 90% paired intervals](reports/figures/two-sided-result.svg)

**Nipah-G: fail.** The bar is `learner_sim_purchased`, the strongest baseline on this pool at
**26.4 hits** per 60 purchases (mean over 100 seeds). The agent (deepseek-flash, B1) falls
**6.2 hits short** on trials 0–19 (90% interval [−9.0, −3.35]), and a confirmatory rerun on a
fresh block, trials 100–119, gives −6.9 [−8.8, −4.95]: a clear fail against the 2-hit margin.
mimo-v2.6-flash misses the same margin narrowly, −2.45 [−5.95, +1.0] on trials 0–19; it was
the replication model and had no confirmatory rerun. The agent still plays competently: it
beats random selection by 14 to 18 hits; deepseek matches the plain `learner`
(+1.2 [−0.8, +3.35]) and mimo beats it (+6.05 [2.8, 9.2]). It just does not beat the strongest
baseline here.

A capability gap was the obvious suspect, but the evidence does not support it. We could not
test Claude Opus 5.5 (the provider declined the requests), and Claude Sonnet 5.5 did no better
than the flash models (−10.4 against `learner_sim_purchased` on trials 0–9). Two rounds of
prompt and tool tuning did not help either: the first had no detectable effect, and we reverted
the second, which made things worse.

**EGFR: pass, in every cell.** On a fresh block of episodes, with the configuration frozen, the
bar is `learner_sim`, the strongest baseline on this pool at about **10.4 hits**:

| cell | agent − bar | 90% interval | verdict |
| --- | --- | --- | --- |
| deepseek, B1 | **+12.25** | [8.85, 15.55] | exceeds |
| deepseek, C | +12.35 | [9.2, 15.25] | exceeds |
| mimo, B1 | **+13.70** | [12.05, 15.35] | exceeds (20 of 20 trials) |
| mimo, C | +11.85 | [9.1, 14.5] | exceeds |

The agent reaches 23–24 hits out of 60 purchases, a 38–40% hit rate on a pool where 16% of
designs bind, using only two predicted-score columns (an ESMFold confidence and a ProteinMPNN
score) plus the sequence length and design class visible in every game.

### 3.3 Why the two pools disagree

The model gate (section 2.3) applies to the baseline learners and to the agent's `fit_and_rank`
tool alike: no fit before 2 hits among 8 delivered purchases. Until then a baseline learner
buys the top of the pool by its first column. On Nipah-G that fallback finds 5 hits in round 1,
the gate opens for round 2, and the learners are strong. On EGFR the fallback finds 1 hit in
round 1 and none in round 2, the gate opens only for round 4 of 4, and the learners spend almost the
whole campaign without a model. The ranking of the baselines reverses between the two pools:

![Mean hits per baseline policy on Nipah and on EGFR: the best policy on Nipah, learner_sim_purchased at 26.4, is the worst on EGFR at 4.5, where random at 9.6 beats learner at 7.4](reports/figures/ladder-inversion.svg)

The agent escapes that cold start. Round 1 is blind for everyone. In round 2 the agent filters
the pool directly with its query tools instead of following one column, finds its first hits,
and passes the gate by the end of the round (6.9 hits on average for B1); it fits from round
3, while the baseline learner first fits in round 4, its last. Rounds 3 and 4 compound the
lead.

![Cumulative hits at each round boundary on EGFR: the baseline bar stays near one hit through rounds 1-2 and reaches 10.6 by round 4, while B1 climbs to 22.8 and campaign-tools-only A to 16.1](reports/figures/e-round-curves.svg)

**Why the purchased-scaling variant flips.** `learner_sim_purchased` gains 11.9 hits over
`learner_sim` on Nipah-G and loses 5.9 on EGFR. This is not seed noise: the gain replicates on
the delayed Nipah game (+8.7) and on an earlier legacy game (+6.3), and on EGFR the loss
persists over a doubled horizon (8 rounds: still −2.6), with 100 seeds per estimate. The cause
is not established. One plausible mechanism, untested: the model's L2 penalty acts on the
standardised features, so the scaler decides how strongly each feature is shrunk. On EGFR the
only fit uses 45 designs that were all bought by the fallback sort on ESMFold pLDDT, so their
spread on that column is narrow, and standardising on them distorts the one fit that matters.
On Nipah-G the purchased set is broader (fits from round 2, with 20% random exploration in each
batch). `docs/design-notes.md` sections 6 and 8 keep the variant tracked as unexplained.

### 3.4 What carries the EGFR result

![Paired differences on EGFR: B1 minus A is +6.70 and +4.50, B1 minus B2 is +7.15 and +3.85, while B2 minus A straddles zero on both models](reports/figures/e-ablation.svg)

* **Campaign tools alone (A) already beat the bar:** +5.55 (deepseek) and +9.2 (mimo). The
  cold-start escape needs no advisor.
* **The advisor tools add on top (B1 − A):** +6.7 and +4.5. With the A gain, that makes
  the total roughly half the agent's own judgment and half the tools; on deepseek the tools
  supply more than half, on mimo less.
* **Almost all of the advisor's value is one tool (B1 − B2):** removing `similar_designs`
  costs 7.15 and 3.85 hits, and B2 falls back to the level of A. On this pool, the advisor helps
  by *following up on hits through the similarity index*; the rest of the kit is close to
  neutral.

**What that tool is actually exploiting, and what it costs.** Hits are clustered. On EGFR the
pool binds at 16.2%, but inside a sequence family that already contains one hit the rate is
**42.7%** — a 2.6-fold lift (Nipah-G: 34.8% against 10.0%). Only 60 of EGFR's 499 families hold
any hit at all, and those families average 5.2 designs. So "ask for the neighbours of a hit and
buy them" harvests a family whose odds are already known to be good. The agent still has to
find the first hit and decide to follow it up, but the payoff for doing so is a property of how
this pool is built, not something the agent learned about binding.

EGFR raised two worries about that, both serious enough to test rather than argue about.

* *Is the structure biological or sociological?* On EGFR a design's nearest neighbour by 3-mer
  similarity is **the same author 74.3% of the time**, against a 4.7% chance baseline — and
  79.8% among pairs of hits. (Figures for pairs where both authors are known, 564 and 84 pairs;
  computed for this write-up, since the preliminary analysis measured only Nipah-G, where the
  rate is 44.7%.) EGFR's authorship is also concentrated: two authors account for 180 of its 826
  designs. A family is therefore largely one author's design series, so following up on a hit
  might be exploiting a shared pipeline rather than a shared backbone. No tool shows the agent
  author metadata — the entanglement sits inside the similarity measure itself, which is why the
  safeguards in section 2.6 do not help.
* *Does the lead in hits cost diversity?* Counting distinct families among the designs bought,
  B1 ends an EGFR campaign with 6.7 (deepseek) and 5.8 (mimo) families against **7.5 for
  `random`** — while finding more than twice the hits. The agent concentrates its purchases
  where hits already are, which is what maximising hits rewards. If the point of a first
  campaign is instead to find as many *distinct* binding families as possible, to be narrowed in
  a follow-up, our objective would be rewarding the wrong behaviour.

**We tested both on Nipah-G, and neither survives.** Nipah-G is the control EGFR cannot be: its
hit-holding families average **11.2 authors** and none is single-author, so there is no author
series to ride. We pre-registered the contrast before running it
([`docs/prereg-authorship.md`](docs/prereg-authorship.md)), on a fresh block of 40 trials per
arm.

| Nipah-G, B1 − B2 | deepseek | mimo |
| --- | --- | --- |
| hits | **+5.30** [+3.15, +7.40] | +4.88 [+2.15, +7.47] |
| distinct families | **+3.58** [+1.32, +5.83] | +4.28 [+1.93, +6.53] |

* **Authorship is not the mechanism.** The tool is worth about five hits in sixty purchases on a
  pool where riding one author's series is impossible. The EGFR correlation is real, but it does
  not license calling the gain a pipeline artefact — the gain is there without it. What stays
  unresolved is narrower: how much of *EGFR's* +7.15 and +3.85 rides authorship cannot be
  separated with this data, so that effect size keeps the caveat even though the mechanism does
  not depend on it.
* **The diversity cost inverts.** On Nipah-G the similarity tool makes the agent find *more*
  families, not fewer. The ratios say why: hits per family is 1.13–1.23 here against 3.40–4.22
  on EGFR. Where hits are spread thinly over many small families, more hits almost mechanically
  means more families and concentration has nowhere to go; on EGFR, hit-holding families are
  large enough that an agent can spend a batch inside one. So the worry is **conditional on pool
  structure** rather than inherent to hit-scored selection.

(deepseek is the registered result on both rows. mimo's cell failed a completeness gate — one
episode hit the turn limit with 45 of 60 purchases — so by the rule we wrote in advance it is
reported as descriptive, not registered; dropping the pair moves hits to +4.74 [+1.92, +7.41].
`docs/design-notes.md` section 18.)

What survives is the first paragraph: the tool's value rests on hits being clustered, which is a
property of how these pools were assembled. Nothing here distinguishes exploiting a
neighbourhood from understanding one, so **"the agent learns sequence-activity relationships"
remains unsupported.** Whether a diversity-scored objective reorders the players where
concentration is possible is still open — `distinct_families` is recorded per episode but has
never been scored — and it needs its own pre-registration and a fresh trial block.

### 3.5 How the agents play

The transcripts show the same style in every run: several dozen tool calls per campaign, most
of them narrow pool queries chained together (filter, sort, re-query); `describe_campaign`
before the first purchase in every episode; a model fit before most later batches. The agents
prefer to **pick their batches by hand** from their own filtered views, calling the
batch-proposal tool only a few times per campaign. Condition A plays the same way without any
advisor: about twelve filtered queries per episode, making every selection through `query_pool`
and `column_summary`. The two models differ in habits; for example, one checks its estimated hit
rate several times per campaign, the other rarely.

### 3.6 A defect in our own toolkit

Reading the transcripts exposed a flaw on our side rather than the model's. One score column,
the ProteinMPNN score, is better when lower. Before fitting, the model *orients* the features:
it multiplies such columns by −1, so that a positive weight always means "higher value, higher
hit probability". The agent, however, saw the raw values, and nothing told it which columns had
been flipped. Two traps followed. `fit_and_rank` reported a *positive* weight on ProteinMPNN,
which against the raw values reads as "higher is better", the opposite of the truth. And
`query_pool` sorts in descending order by default, so a default sort on ProteinMPNN returned
the worst designs first.

Across all 350 recorded episodes, 607 pool queries sorted on ProteinMPNN; **113 of them asked
for the worst designs first** (descending, with no filter bounding the values), and the agents
submitted 86 batches straight afterwards. The damage concentrated on EGFR, which has only two
score columns and so relies on ProteinMPNN for half its signal: 39% of EGFR episodes contain
at least one such query, against 6% on Nipah-G. Buying the worst-scoring designs costs hits, so
the agent earned its EGFR result despite this defect, not because of it.

We then fixed the outputs (each column now reports `higher_is_better`, the fit tools report
weights in the raw-value direction, and the prompt warns that the scores do not all point the
same way) and tested the fix against the old toolkit on a fresh block of 80 episodes. The test
was registered with a *manipulation check*: before reading any hit count, confirm that the fix
changed the behaviour it targets, here the number of worst-first ProteinMPNN queries per
episode. It did not (+0.25 per episode, interval spanning zero), so by the registered rule the
hit difference (pooled +0.17 [−2.70, +3.00]) cannot tell us whether the fix helps: the fix did
not change what the agents did.

The old toolkit had already warned in plain words, in the `query_pool` description, that a
lower ProteinMPNN score is better. In both arms the agents still aimed about 20% of their
ProteinMPNN sorts at the worst designs. This taught us a lesson about tool design: a warning the
agent has to read and remember at the right moment does not reliably change what it does, so a
hazard in a tool is better **removed** (for example, by refusing a worst-first sort on such a
column unless it is asked for explicitly) than documented. The fix we tested corrected the
reported weights but left the worst-first sort in place, only better documented.

## 4. Limitations

* **Offline only.** We replay everything from recorded outcomes, so the campaign is simpler
  than a real one in several ways: the pool is fixed (the agent can only choose among designs
  that were already made and tested, never request new ones), every design has a recorded
  outcome, and results arrive on schedule, without failed expressions, repeat measurements or
  variable turnaround. Running against the live lab API is future work.
* **Two pools.** Nipah-G and EGFR (plus small stop games) are two targets. Seed-level
  intervals measure luck within *one* pool, not performance across targets. The Anthropic test
  adds four targets, but from a single design campaign, so they are not four independent
  replications.
* **Small samples.** Each cell holds 20 paired episodes, except the authorship ablation in
  section 3.4, which holds 40. Some intervals are wide, and one cell (mimo on Nipah) still
  straddled its margin at the largest planned sample.
* **The baselines could be stronger.** Before the model gate opens, a baseline learner buys
  the top of one fixed column — on EGFR that finds 1 hit in round 1 and none in round 2. A
  fallback that spread its early purchases across sequence families instead might reach the
  gate far sooner and close much of the agent's lead. The baselines are a reference point, not
  an attempt at the best possible script.
* **Hits are the only scored objective.** They reward concentrating purchases in families that
  already contain a hit. The agent does that where the pool allows it — on EGFR, where
  hit-holding families are large — but not on Nipah-G, where the same tool raises family
  diversity (section 3.4). A campaign judged on distinct binding families would be a different
  experiment and might rank the players differently on pools of EGFR's shape.
* **Sequence similarity only.** Families come from 3-mer sequence similarity, not structural
  comparison (TM-score, Foldseek), which is what a real campaign would likely add. The
  structure-comparison columns in Proteinbase could not be used: they cover only ~20% of
  designs and carry a missingness leak (section 2.4).
* **Relative bars.** Passing in section 3.2 means beating the strongest baseline *on that
  pool*: a high bar on Nipah-G (26.4), a low one on EGFR (10.4). It is not an absolute measure
  of usefulness, and the bar is known only in hindsight (section 2.5).
* **Possibly seen data.** Nipah and EGFR competition data is public, so models may have seen
  it in training. This affects every condition equally but weakens absolute claims.
* **Provider variability.** Runs depend on live APIs: transient server errors (retried, and
  the episode replayed), cost inflation from thinking modes, and no sampling seed. A rerun is
  not token-for-token reproducible; the recorded file *is* the result.
* **Little tuning.** We tried only two prompt and tool iterations, both on the development
  pool. This is not an exhaustive search of what prompting can do.
* **A known defect in the played results.** We played every episode in sections 3.1–3.5 before
  the orientation fix (section 3.6). The defect cost the agent rather than helped it, but the
  absolute numbers belong to a toolkit with that flaw.
* **Early stopping barely tested.** Almost every campaign spent its full budget. A
  stopping-advice tool exists, but these results say nothing for or against stopping early.

## 5. Conclusion

Purpose-built tools make the same model find more binders: on both pools the point estimate is
positive for both models, and three of the four intervals exclude zero. On EGFR the gain comes
almost entirely from one tool, the similarity lookup that lets the agent follow up on its own
hits — and a pre-registered ablation on Nipah-G, where authorship cannot explain it, shows that
tool paying about five hits in sixty purchases there too.

Compared with the baseline policies, the tool-using agent is not better everywhere. A fixed
rule wins when its assumptions hold: on Nipah-G the first sort column finds hits at once, the
learner can fit a model from round 2, and a well-tuned learner beats the agent. The agent wins when
a fixed rule gets stuck: on EGFR the first column finds almost nothing, the baseline learners
cannot fit a model until the last round, and the agent, which can change strategy after every
result, escapes that cold start and can fit a round earlier.

On this evidence, agentic lab platforms might benefit from giving agents analysis tools of
this kind (free inspection of the pool, follow-up on discoveries, a model fitted on what has
been learned) rather than relying on more raw model capability. The next checks are the
once-only test on the Anthropic dataset and, eventually, a live campaign.

---

## For developers

```text
src/replaylab/
  domain/     core types and feature specs
  config.py   YAML configuration loading and validation
  datasets/   Proteinbase tables, 3-mer similarity and clustering, folds, aliases,
              pool loader
  store/      SQLite store: append-only campaign history
  backends/   replay backend: serves recorded outcomes to the agent
  engine/     campaign rules (clock, submit/settle/wait), scoring
  advisor/    preprocessing, logistic model, similarity index, rate and stop rules
  policies/   batch selection: random, fixed metric, learner, learner_sim (+stop)
  eval/       evaluation harness: agent loop, runner, spend guard, comparisons
  interfaces/ the 15 tools, MCP server and CLI (the agent-facing layer)
  spec_pin.py OpenAPI spec pin
data/         scripts that build the processed tables, clusters and split
scripts/      ladder, figures, exports and other convenience scripts
tests/        one test file per module, plus the import-boundary guard
```

```bash
uv sync                                    # project + dev group (pytest, ruff); without uv: pip install -e . pytest ruff
make test                                  # CI set: no data, network or key needed
make lint                                  # ruff check + format check
make test-all                              # everything, including real-data checks

# Download the Proteinbase CSV to data/raw/proteinbase.csv (git-ignored), then:
python data/build_tables.py --csv data/raw/proteinbase.csv --check   # expect 15 x OK
python data/build_clusters.py --check                                # cluster summary check
python data/build_folds.py                                           # writes the frozen split
pytest -m data -q                                                    # real-data checks

# Evaluation harness
python scripts/run_ladder.py                                         # reports/ladder.{csv,md}
uv sync --extra eval                                                 # adds the LLM client extras
replaylab-eval conditions                                            # the tool set of every condition
replaylab-eval pilot --run-dir runs/x --model deepseek --games N-dev \
    --conditions A,B1 --trials 20 --max-usd 1                        # first episodes + cost projection
replaylab-eval run --run-dir runs/x --model deepseek --games N-dev \
    --conditions A,B1 --trials 20 --max-usd 1                        # plays only what is missing
replaylab-eval scripted --run-dir runs/x --games N-dev \
    --policies random,learner,learner_sim,learner_sim_purchased --trials 20
replaylab-eval compare --records runs/x/episodes.jsonl --game N-dev \
    --a B1 --b scripted:learner_sim --ci 0.9                         # paired bootstrap over seeds
replaylab-eval view --run-dir runs/x                                 # one HTML page that replays the run
```

`build_folds.py` refuses to overwrite a frozen split (exit code 3); pass `--force` only if you
really mean to replace it.

The figures in this README are generated, not pasted. `scripts/export_study.py` extracts the
per-episode scores from the run records (scores only: no transcripts, identifiers or
sequences), and `scripts/make_figures.py` redraws the SVGs from that CSV with the project's own
paired bootstrap, so a figure cannot drift from what `replaylab-eval compare` reports.

```bash
python scripts/make_figures.py
```

### Testing on a new dataset

A new pool needs two tables, two config entries and nothing else. The Anthropic pools were
added this way (`datasets/anthropic.py` with `scripts/build_anthropic.py` is a worked example).

1. **Write the tables** to a directory of your own, for example `data/processed/mypool/`:
   * `designs.parquet`: one row per design, with `design_id`, `sequence`, `design_class` (may
     be empty) and one column per score you want to expose;
   * `outcomes.parquet`: one row per (design, target), with `design_id`, `target`, `hit`
     (bool), `strength` and `kd_molar` (may be empty), `n_records`, `n_true` and `conflict`.
2. **Declare the features** in `config/features.yaml` under your target id: the visible
   columns, `higher_is_better: false` on every column where lower is better, and a
   `first_column` for the baseline learners' fallback. Choose these before looking at
   outcomes.
3. **Declare a game** in `config/games.yaml`: target, `split: all` (or `dev` with a folds
   file), rounds, batch size, budget and, optionally, `delay_rounds`.
4. **Check it with the baselines first**, which costs nothing:

```bash
uv run replaylab run-scripted --policy learner_sim --game MY-GAME --seed 0 --tables data/processed/mypool
```

```bash
uv run python scripts/run_ladder.py --game MY-GAME --seeds 100 --tables-dir data/processed/mypool --out-csv runs/my-ladder.csv --out-md runs/my-ladder.md
```

   (Without `--out-csv` and `--out-md` the ladder overwrites the committed `reports/ladder.*`.)

5. **Play it with an agent.** Fill in `config/models.yaml` and `config/prices.yaml` from the
   `.example` files (check every price on the provider's own pages), then run `pilot` and
   `run` with `--games MY-GAME --tables data/processed/mypool`, as in the commands above, and
   `compare` B1 against A. Optionally build sequence families with `data/build_clusters.py`
   for the distinct-family score, and folds with `data/build_folds.py --target <id>` for a
   held-out split.

If you mean the new pool as a confirmatory test, write the analysis plan down before the first
episode and mark the game `final_test: true`, which puts it behind the run-once lock
(`eval/final_run.py`).

### Adding a tool or a policy

* **A tool:** a method on `Toolbox` in `interfaces/tools.py` with a response model in
  `interfaces/schemas.py`, a closure in `interfaces/server.py`, and its name in one of the tool
  groups at the top of `tools.py`. Expose it through a new condition in
  `config/conditions.yaml` (conditions are tool allow-lists), then refresh and review the tool
  snapshot with `pytest tests/test_mcp_server.py --update-tool-snapshot`. New tools must keep
  the leak rules that `tests/test_tools_leaks.py` checks.
* **A baseline policy:** a class with `choose(state, k, rng) -> list[str]` in `policies/`,
  registered in `interfaces/policy_factory.py` (`make_policy`, `POLICY_NAMES`), and in
  `eval/ladder.py` if the ladder should run it.

[`docs/code-tour.md`](docs/code-tour.md) shows where each of these sits in the call path.

### Further reading

| document | what it holds |
| --- | --- |
| [`docs/code-tour.md`](docs/code-tour.md) | entry points, call paths, and the order to read the modules in |
| [`docs/user_guide/`](docs/user_guide/) | architecture, vocabulary and the evaluation workflow |
| [`docs/walkthrough.md`](docs/walkthrough.md) | a hands-on session on the real tables |
| [`docs/design-notes.md`](docs/design-notes.md) | every number behind every decision, plus each experiment's registration and results |
| [`docs/caveats.md`](docs/caveats.md) | the conditions these results hold under; read before quoting them |
| [`docs/prereg-anthropic.md`](docs/prereg-anthropic.md) | the final test, fixed before it is played |
| [`docs/prereg-orientation.md`](docs/prereg-orientation.md) | the toolkit-defect experiment and its null result |
| [`docs/prereg-authorship.md`](docs/prereg-authorship.md) | whether the similarity tool rides authorship, fixed before it was played |
| [`reports/ladder.md`](reports/ladder.md) | every baseline policy, 100 seeds per game |
| `reports/agent_study.csv` | the per-episode scores behind every figure |

### Data definitions

* **hit**: any binding record for the (design, target) pair is true.
* **conflict**: the binding records for the pair disagree.
* **strength**: the strongest label among the strength records (Strong, Medium, Weak).
* **kd_molar**: the smallest KD, converted to molar. A KD with no target is attached to the
  design's only binding target, or dropped if that is ambiguous.
* Metadata (`meta_name`, `meta_author`, `meta_method`) is stored apart from the features and
  is never visible to an agent.

**Output** (`data/processed/`, git-ignored): `designs.parquet`, `outcomes.parquet`,
`targets.parquet`, `evaluations_long.parquet`, `build_report.json`.
