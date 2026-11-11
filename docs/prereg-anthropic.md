# Pre-registration: the Anthropic binder release as the final test

Written 2026-10-30, **before any episode of `AB-nipah`, `AB-egfr`, `AB-bbf14` or `AB-bhrf1`
has been played** and before any hit rate of any of these pools was computed. It fixes the pools, the
features, the label and the game shape (the project's planned final test);
`docs/design-notes.md` section 9 holds the decision rule this will be judged by, with the
bar and the margin of section 5 below, and `docs/caveats.md` holds the conditions under which
the result should be read.

Nothing here was chosen by looking at outcomes. Where a choice could have been made by
looking, this file says so and says what was used instead.

## 1. The data

The release is the public "Protein binder design data release (v1.0)": 1,440 de novo
miniprotein binders against 16 targets, with binding calls from two independent vendors
(Adaptyv Bio and Twist Bioscience), a standardized re-fit, and co-folded predictions from
ten structure predictors. Licence CC BY 4.0, cited as in its `CITATION.cff`.

Two files are used, pinned by the sha256 the publisher's own `data/MANIFEST.sha256` gives:

| file | sha256 |
| --- | --- |
| `tables/design_summary.parquet` | `3e796869b8dc0c90d7ab0e12daf25308c0cf272d04c6a5e27281f903ecd18744` |
| `tables/insilico/cofold_predictions.parquet` | `ba48010610daa9b9d184aa1ddb419beebdf728ccf745f5a8f3b48e4041e1a87d` |

All 22 files downloaded were verified against that manifest and matched.
`datasets/anthropic.py` re-checks both hashes on every build and refuses to build otherwise.

**No sequence of this release appears in Proteinbase** (checked by exact sequence match:
0 of 1,440), so the pools are genuinely unseen by anything tuned on N-dev or E.

## 2. The pools: one game per target, and which four

A target's designs are its own pool, which is what a real campaign selects from. A pooled
game across targets was considered and rejected: it would have required rescaling the
features to make the targets comparable (section 3), and it would have made the target name
meaningless, which `describe_campaign` reports and which every N-dev and E episode carried
(`docs/caveats.md` entry 2). Pooling the *results* of several single-target games gives the
same statistical power without either problem, so that is what section 5 does.

The 1,320 designs that carry a binder call split into 15 targets. The games were chosen by
this rule, applied to documented columns and nothing else:

1. **90 designs with a binder call** -- the modal pool size, so the four games are one
   shape. Drops 15-PGDH (30), Latent GDF-8 (60), TNFa (150) and mature GDF-8 (0 -- the
   release says its wet-lab results were inconclusive and are not included).
2. **`1to1` stoichiometry**, as Nipah-G and EGFR are, so the games are comparable to each
   other. Drops VEGF-A's dimer and the Cas9 ribonucleoprotein.
3. **No cofactor in the co-folds** (`INSILICO.md` section 3). Drops RBX1's three zinc ions.
4. **Nipah-G and EGFR**, because Proteinbase has the same two targets: the same target under
   a different design campaign is the most direct test of transfer available.
5. **Two targets Proteinbase does not have**, so that a result on the first two cannot be
   explained by target familiarity alone, and so that the unfamiliar half is not a single
   game. Seven candidates pass filters 1 to 3 (BBF-14, BHRF1, IL-7Ra, MBP, PD-L1, TREM2,
   TrkA) and the **first two alphabetically** are taken: **BBF-14** (a 110-residue
   construct) and **BHRF1** (157 residues). The tie-break is arbitrary by design, which is
   the point: it cannot correlate with anything in the data.

So **four games**, all 90 designs and `1to1`: `AB-nipah`, `AB-egfr`, `AB-bbf14`, `AB-bhrf1`.
TNFa is the only target with more designs (150) and was **not** added: it folds as a trimer
(`1to3`), so taking it would have meant amending filter 2 after the filters were written,
which is the move a pre-registration exists to prevent. It remains available as a declared
amendment if a larger pool is wanted, decided before any play.

Cas9 is excluded by filter 2 rather than by a judgment about difficulty. `DATA_NOTES.md`
note 20 separately records that its ipSAE is exactly zero for most of its designs.

Target ids are the release's codes under an `ab-` prefix: `ab-nipah-g`, `ab-egfr`,
`ab-bbf-14`. The prefix keeps them distinct from the Proteinbase targets of the same name,
whose features differ; the biological target stays named on purpose (section 4).

## 3. The features

**The crosswalk** is the set of Proteinbase feature columns that also exist in this release,
under their Proteinbase names and their Proteinbase orientation (all higher-is-better):

| replaylab name | release source |
| --- | --- |
| `boltz2_min_ipsae` | Boltz-2 `ipsae_min`, seed-best |
| `boltz2_iptm` | Boltz-2 `iptm_pae`, same seed |
| `boltz2_plddt` | Boltz-2 `plddt_binder`, same seed |
| `esmfold_plddt` | ESMFold2 (full) `plddt_binder`, same seed |

`proteinmpnn_score` and `shape_complimentarity_boltz2_binder_ss` have **no counterpart** in
the release and are left out. Nine further predictors are available and are also left out:
the crosswalk is the subset that exists in both pools, which is what direction A asked for,
and four features keep these games close to E's two and N-dev's seven rather than handing
them twenty columns no other game had.

**Values are as published.** `docs/INSILICO.md` section 4 warns that these metrics are "not
comparable across targets, stoichiometries or predictors" -- which is a reason not to pool
targets, and, once the games are per-target, no reason to transform anything. Nothing is
rescaled at load time (D7); the advisor standardizes within the pool, as it does for every
other game.

**Seed-best rows.** `design_summary.ipsae_min_<m>` is the highest `ipsae_min` over the five
seeds at the design-complex stoichiometry on the design target, ties to the lowest seed
label (`INSILICO.md` section 4). ipTM and pLDDT of that same seed are taken from
`cofold_predictions` by reproducing that rule, excluding the `2to2` / `3to3` folds and the
counter-screen and assay-form rows. The rule is **verified, not trusted**: the reproduced
`ipsae_min` matches `design_summary` exactly (maximum absolute difference 0.0 over all 1,440
designs, both predictors), and `build_designs` raises if it ever stops matching.

**`first_column`** is `boltz2_min_ipsae`: N-dev's pre-registered first column is the Boltz-2
ipSAE, which exists here only as the min-direction variant. Chosen by that analogy, not by
which column sorts best.

## 4. The label, and what the player sees

**The label** is `binder_final`: the release's own binder call on the human target, from both
vendors plus a standardized re-fit, each call checked against the raw sensorgrams of both
vendors (`WETLAB.md` section 6). `kd_nM_final` becomes `kd_molar` by multiplying by 1e-9,
exactly, never rounded (rules of engagement 1). `strength` is empty: the release has no
categorical comparable to Proteinbase's, and inventing one would be a choice about which
hits count as good. `vendor_agreement` becomes the record counts and the `conflict` flag by
the documented mapping of its seven classes. This is a **better** label than Proteinbase's
"any binding record is True", which is worth noting when the two results are compared.

**The target is named to the player**, as it was in every N-dev and E episode (verified in
the recorded transcripts: `egfr` appears in the E episodes). Withholding it here would make
these games differ from the played ones in a second way on top of the new pool, and would
confound the comparison; it would also be a less realistic campaign, since a real one knows
its target. The cost is that these targets are public and a model may recall published
work. That cost is already carried by every result in this project, it falls identically on
every condition, and the designs themselves are unpublished. It is recorded as
`docs/caveats.md` entry 2, and it bounds what the write-up may claim: a paired difference
between conditions stays interpretable, an absolute hit rate is not evidence of de novo
reasoning.

`design_class` carries `generator` (the design method: RFdiffusion, PXDesign, BoltzGen, ...),
which is what Proteinbase's `design_class` holds. Sequences are never shown, so a player
that recognised the release could still not identify an individual design.

## 5. The game shape, and the order it was decided in

The shape rule was fixed in conversation **before any hit rate or hit count of these pools
had been computed**, and is recorded here as it was stated:

> budget at most 20% of the pool, 4 rounds

That rule was written for a 1,320-design pooled game. Moving to 90-design per-target games
changes what it implies, and the implication was re-derived from the fit gate rather than
from any outcome:

**4 rounds of 8, budget 32 of 90 (36%).** Four rounds match N-dev and E, so the horizon is
not a new variable. The batch size is set by neutrality, not by share: the scripted fit gate
needs 2 hits and 8 purchases, and at an assumed rate of 20% a batch of 5 makes the gate
likely only after round 3 of 4, while a batch of 8 opens it in round 2, as on E. A game
whose gate stays shut favours the agent systematically -- that is the mechanism behind E's
result -- so a batch of 8 is the choice that does **not** build in the effect we are trying
to measure. The 36% share is the price, and it is above the 20% the original rule named;
that deviation is recorded here rather than quietly taken.

**Pooling results, not designs.** Each game is compared on its own (paired by seed, as
always), and the four games' paired differences are also pooled, giving 4 x *trials* paired
episodes. The pooled difference is the headline (section 6).

### The bar: `scripted:learner_sim`

Named by the author, 2026-10-30, before any episode. The agent is judged against
**`scripted:learner_sim`**, played on the same seeds, as condition `scripted:learner_sim`.

Why that policy and not the stronger-looking one:

* **The purchased-row variant is not a consistent baseline.** From `reports/ladder.md`:
  `learner_sim_purchased` scores 26.43 against `learner_sim`'s 14.52 on N-dev (+11.9) and
  4.45 against 10.39 on E (-5.9). A bar that swings by 12 hits in one direction and 6 in the
  other, depending on the pool, is not measuring the same thing twice.
* **The project already chose the pool-wide scaler.** The two variants differ in *nothing but
  the scaler* -- pool-wide standardization against standardization on the delivered purchases,
  recomputed each round (`policies/purchased.py`). Decision D-2026-10-13 made pool-wide the
  library default and keeps the purchased variant as a legacy comparison only. Using it as
  the bar would contradict a decision already frozen.
* **Naming one policy in advance avoids a multiplicity bias.** "Beat the best of the ladder"
  sounds stronger but takes a maximum over about 11 noisy policy means, and the maximum of
  noisy estimates is biased upward: with 20 seeds and per-policy sd of 1 to 3 hits, that is
  roughly +0.3 to +0.9 hits of inflation when the top policies are close. Against a margin of
  1.07 hits that bias could consume most of the margin, so the bar would be unfairly high for
  a reason that has nothing to do with the agent.

**The residual risk, stated plainly.** `learner_sim` is not guaranteed to be the best
scripted policy on these pools -- on N-dev it was beaten by plain `learner` (18.18) as well as
by the purchased variant. So clearing this bar is **not** a claim of beating every scripted
player. To keep that honest, the full ladder is played on the same seeds and reported
descriptively beside the verdict, exactly as section 15 did on E; if any policy beats the
agent, the write-up says so in the same table as the verdict.

### The margin, rescaled

Section 9's margin is **2 hits out of 60 purchases**. Carried over unchanged to a
32-purchase game it would be a far stricter test: the same absolute margin against a
random-to-oracle gap of roughly 12 to 14 hits rather than roughly 55. The author has agreed to
rescale it, and the rescaling registered here is **by purchase rate**, which keeps N-dev's
practical standard identical:

> 2 hits of 60 purchases = **3.33 percentage points** of hit rate within the purchased
> batch. On 32 purchases that is **1.07 hits**, which is the margin these four games use.

The alternative scaling -- by the random-to-oracle gap, which is what was first floated in
conversation -- was computed and **rejected**: 2/55 of a gap of 13 is 0.47 hits, below one
hit, so it would make the margin unreachable as a practical statement (hits are integers)
and amount to "beat the bar at all". Recording both numbers so the choice is visible.

A margin of 1.07 hits is a statement about effect **size**. It is not made easier by playing
more trials, and that distinction is the answer to a question raised in conversation:

* **More trials buy precision, not a lower bar.** They narrow the interval, so a true
  difference above the margin becomes easier to *establish*; a true difference below it
  becomes more clearly below it. That is legitimate and not confounding, on one condition:
  the trial count is fixed in advance, or extended by a rule declared in advance. Looking
  first and then adding trials because the result was nearly decisive is optional stopping,
  and it inflates false positives.
* **What the numbers say.** E's paired interval implies a per-episode paired sd of about 9
  hits on 60 purchases, so about 5 hits on 32. At 20 trials one game resolves to a 90%
  half-width of about 1.8 hits; the four games pooled (80 paired episodes) resolve to about
  0.9 hits, which is below the 1.07-hit margin. So **20 trials per game, pooled over the
  four games, is enough to clear the margin if the effect is real**, while a single game at
  20 trials would need an effect of about 2.9 hits to clear it on its own.

`docs/caveats.md` entry 3 says what the write-up must report alongside any margin.

### Disclosure: what was seen, and when

While checking feature coverage, the counts of the `vendor_agreement` classes were printed
for the whole release. Those counts bound the pooled binder count across all targets, so the
approximate overall base rate **has been seen** by the AI coding assistant before this file was
written. It was not sought, and it is disclosed here rather than left implicit. No
per-target rate has been computed, and none of the three games' rates is known.

What this does and does not affect:

* **The shape rule was already fixed** and stated before those counts appeared, and the batch
  size follows from the gate arithmetic above, which holds at any rate in the plausible
  range.
* **The features, the label, the pooling and the target choice** follow from column
  availability and the release's documentation, all of which was read before the counts
  appeared; filter 5's tie-break is alphabetical precisely so that no judgment enters it.
* **What is now unavailable** is the claim "nobody involved had any idea of the base rate".
  Anyone reading the result should treat the shape as pre-registered but not rate-blind.
* **The remedy** for any later choice: no further parameter of these games may be set by the
  AI coding assistant without being written here first. Section 6 lists what is still open,
  and those belong to the author.

## 6. The run, fixed (the author's decision, 2026-10-30, before any episode)

Everything below is frozen. Nothing here may be changed once the first episode of a game is
played: the lock file records the model, the trial block **and the condition set**, and an
otherwise-identical rerun with a different condition set is refused as a second draw
(section 8). So a condition cannot be added to a game later -- it would need its own game or
its own fresh block.

| | decision |
| --- | --- |
| games | all four, played together: `AB-nipah`, `AB-egfr`, `AB-bbf14`, `AB-bhrf1` |
| conditions | **A, B1, B2** |
| models | `deepseek` and `mimo`, each in its own run directory |
| trials | 20 per game, per condition, per model: block **0-19** (these games have never been played) |
| headline | the **pooled** paired difference over the four games |
| caps | 6 USD for the deepseek run directory, 4 USD for mimo |

**Why pooled, and not per game.** Section 5's arithmetic: at 20 trials one game resolves to a
90% half-width of about 1.8 hits, so a per-game verdict would need a true effect of ~2.9 hits
to clear the 1.07-hit margin -- four likely-inconclusive answers. Pooled over four games (80
pairs per model) the half-width is about 0.9, so an effect of ~2.0 hits clears. The per-game
numbers are still reported beside the pooled one, descriptively, with no threshold attached.

**Why A, B1, B2 and not C.** A and B1 repeat section 16's split between the model's own
judgment and the tools. B2 (B1 without `similar_designs`) is the one that carries new
information: on E the advisor's entire measurable contribution was that single tool
(B1-B2 +7.15 ds / +3.85 mimo, while B2-A straddled zero on both models), and only a fresh
pool can say whether that is a property of E or of the method. C is dropped because it has
never separated from B1 -- +0.1 on N-dev (section 13) and +0.1 on E (section 15).

**Why 20 trials.** It is what the pooled arithmetic above assumes. Fewer loses the
resolution; more buys little against a 1.07-hit margin. **No extension is registered**: if the
pooled interval straddles the margin the result is reported as inconclusive, because an
extension decided after seeing the numbers is optional stopping.

**Cost.** 240 episodes per model. At the per-episode rates measured on E (deepseek 0.0172,
mimo 0.0109 USD) that projects to 4.13 and 2.62 USD, and the caps above leave headroom. Those
rates are an **upper bound**: these pools hold 90 designs against E's 1,340, so tool results
and transcripts are shorter. The caps are hard limits, checked before every model call.

**Registered configuration hashes**, from a dry run of the registered command (the game does
not enter this hash, so one value per model covers all four games). A run whose records carry
anything else fails the gate:

| model | `config_hash` |
| --- | --- |
| `deepseek` | `0fecb4a7e74a` |
| `mimo` | `c2c8dac7e136` |

Verified at the same time: the command is refused without `--final-run`; with it, a dry run
plans 60 episodes per game (3 conditions x 20 trials) with A at 8 tools, B1 at 14 and B2 at
13; and a dry run writes **no** lock, so planning does not consume the test.

**The scripted ladder** is played on the same seeds, in the same run directories, at the same
time -- it is free, and these pools' ladder numbers cannot be computed any earlier than the
final run itself (section 2).

### What this run can and cannot say

Four games are **four targets from one design campaign, not four independent replications**.
Pooling them measures the method across targets within this release; it does not measure it
across campaigns, and the write-up must not read it as the latter. The targets also share a
single generator mix, a single scoring pipeline and a single pair of vendors.

### Amendment, 2026-11-01: the primary outcome is the tool contrast

Added **before any episode of any `AB-` game was played**, at commit `2d568d9` with a clean
tree. Checked at the time of writing: no `runs/ab-*` directory exists, no lock file exists for
any of the four games, and no record in any run directory names an `AB-` game. Nothing about
the run itself changes -- same four games, same conditions A/B1/B2, same two models, same
block 0-19, same caps, same registered `config_hash` values. What changes is the **analysis
plan**, which does not enter the config hash and was not fixed by section 6. The "headline"
row of section 6's table is to be read through this amendment: the pooled paired difference
is still the headline quantity, and the primary comparison it is taken over is now `B1 - A`.

**Why.** The author's argument, 2026-11-01: a bar set at "the best scripted policy" is
unfair to the agent, because in a real campaign nobody knows in advance which policy is
best, so the agent would be measured against an oracle that has already seen the results.
The more meaningful question is whether the agent *with* advisor tools beats the agent
*without* them, which is also the question the study set out to answer.

Section 5 already accepted half of this when it refused a maximum over the ladder and named
one policy in advance. The amendment accepts the rest. Two facts support it:

* **"Best scripted policy" is not a target anyone can aim at.** The ladder reorders between
  pools: `learner_sim_purchased` scores 26.43 on N-dev and 4.45 on E, `learner` 18.18 and
  7.44 (`reports/ladder.md`). A practitioner must commit to a policy before seeing any
  outcome, so a bar defined as "whichever policy turned out best on this pool" cannot be met
  by anything a practitioner could have done either.
* **The tool contrast needs no oracle.** B1 - A is paired by seed within one model and one
  game, and both arms are the same player with a different tool list. Nothing in it depends
  on knowing which scripted policy is strongest.

**The hierarchy, in order.** All comparisons are paired by seed and use
`replaylab.eval.stats.paired_bootstrap` at 90%, 10,000 resamples, resampling seed 0.

1. **Primary -- do the advisor tools help?** `B1 - A`, pooled over the four games, reported
   per model (80 paired episodes each). **Decision rule: the lower bound of the 90% interval
   is above zero.** No effect-size margin (see below).
2. **Secondary -- how does the agent compare to scripted play?** `B1 - scripted:learner_sim`,
   pooled over the four games, against the **1.07-hit margin** of section 5, unchanged.
3. **Descriptive, no threshold attached.** Per-game rows for both of the above; `B1 - B2`
   (what `similar_designs` contributes); `A - scripted:learner_sim`; and the full scripted
   ladder on the same seeds, so that any policy beating the agent is visible in the same
   table as the verdict, exactly as section 5 requires.

**The hierarchy is a hierarchy, not a maximum.** A primary interval that includes zero is
**not** rescued by the secondary clearing its margin, and the write-up may not promote the
secondary to the headline after the fact. Both are reported whatever their signs. This is the
same refusal as section 5's refusal to take a maximum over policies, applied to outcomes
instead of policies.

**Why no margin on the primary.** The 1.07-hit margin was derived by rescaling N-dev's
2-of-60 standard by purchase rate, and its purpose is to stop a trivially small win over a
*competitor* being reported as beating it. The primary is an ablation of our own toolkit, not
a contest: the quantity of interest is the sign and the width of the interval, and "the tools
help" has no natural effect size attached to it. The alternative -- reusing 1.07 for
consistency -- was considered and rejected on that ground, and is recorded here so the choice
is visible rather than silent.

**Resolution, from the played pools.** The per-episode paired sd of `B1 - A` measured on
N-dev and E (60 purchases) is 15.30 and 8.65 hits on E (deepseek, mimo) and 9.51 and 11.78 on
N-dev. Scaled to 32 purchases by the same purchase-rate factor section 5 uses (32/60) that is
roughly 5 to 8 hits, so 80 pooled pairs give a 90% half-width of about **1.1 to 1.5 hits**.
The effect measured on the played pools, scaled the same way, is **+2.5 to +3.0 hits**
(pooled `B1 - A` of +4.62 on N-dev and +5.60 on E, both over 40 pairs). So the primary
resolves the effect if it transfers to a fresh campaign, and reports it as inconclusive if it
shrinks by more than about half.

**Pairing does not reduce variance for this contrast, and that is the conservative
direction.** Measured on the played blocks, the two agent arms' per-seed hit counts are
*negatively* correlated in three of the four cells (r = -0.42 on E deepseek, -0.16 and -0.22
on N-dev; +0.01 on E mimo, and -0.45 for `B1 - B2` on E deepseek), so differencing by seed
gives an interval **wider** than treating the arms as independent would -- by up to 19% in
the cells with the strongest negative correlation, and by nothing in the neutral one.
Pairing is kept regardless, because it is what the seed design licenses and because the
inflation is in the direction that cannot manufacture a result. See `docs/caveats.md`
entry 6.

**No extension.** Unchanged from section 6: if the primary interval straddles zero the result
is reported as inconclusive. Adding trials after seeing it would be optional stopping.

**Disclosure.** Every number quoted in this amendment comes from the already-played N-dev and
E blocks, which are development and confirmatory data for those two pools. No quantity of any
`AB-` pool was computed, looked at or estimated in writing it, beyond what section 5's
disclosure already records.

## 7. The configuration these games run on

The orientation defect of `docs/caveats.md` entry 4 was **fixed before any of these games was
played**, and this is the one respect in which their configuration differs from the one that
produced the N-dev and E results:

* `describe_campaign` and `column_summary` now report each column's `higher_is_better`
  (`None` for `length` and `design_class`, which have no pre-registered direction);
* `fit_and_rank` and `fit_custom_model` report each weight in the **raw-value direction**, so
  a positive weight means a higher raw value goes with binding. Magnitudes, the ranking and
  which weights l1 sets to zero are unchanged -- only the reported sign is re-expressed;
* `query_pool`'s description states the hazard without naming a column, so it stays true for
  any pool; the system prompt says the scores do not all point the same way, without naming a
  tool (it has to hold for every condition).

Nothing about D7 changed: the values the player sees are still raw and still never
reoriented. The fix describes the orientation, it does not apply it.

This changes the prompt and tool-description hashes, so these four games do not share a
`config_hash` with the played N-dev and E blocks. That is accepted rather than avoided: the
Anthropic games already differ in pool, features, target set, shape and margin, so the
comparison was never a paired one, and running the final test through a surface known to
mislead would have been the worse trade. The played results stand as results about the
configuration they were played on; **nothing is replayed** (`docs/caveats.md` entry 4 says why
the direction of the defect means neither verdict is at risk).

## 8. The guard

`AB-nipah`, `AB-egfr`, `AB-bbf14` and `AB-bhrf1` are each marked `final_test: true` in
`config/games.yaml`, so:

* reading their outcomes, building a runtime for them (`serve`, `campaign`), the ladder, the
  replay page and `smoke` all refuse them outright;
* only `replaylab-eval scripted|pilot|run --final-run` play them, one game at a time, from a
  committed `src` / `prompts` / `config`, under the lock `reports/final-test/<game>.json`;
* each lock pins the frozen configuration hash and the commit, and records one leg per model
  (or `scripted`); an identical rerun resumes and plays only the missing episodes, and any
  other setting is refused.

Each game has its own lock, so playing one game does not consume the others.

## 9. The rehearsal

`AB-rehearsal` is the same shape on a same-shape copy built by
`scripts/build_anthropic.py --rehearsal`, which takes one game-sized slice of the real
feature matrix and

* shuffles the feature **column names**, so no column means what it is named,
* replaces every label with a fresh draw at a rate of **0.15**, seed **20261030**, and
* relabels the target `ab-rehearsal`,

all declared here in advance. **No number measured on the rehearsal copy says anything about
the real pools**, by construction, so nothing measured there can tune the real run. The
separate target id also means the rehearsal game cannot be pointed at the real tables: that
target is not in them, and `load_pool` refuses.

What the rehearsal has already exercised, before any real episode:

* the full scripted ladder on a 90-design pool, 32 purchases, every policy, fit gate
  included;
* the agent path end to end through the MCP server with a scripted fake model: episodes
  finished, 0 invalid tool arguments;
* the guard itself, both halves -- a final-test game refuses its own outcomes, and the
  rehearsal game cannot reach the real pools.

## 10. The commands, as registered

Played once per model. The condition set and trial block below are what the lock records; a
rerun must repeat them exactly, and does nothing but fill in episodes that are missing.

The scripted ladder first (free, no key, and the only time these numbers can be computed):

```bash
for g in AB-nipah AB-egfr AB-bbf14 AB-bhrf1; do
  replaylab-eval scripted --run-dir runs/ab-ds --games $g \
      --policies random,learner,learner_sim,learner_purchased,learner_sim_purchased \
      --trials 20 --final-run
done
```

Then the agent, one game at a time (each game has its own lock):

```bash
for g in AB-nipah AB-egfr AB-bbf14 AB-bhrf1; do
  replaylab-eval run --run-dir runs/ab-ds --model deepseek --games $g \
      --conditions A,B1,B2 --trials 20 --max-usd 6 --final-run
done
```

Then both again with `--run-dir runs/ab-mimo --model mimo --max-usd 4`. Finally, per model:

```bash
replaylab-eval compare --records runs/ab-ds/episodes.jsonl --game AB-nipah \
    --a B1 --b scripted:learner_sim --ci 0.9 --resamples 10000 --seed 0 --trial-range 0-19
```

The pooled headline is the same paired bootstrap over all four games at once, pairing on
`(game, trial)` rather than on `trial`.

**Gates before any hit count is read** (section 5 of `docs/prereg-orientation.md` is the
worked example): every episode `finished`; one `config_hash` per model, equal to the frozen
hash the lock recorded; the scripted rows present on every game; and no uncommitted change in
`src`, `prompts` or `config` in the tree that played them.

### Amendment, 2026-11-11: the pilot step, and the pooled command

Added **before any episode of any `AB-` game was played**: no `runs/ab-*` directory, no lock
file and no record naming an `AB-` game exists. Nothing about the run changes -- same games,
conditions, models, block, caps and registered hashes (a dry run of the registered command
still reports `0fecb4a7e74a` and `c2c8dac7e136`). Two gaps in section 10 are closed.

**The pilot.** As written, section 10 goes straight to `run`, which refuses to start without a
pilot (the money gate: a run starts only if the pilot's cost, projected over the matrix, fits
the cap). The registered order for each game and model is therefore:

```bash
replaylab-eval pilot --run-dir runs/ab-ds --model deepseek --games $g \
    --conditions A,B1,B2 --trials 20 --max-usd 6 --n 12 --final-run
replaylab-eval run   --run-dir runs/ab-ds --model deepseek --games $g \
    --conditions A,B1,B2 --trials 20 --max-usd 6 --final-run
```

The pilot plays the first 12 episodes of the matrix (trials 0-3, every condition) and is
**the start of the final run, not a separate draw**: it takes the game's lock, and its
episodes are kept; `run` then plays the rest. The cap is the registered one in both commands
(`--max-usd 4` for mimo), because both write to the same spend ledger. The pilot is not
skipped with `--skip-pilot` (which exists for development), because the money guard is not to be
bypassed.

**The pooled headline command.** Section 10 described the pooled comparison but no command
computed it: `compare` took one game. It now takes `--game` more than once and pools every
pair into one bootstrap, paired on `(game, trial)` (a game with no pair is an error, so one
cannot drop out unnoticed). The primary of the 2026-11-01 amendment, per model:

```bash
replaylab-eval compare --records runs/ab-ds/episodes.jsonl \
    --game AB-nipah --game AB-egfr --game AB-bbf14 --game AB-bhrf1 \
    --a B1 --b A --ci 0.9 --resamples 10000 --seed 0 --trial-range 0-19
```

and the secondary the same way with `--b scripted:learner_sim`. Per-game rows are the same
command with one `--game`.

**Disclosure.** The pooled command was written and tested on 2026-11-11 and rehearsed on
scripted records only: the five-policy ladder on `AB-rehearsal` (whose labels are random
draws) and on `N-dev`, trials 0-19, pooled over the two. The rehearsal checks the plumbing --
the pooled mean equals the average of the two per-game means, the ledgers add up -- and says
nothing about any `AB-` pool.

---

*Editorial note, 2026-11-10, on publication:* before the repository went public, its history
was rewritten to remove private details. In every version of this file the attribution
wording was neutralised ("the owner" became "the author", a quoted argument became a
paraphrase with its date); in this version "the assistant" also became "the AI coding
assistant" and "the plan the handoff calls direction A" became "the project's planned final
test". No registered content changed: the games, the bar, the margin, the conditions, the
trial block and the registered hashes are as written before any episode. The original
history is kept as a git bundle with sha256
`f365dc7831569148068c9fe4c4dfacf248f92ad9339608c97d5e444db81a39ab`, so every
pre-publication version can be produced and checked against it.
