# Pre-registration: the Anthropic binder release as the final test

Written 2026-10-30, **before any episode of `AB-nipah`, `AB-egfr`, `AB-bbf14` or `AB-bhrf1`
has been played** and before any hit rate of any of these pools was computed. It fixes the pools, the
features, the label and the game shape. It is the plan the handoff calls direction A;
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
amendment if the author wants a larger pool, decided before any play.

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
episodes. Which is reported as the headline is section 6's call.

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
approximate overall base rate **has been seen** by the assistant before this file was
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
  assistant without being written here first. Section 6 lists what is still open, and those
  belong to the author.

## 6. What is still open (the author's call, before the run)

The bar is fixed (section 5). What remains:

1. **How the four games combine.** Whether the headline is the pooled paired difference over
   the four games (which section 5 shows is the version with enough precision) or a per-game
   verdict, and whether all four are played at once or the two Proteinbase targets first.
2. **Which conditions and how many trials.** A and B1 at minimum, to repeat the section 16
   split between the model's own judgment and the tools; C and B2 if the budget allows. 20
   trials per game is what section 5's arithmetic supports. The trial block starts at 0 for
   these games: they have never been played. Whatever is chosen is fixed before the first
   episode, and any extension needs its rule written here first.
3. **The cost.** One pilot episode, then the usual projection and cap. These pools are
   90 designs rather than 1,320, so transcripts and costs should be well below E's.

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
