# Pre-registration: the Anthropic binder release as the final test

Written 2026-10-30, **before any episode of `AB-final` has been played** and before any hit
rate of the pool was computed. It fixes the pool, the features, the label and the game
shape. It is the plan the handoff calls direction A; `docs/design-notes.md` section 9 holds
the decision rule this will be judged by, and section 18 of that file will hold the result.

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
0 of 1,440), so the pool is genuinely unseen by anything tuned on N-dev or E.

## 2. The pool

One pooled game over the **15 targets that carry a binder call**: 1,320 designs. The 120
mature GDF-8 designs are dropped because `binder_final` is blank for every one of them (the
release states the wet-lab results were inconclusive and are not included). That is the only
exclusion, and it is forced by the absence of a label, not chosen.

Every design carries the target id `anthropic-binders-pooled`, so the existing engine plays
it with no change; the final run passes `--tables data/processed/anthropic`.

Pooling rather than one game per target was chosen because a per-target pool holds about 90
designs, where a 4-round game would buy most of the pool and the scripted fit gate (2 hits
and 8 purchases) would be reached, if at all, only at the very end. The cost of pooling is
addressed in section 3.

**Cas9 is kept.** `DATA_NOTES.md` note 20 records that its ipSAE is exactly zero for most of
its designs, because the selection-time folds were ribonucleoprotein co-folds scored across
the sgRNA. That makes its features weak, not absent, and dropping a target for weak features
would be a choice about which designs are easy. The per-target scaling of section 3 keeps a
constant column at zero rather than dividing by zero.

## 3. The features

`docs/INSILICO.md` section 4 states plainly that these metrics are "not comparable across
targets, stoichiometries or predictors". Pooling targets therefore requires a scaling step,
and the step is fixed here:

* **Per-target z-scores.** Each feature is centred and scaled within its own target
  (population sd) before the targets are pooled. This uses feature values only -- no
  outcome, no label, no purchase -- so it cannot leak. A target whose column is constant
  keeps the centred value, which is zero.
* Orientation is still never applied at load time (D7). All four features are
  higher-is-better in Proteinbase, so the z-score keeps their direction.

**The crosswalk** is the set of Proteinbase feature columns that also exist in this release,
under their Proteinbase names:

| replaylab name | release source |
| --- | --- |
| `boltz2_min_ipsae` | Boltz-2 `ipsae_min`, seed-best |
| `boltz2_iptm` | Boltz-2 `iptm_pae`, same seed |
| `boltz2_plddt` | Boltz-2 `plddt_binder`, same seed |
| `esmfold_plddt` | ESMFold2 (full) `plddt_binder`, same seed |

`proteinmpnn_score` and `shape_complimentarity_boltz2_binder_ss` have **no counterpart** in
the release and are left out. Nine further predictors are available and are also left out:
the crosswalk is the subset that exists in both pools, which is what direction A asked for,
and four features keep the pool close to E's two and N-dev's seven rather than handing this
game twenty columns no other game had.

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

**The target identity is not exposed to the player.** `design_class` carries `generator`
(the design method: RFdiffusion, PXDesign, BoltzGen, ...), which is what Proteinbase's
`design_class` holds. The targets are public and widely written about, so naming them could
let a model recall published results instead of predicting; and the z-scores already make the
features comparable without naming them. This withholding applies identically to every
player, agent and scripted alike. It is a limitation the write-up must state: a pooled game
whose player cannot see the grouping is not the same decision problem as a real campaign.

## 4. The label

`binder_final`: the release's own binder call on the human target, from both vendors plus a
standardized re-fit, each call checked against the raw sensorgrams of both vendors
(`WETLAB.md` section 6). It covers 1,320 of 1,440 designs.

`kd_nM_final` becomes `kd_molar` by multiplying by 1e-9, exactly, never rounded (rules of
engagement 1). `strength` is empty: the release has no categorical comparable to
Proteinbase's, and inventing one would be a choice about which hits count as good.

`vendor_agreement` becomes the record counts and the `conflict` flag, by the documented
mapping of its seven classes.

This is a **better** label than Proteinbase's "any binding record is True", which is worth
noting when the two results are compared.

## 5. The game shape, and the order it was decided in

The shape rule was fixed in conversation **before any hit rate or hit count of this pool had
been computed**, and is recorded here as it was stated:

> budget at most 20% of the pool, 4 rounds

With 1,320 designs that gives **4 rounds of 60, budget 240** (18.2% of the pool). Four rounds
match N-dev and E, so the horizon is not a new variable. The aim of the rule was neutrality:
a batch of 60 reaches the scripted fit gate (2 hits, 8 purchases) in round 2 at any base rate
above roughly 4%, so neither the cold start that favoured the agent on E nor the fully warm
ladder of N-dev is built in by the shape.

### Disclosure: what was seen, and when

While checking feature coverage, the counts of the `vendor_agreement` classes were printed.
Those counts bound the pooled binder count (both vendors binding: 253 of 1,320; one vendor
only: 136 more), so the approximate base rate of the pool **has been seen** by the assistant
before this file was written. It was not sought, and it is disclosed here rather than left
implicit.

What this does and does not affect:

* **The shape rule was already fixed** and stated before those counts appeared, so 4 x 60 was
  not chosen with the rate in hand. The rule as stated also makes the gate reachable at any
  rate above about 4%, so knowing the rate would not have changed it.
* **The features, the label, the pooling and the exclusions** follow from column availability
  and the release's documentation, all of which was read before the counts appeared.
* **What is now unavailable** is the claim "nobody involved had any idea of the base rate".
  Anyone reading the result should treat the shape as pre-registered but not rate-blind.
* **The remedy** for any later choice: no further parameter of this game may be set by the
  assistant without being written here first. Section 6 lists what is still open, and those
  belong to the author.

## 6. What is still open (the author's call, before the run)

1. **The bar and the decision rule.** Section 9's rule (90% paired interval, 2-hit margin)
   applies, but the bar has to be named: on N-dev it was `learner_sim_purchased`, on E it was
   `learner_sim`. The scripted ladder of this pool can only be computed **at the final run**,
   so the bar must be named by policy, not by number.
2. **Which conditions and how many trials.** A and B1 at minimum, to repeat the section 16
   split between the model's own judgment and the tools; C and B2 if the budget allows. The
   trial block starts at 0 for this game: it has never been played.
3. **The cost.** One pilot episode, then the usual projection and cap.

## 7. The guard

`AB-final` is marked `final_test: true` in `config/games.yaml`, so:

* reading its outcomes, building a runtime for it (`serve`, `campaign`), the ladder, the
  replay page and `smoke` all refuse it outright;
* only `replaylab-eval scripted|pilot|run --final-run` play it, on its own, from a committed
  `src` / `prompts` / `config`, under the lock `reports/final-test/AB-final.json`;
* the lock pins the frozen configuration hash and the commit, and records one leg per model
  (or `scripted`); an identical rerun resumes and plays only the missing episodes, and any
  other setting is refused.

## 8. The rehearsal

`AB-rehearsal` is the same game shape on a same-shape copy built by
`scripts/build_anthropic.py --rehearsal`, which keeps the real feature matrix but

* shuffles the feature **column names**, so no column means what it is named, and
* replaces every label with a fresh draw at a rate of **0.15**, seed **20261030**,

both declared here in advance. **No number measured on the rehearsal copy says anything
about the real pool**, by construction, so nothing measured there can tune the real run.

What the rehearsal has already exercised, before any real episode:

* the full scripted ladder on 1,320 designs, 240 purchases, every policy, fit gate included;
* the agent path end to end, conditions A and B1 through the MCP server with a scripted fake
  model: both episodes finished, 0 invalid tool arguments;
* the guard itself: the same tables refuse `AB-final` while serving `AB-rehearsal`.
