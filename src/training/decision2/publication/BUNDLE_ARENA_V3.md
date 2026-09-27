# JevArena v3 first-release card and package

This is a **separate** release path. `generate_arena` and `bundle_arena`
continue to mean the previous six-axis v2 protocol; their reports, manifests
and gates are not converted or renamed. V3 uses only typed FINAL 1,600 and CSS
human transfer 6,547 in its sealed-core score. The 231 exposed JevBench items
are required for the first card but reported as an independent public rerun.
Decision Bench, Decision Index, and new authored questions are not v3 release
axes; independently authored questions can extend a later v3.1 protocol.

## Gold-free publication inputs

1. Run `jev_arena.arena_v3` with its exact pre-key freeze to obtain
   `jevarena-ranking/3`. A scorer output has status
   `scored_pending_independent_release_audit`; this is not a release pass.
2. Run `jev_arena.public_rank` on the same model roster and pinned public231
   scorer reports. All model IDs, revisions and actual parameter counts must
   match the v3 ranking. Public rank/score are never included in v3's
   `100 * sqrt(T * H)` headline.
3. Run `jev_arena.compare_v3` using paired original predictions and gold for
   the chosen 2.0/1.0 comparison. The release gate consumes its **joint**
   aggregate interval. It cannot reconstruct an aggregate interval by
   combining marginal typed and transfer intervals.
4. Generate card artifacts from a JSON config with three keys:

```json
{
  "arena_rank": "reports/arena-v3.json",
  "jevbench_public_rank": "reports/jevbench-public.json",
  "comparison_pairs": [
    {
      "new": "decision2-key",
      "old": "decision1-key",
      "typed_comparison": "reports/typed-paired.json",
      "transfer_comparison": "reports/css-paired.json",
      "new_typed_report": "reports/d2-typed.json",
      "old_typed_report": "reports/d1-typed.json",
      "new_transfer_report": "reports/d2-css.json",
      "old_transfer_report": "reports/d1-css.json"
    }
  ]
}
```

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.generate_arena_v3 \
  --config release-v3-card.json --output-dir new-v3-card-artifacts
```

The generator requires matching rosters, v3's exact two axes, 8,147 sealed
items, the public231 panel, and prediction-bound 2.0/1.0 paired reports. It
emits a score table and separate JevArena and JevBench rank/Pareto SVGs plus
two-axis and model-by-task matrices. The card and chart style follows the
earlier Decision family; the mosaic sticker remains a distinct 2.0 asset.
Ranks and Pareto status are relative to the identical displayed roster.

## Native package gate

`publication.bundle_arena_v3` takes a functional native model directory and
the v3 artifacts. Its JSON config uses the same top-level paths as the v2
packager: `model_dir`, `artifacts`, `arena_rank`, `public_rank`,
`package_record`, `parity_receipt`, `release_gate`, `provenance_inputs`,
`freeze_manifest`, `gate_evidence`, `score_key` and `score_inputs`. For a
pinned external-base PEFT package, also set `base_source` and
`adapter_source_parity_receipt`. New v3 inputs are `aggregate_pair_report`,
`old_typed_report` and `old_css_report`.

`score_inputs` has **exactly** `typed`, `css`, and `public`; each contains
`score`, `predictions`, and `native_manifest` paths. No authored or Decision
Bench score file is required or accepted. `provenance_inputs` names the exact
`data_manifest`, `run_provenance`, and `training_code` files hashed by the
reviewed `decision2-release-package-record/2`. All paths may be absolute or
relative to the config file. Scores and scored predictions stay private; the
public bundle contains hashes and review receipts, not gold or raw rows.

```bash
PYTHONPATH=src/training/decision2 python3 -m publication.bundle_arena_v3 \
  --config release-v3-package.json --output new-model-repository

PYTHONPATH=src/training/decision2 python3 -m publication.bundle_arena_v3 \
  --verify new-model-repository
```

The bundle uses the same native rights, weight inventory, actual parameter
count, full package/source adapter parity, and DEV1,600/CSS-pilot1,430
gold-free parity validators as the v2 package. It additionally verifies that
every scored native run used the frozen native checkpoint, adapter,
calibration and model revision; the public scorer must use the pinned 231
prompts and targets. The selected v3 ranking row's `native_model_sha256`
must equal the verified native checkpoint fingerprint. For external PEFT,
this includes the pinned base dependency and does not merge or copy its
weights into the publication repository.

The release gate schema is `decision2-jevarena-v3-release-gate/1`. All six
reviews must pass and carry exact source hashes:

- `candidate_freeze`: `decision2-jevarena-v3-freeze-audit/1`, binding the v3
  pre-key freeze, candidate model, scorer and policy versions, and typed/CSS
  prediction seals. Its declared chronology must satisfy freeze before
  prediction seals before first protected label access.
- `train_eval_overlap`, `same_panel_evaluation`, `rights_and_provenance`,
  `native_parity`, `release_thresholds`: each uses
  `decision2-jevarena-v3-release-check/1`, a reviewer commitment, a source
  evidence digest and the exact release-context hash. No authored editorial
  review is required for v3 because authored items are absent.

The `release_thresholds` review must bind the pre-key policy and the exact
joint paired-bootstrap report and 1.0 comparator scorer reports. The program
recomputes the numeric rules from those reports: both T and H strictly
increase, the joint v3 aggregate paired 95% interval has lower bound above
zero with at least 5,000 fixed-seed replicates, no Choice/Noul/Score accuracy
drop exceeds 0.02, invalid/missing fractions on each axis do not exceed
`max(0.02, comparator + 0.01)`, and coverage-adjusted typed Brier does not
exceed the comparator plus 0.03. The coverage adjustment assigns normalized
Brier 1 to a typed item without an accepted probability answer, using the
`probability_n` count in each scorer report. The original valid-only Brier
and probability coverage remain available separately. The exact
preregistration is
[`jev-arena-v3-first-release-gates-2026-09-27.md`](../research/jev-arena-v3-first-release-gates-2026-09-27.md),
whose SHA-256 must match the pre-key freeze. A failing dimension remains
HOLD; public231 cannot compensate for a sealed-core failure.

**An evidence JSON with a reviewer identity and timestamp is a structured
assertion, not proof that the reviewer was independent or that the chronology
is genuine.** The release owner must inspect the external append-only
timestamp log, source receipts and raw private paired outputs. The package
assembler is a byte and process gate, not an inference runner or an external
review service. Synthetic CPU test fixtures are not release evidence.

No command here uploads a model, creates a collection, opens sealed labels,
or validates a real candidate's performance. Publish only a package whose
full same-panel score, joint comparison and external review actually pass.
