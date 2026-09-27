# JevArena v3 first-release planning

`plan_first_release_v3.py` is a command-only planner and gold-free pre-key
auditor. It does not train, infer, score, open FINAL labels, or publish. The
first-release sealed score uses exactly typed FINAL 1,600 plus the 15-task CSS
evaluation 6,547. The public JevBench 231 is run and reported separately.
Authored questions are reserved for JevArena v3.1. The older
[`FINAL_EVAL.md`](FINAL_EVAL.md) and `plan_final_eval.py` retain their own
historical v2 meaning.

## Pretest inputs

1. An existing `decision2-pretest-freeze/1` candidate lock with every selected
   checkpoint and CAL. The reused checker validates training completion,
   source/adapter/head/tokenizer hashes, CAL lineage, and, for Eikos SemIf,
   standalone package rights and full DEV/CSS-pilot parity receipts. Never put
   FINAL or CSS evaluation in `selection_sources`.
2. The frozen gold-free CSS evaluation prompt file with its pinned 6,547-row
   SHA-256. This planner does not read the CSS label file.
3. A pinned public JevBench panel directory with `manifest.json` and
   `prompts.jsonl`. The public target file is not opened by this planner.
4. A private roster JSON, schema `decision2-first-release-v3-pretest-roster/1`:

```json
{
  "schema_version": "decision2-first-release-v3-pretest-roster/1",
  "candidate_keys": ["d2-4b"],
  "candidate_size_b": {"d2-4b": 4.2},
  "baseline_keys": ["nox", "eikos4b"],
  "baseline_attestations": [
    {
      "key": "nox",
      "model_id": "llm-semantic-router/Decision-1.0-Nox-4B",
      "revision": "PINNED_HF_REVISION",
      "size_b": 4.0,
      "native_model_sha256": "LOWERCASE_64_HEX",
      "adapter_sha256": "LOWERCASE_64_HEX",
      "calibration_sha256": null,
      "receipt_path": "/PRIVATE/nox-package-receipt.json",
      "receipt_sha256": "LOWERCASE_64_HEX"
    }
  ],
  "pairs": [
    {"candidate": "d2-4b", "comparator": "nox", "size_relation": "same", "rationale": ""}
  ],
  "gate_document_sha256": "LOWERCASE_64_HEX"
}
```

The example is illustrative and incomplete: add the `eikos4b` attestation
and exact numeric values before planning. Every selected 2.0 size needs one
predeclared 1.0 comparator; an actual-size mismatch must say `nearest` and
include a substantial rationale. At least one separate open-model control is
required. The `candidate_size_b` and baseline `size_b` values must be measured
loaded parameter counts, not rounded model names. `same` additionally requires
the preregistered family match and a measured maximum/minimum parameter ratio
at most 1.25; larger ratios require `nearest` and cannot support an equal-size
win claim.

Each baseline attestation points to a private JSON receipt with the same
`model_id`, `revision`, `native_model_sha256`, `adapter_sha256`, and
`calibration_sha256`. It also contains `package_path` equal to the model path
used by native inference, plus `files` mapping **every** relative package file
to its SHA-256. For backends with an external runtime, include `runtime_path`
and every file under `runtime_files`. The planner rehashes all listed package
and runtime files and rejects omissions or additions. Compute
`native_model_sha256` as SHA-256 of the compact, lexically sorted JSON
serialization of the `files` map (`sort_keys=True`, separators `(',', ':')`).
The adapter digest is the source SHA-256 of the selected native inference
module. Hosted Jev has no local package and is compared on a separate track.
The roster and receipts stay private; they may expose local paths.

The numeric rules are in
[`jev-arena-v3-first-release-gates-2026-09-27.md`](../research/jev-arena-v3-first-release-gates-2026-09-27.md).
The roster binds the exact SHA-256 of that document. The planner also hashes
the typed/CSS scorers, v3 aggregate and paired-CI implementation, public
scorer/ranker, native adapters, publication code, and its own source.

## Create the plan without opening FINAL labels

Run from the exact local source mirror. The output directory must not yet
exist. Keep the resulting plan in a private path, then record its SHA-256
before executing any command. The plan contains local package paths, so it
must not be copied to a public gist or model card.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE_ROOT" python3 -m scripts.plan_first_release_v3 plan \
  --candidate-freeze "$CANDIDATE_LOCK" --roster "$V3_ROSTER" \
  --css-prompts "$CSS_GOLD_FREE_PROMPTS" --public-panel "$PUBLIC_PANEL" \
  --source-root "$SOURCE_ROOT" --evaluation-root "$NEW_EVAL_ROOT" \
  --model-root "$MODELS_ROOT" --external-root "$EXTERNAL_ROOT" \
  --kai-lex-python "$QUALIFIED_KAI_PYTHON" --fla-path "$QUALIFIED_FLA" \
  > "$PRIVATE_PLAN_JSON"
sha256sum "$PRIVATE_PLAN_JSON"
```

The emitted `preparation_commands_after_candidate_freeze` create a fresh
private typed FINAL seed and gold-free prompts **only after** the candidate
freeze is locked. The typed generator also writes labels; keep those bytes
inaccessible to the model-selection process. Native inference commands then
produce typed, CSS and public predictions with one frozen model identity per
run. Hash all raw predictions before using any scoring command.

The saved plan also contains `comparison_pairs` and their canonical
`comparison_pairs_sha256`. That digest fixes which 1.0 model each candidate
must beat. The same digest and mapping must be bound into the pre-key freeze
and checked by the publication gate. The JevArena rank scorer validates the
frozen roster and pair digest; the publication gate validates the exact
candidate-to-1.0 mapping and numeric comparison. A plan alone is **not** a
release pass.

After prediction generation, but before opening either held-out label file,
run the read-only audit with the recorded plan digest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE_ROOT" python3 -m scripts.plan_first_release_v3 audit-predictions \
  --plan "$PRIVATE_PLAN_JSON" --expected-plan-sha256 "$PLAN_SHA256" \
  > "$PRIVATE_PREKEY_AUDIT_JSON"
```

The audit revalidates candidate package/CAL/parity, baseline package files,
source code, gate document, full gold-free prompt IDs and input hashes,
8,147 sealed-core plus 231 public predictions per model, native prediction
manifests and the saved raw prediction SHA list. It never reads
`typed-final.gold.jsonl` or the CSS gold
file. Save the audit and SHA in the private ledger.

Only after that audit, a separate pre-key freeze receipt must bind the typed
and CSS **gold digests**, every model's three prediction digests, native model,
adapter and calibration identities, the candidate lock, the plan and audit
receipt digests, comparison pairs, numeric policy and source hashes, under
`jevarena-v3-freeze/2`. The auditable event order is
candidate and policy lock, complete prediction seals, gold-free audit,
pre-key receipt seal, then first label access. Write the receipt with exclusive
creation and record its time and SHA. Replace the placeholder in the planned
`arena_roster` template with that actual receipt SHA; write both the arena and
public roster files. The v3 scorer independently verifies the freeze receipt.
The first-release planner does not generate a pretend freeze receipt or make
an automatic publish decision. Run its deferred scoring, paired-CI and rank
commands only after this freeze, followed by the separate v3 publication
gate and full package parity audit.
