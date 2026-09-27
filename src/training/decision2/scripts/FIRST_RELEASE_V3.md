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
   For `d2-4b`, the v3 lock additionally needs `v3_eikos_stable_runtime` at
   the lock top level. It binds the independent CSS repeatability execution,
   full DEV/CSS-pilot/public execution, and combined stable-backend parity
   receipts with absolute private paths and SHA-256s:

```json
{
  "v3_eikos_stable_runtime": {
    "d2-4b": {
      "repeat_execution": {"path": "/PRIVATE/repeat/execution.receipt.json", "sha256": "LOWERCASE_64_HEX"},
      "full_execution": {"path": "/PRIVATE/full/execution.receipt.json", "sha256": "LOWERCASE_64_HEX"},
      "full_parity": {"path": "/PRIVATE/full/parity-dev-css.receipt.json", "sha256": "LOWERCASE_64_HEX"}
    }
  }
}
```

   Set the candidate's `parity_reports.dev` and `parity_reports.css_pilot` to
   the complete fixed-backend reports in that full-execution directory. The
   planner verifies all receipt hashes, package/CAL/collector identity, two
   independent CSS runs, selected-source/package parity, validity and the
   exact FLA-to-PyTorch reference runtime keys. Historical v2 reports remain
   unchanged; newly generated publication artifacts use the public ID family.
   The candidate `model_id`, each repeat/full/combined-parity receipt, and
   every Eikos prediction manifest must all use
   `llm-semantic-router/DEV2.0-4B`. Recollect the two CSS repeat runs and
   complete DEV/CSS-pilot/public predictions using the new collector source,
   rerun the gold-free direct parity reports, reseal combined parity and
   execution receipts, then create a new prospectively reviewed candidate
   lock. Historical lowercase-ID receipts cannot qualify this lock.
2. The frozen gold-free CSS evaluation prompt file with its pinned 6,547-row
   SHA-256. This planner does not read the CSS label file.
3. A pinned public JevBench panel directory with `manifest.json` and
   `prompts.jsonl`. The public target file is not opened by this planner.
4. A private roster JSON, schema `decision2-first-release-v3-pretest-roster/2`:

```json
{
  "schema_version": "decision2-first-release-v3-pretest-roster/2",
  "candidate_keys": ["d2-4b"],
  "candidate_size_b": {"d2-4b": 4.2},
  "baseline_repeat_panel": {
    "choice32_path": "/PRIVATE/choice32.prompts.jsonl",
    "typed_dev_path": "/PRIVATE/typed-dev.prompts.jsonl",
    "prompts_path": "/PRIVATE/baseline-repeat32.prompts.jsonl",
    "manifest_path": "/PRIVATE/baseline-repeat32.manifest.json"
  },
  "baseline_keys": ["nox", "OPEN_CONTROL_KEY"],
  "baseline_attestations": [
    {
      "key": "nox",
      "model_id": "llm-semantic-router/Decision-1.0-Nox-4B",
      "revision": "PINNED_HF_REVISION",
      "size_b": 4.208383488,
      "native_model_sha256": "LOWERCASE_64_HEX",
      "adapter_sha256": "LOWERCASE_64_HEX",
      "calibration_sha256": null,
      "receipt_path": "/PRIVATE/nox-package-receipt.json",
      "receipt_sha256": "LOWERCASE_64_HEX"
    },
    {
      "key": "OPEN_CONTROL_KEY",
      "model_id": "PINNED_OPEN_MODEL_ID",
      "revision": "PINNED_HF_REVISION",
      "size_b": 4.2,
      "native_model_sha256": "LOWERCASE_64_HEX",
      "adapter_sha256": "LOWERCASE_64_HEX",
      "calibration_sha256": null,
      "receipt_path": "/PRIVATE/open-control-package-receipt.json",
      "receipt_sha256": "LOWERCASE_64_HEX"
    }
  ],
  "baseline_repeatability": [
    {
      "key": "nox",
      "receipt_path": "/PRIVATE/nox-repeat32.receipt.json",
      "receipt_sha256": "LOWERCASE_64_HEX",
      "first_predictions_path": "/PRIVATE/nox-repeat32.first.jsonl",
      "second_predictions_path": "/PRIVATE/nox-repeat32.second.jsonl"
    },
    {
      "key": "OPEN_CONTROL_KEY",
      "receipt_path": "/PRIVATE/open-repeat32.receipt.json",
      "receipt_sha256": "LOWERCASE_64_HEX",
      "first_predictions_path": "/PRIVATE/open-repeat32.first.jsonl",
      "second_predictions_path": "/PRIVATE/open-repeat32.second.jsonl"
    }
  ],
  "pairs": [
    {"candidate": "d2-4b", "comparator": "nox", "size_relation": "same", "rationale": ""}
  ],
  "gate_document_sha256": "LOWERCASE_64_HEX"
}
```

The example is illustrative and deliberately cannot run with
`OPEN_CONTROL_KEY`: replace it with a pinned catalog key only after that
model's native loader, two-process repeatability and all-file attestation pass
a new prospective gold-free preflight. Replace every placeholder and rounded
size with measured exact values before planning. Earlier Eikos 4B and Decider
smokes failed their frozen repeatability thresholds; neither is currently an
eligible first-release control merely because it remains in the catalog.
Every selected 2.0 size needs one
predeclared 1.0 comparator; an actual-size mismatch must say `nearest` and
include a substantial rationale. At least one separate open-model control is
required. The `candidate_size_b` and baseline `size_b` values must be measured
loaded parameter counts, not rounded model names. `same` additionally requires
the preregistered family match and a measured maximum/minimum parameter ratio
at most 1.25; larger ratios require `nearest` and cannot support an equal-size
win claim.

Every selected 1.0 or open baseline also needs a separate two-process smoke
receipt. The planner reconstructs the fixed 32 gold-free prompts from the
source files named in `baseline_repeat_panel`, compares the rebuilt bytes and
manifest with the stored files, and requires prompt SHA-256
`3376ed4093c7efb591519912b88605960923cf33fd19d8adfa5018eb02270a55`.
For each baseline it rehashes the receipt and both prediction files, reruns
the native comparison against the pinned model ID, revision, backend and
adapter version, and requires zero category changes and maximum option
probability drift at most `1e-6`. All 32 rows must match the rebuilt input
fingerprints and the same attested package config, weights and runtime source
digests used for full evaluation. Missing or failed smoke evidence blocks the
plan before any sealed prediction. The plan saves each baseline's smoke and
package attestation hashes; the gold-free pre-key audit verifies them again.

Each baseline attestation points to a private JSON receipt with the same
`model_id`, `revision`, `native_model_sha256`, `adapter_sha256`, and
`calibration_sha256`. It also contains `package_path` equal to the model path
used by native inference, plus `files` mapping **every** relative package file
to its SHA-256. For backends with an external runtime, include `runtime_path`
and every file under `runtime_files`. The planner rehashes all listed package
and runtime files and rejects omissions or additions. Compute
`native_model_sha256` as SHA-256 of the compact, lexically sorted JSON
serialization of the `files` map (`sort_keys=True`, separators `(',', ':')`).
Kev's composite digest instead covers all three package/source file maps and
its native loader count receipt, while `files` retains its package-only meaning.
The adapter digest is the source SHA-256 of the selected native inference
module. Hosted Jev has no local package and is compared on a separate track.
The roster and receipts stay private; they may expose local paths.

Use `scripts.baseline_attestation_v3` to create each roster item and its
receipt; do not hand-write the digests or size. Its `build` command takes one
pinned key from `plan_final_eval.BASELINES`, the exact inference model root,
source root, and external native-runtime root. The two output files must have
new paths in a caller-owned `0700` private directory; the tool creates each
as `0600`, refuses overwrite, and prints neither contents nor paths. For
example, with private variables already set:

```bash
PYTHONPATH="$SOURCE_ROOT" python3 -m scripts.baseline_attestation_v3 build \
  --key sol --source-root "$SOURCE_ROOT" --model-root "$MODEL_ROOT" \
  --external-root "$EXTERNAL_ROOT" \
  --loaded-parameter-count "$NATIVE_LOADER_PARAMETER_COUNT" \
  --receipt-output "$PRIVATE_RECEIPT" \
  --attestation-output "$PRIVATE_ATTESTATION"
PYTHONPATH="$SOURCE_ROOT" python3 -m scripts.baseline_attestation_v3 verify \
  --key sol --source-root "$SOURCE_ROOT" --model-root "$MODEL_ROOT" \
  --external-root "$EXTERNAL_ROOT" \
  --attestation-output "$PRIVATE_ATTESTATION"
```

Record `NATIVE_LOADER_PARAMETER_COUNT` from the exact native loader's unique
`model.parameters()` count, before building; it must equal the independently
counted elements in all package safetensors files. Include
`--calibration-file "$CAL_FILE"` when the adapter loads a separate CAL file;
an in-package CAL remains covered by the full package hash. The receipt
records the integer count, exact `size_b = count / 1e9`, all weight filenames,
and the pinned adapter/CAL hashes. The tool rejects symlinks, special files,
unsafe output permissions, duplicate tensor names, malformed safetensors,
non-safetensors-only weights and any count mismatch. Its verification runs
again inside the v3 planner, so a changed package, runtime, adapter or CAL
blocks planning. This check does not prove the native loader actually uses
every file: preserve the loader count receipt and native prediction/parity
evidence separately. Never use the rounded model name as `size_b`.

The gold-free prediction audit checks each planned baseline identity against
its reverified attestation and requires `revision_attested: true` in every
native row. It also compares native per-row config, weight, release-manifest
and runtime-source digests with the exact attested files where the native
adapter exposes those fields. Kev's composite fingerprint follows its native
three-file recipe. A config digest is not the whole package digest: adapters
that do not emit a full-package per-run digest still require the byte-complete
attestation at planning and pre-key audit, plus separately retained native run
evidence. Do not claim a config match alone proves weight identity at runtime.

Kev is a composite package: its published package contains a LoRA adapter and
`head.pt`, while its native FP32 loader reads a separately pinned Qwen Base and
merges the LoRA into the text backbone. Build its private attestation with the
same command plus `--native-loaded-count-receipt "$NATIVE_COUNT_RECEIPT"`. The
count receipt must come from the same pinned native loader revision. The
attestation hashes every file in the Kev package, Qwen Base local directory,
and author runtime tree; checks their pinned revisions and release provenance;
and binds the count receipt SHA-256. Its loaded size is 4,207,062,528
parameters: 4,205,751,296 text backbone plus 1,311,232 pointer head. The
adapter's 32,464,896 tensor elements are merged, so they are not additional
inference parameters. The full Qwen download also contains visual and MTP
tensors that this native loader does not instantiate. Verification rehashes
the count receipt and every component file. For all other baselines, the
original packaged safetensors count equals native loader count rule remains
strict. Keep private receipts and package paths out of public reports.

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
run. All three Eikos commands explicitly enable deterministic algorithms and
the verified PyTorch reference gated-delta backend. Hash all raw predictions
before using any scoring command. The Eikos public scorer uses the same 231
answer rows without its optional generic manifest argument; the pre-key audit
instead verifies the Eikos native manifest, including package/CAL, collector
and stable runtime identity. Other candidate public commands retain the
generic manifest argument.

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
manifests, the exact Eikos stable backend in each of its three manifests, and
the saved raw prediction SHA list. It never reads `typed-final.gold.jsonl`
or the CSS gold file. Save the audit and SHA in the private ledger.

Only after that audit, a separate pre-key freeze receipt must bind the typed
and CSS **gold digests**, every model's three prediction digests, native model,
adapter and calibration identities, the candidate lock, the plan and audit
receipt digests, comparison pairs, numeric policy and source hashes, under
`jevarena-v3-freeze/2`. The auditable event order is
candidate and policy lock, complete prediction seals, gold-free audit,
pre-key receipt seal, then first label access. Write the receipt with exclusive
creation and record its time and SHA. A private chronology JSON must contain
exactly `schema_version: jevarena-v3-prekey-chronology/1` and
`candidate_lock`, `prediction_seal`, and `audit_seal` objects, each with
`at_utc` and `sha256`. Their hashes must respectively identify the candidate
lock, `RAW_PREDICTIONS.sha256`, and the saved audit; the times must be strictly
ordered. Preserve independently recorded events so a reviewer can corroborate
the declared times. The ranker checks their bound identities and order, not an
external trusted clock.

The private freeze generator accepts only the two gold **digest strings**, not
label paths. It reruns the full gold-free audit, requires byte-exact agreement
with the saved audit, checks formula and paired-bootstrap policy, and writes
the receipt once into an existing owner-held mode-0700 directory as mode 0600.
It refuses a destination inside the code checkout and never prints private
paths or prompts:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE_ROOT" python3 -m scripts.freeze_first_release_v3 \
  --plan "$PRIVATE_PLAN_JSON" --plan-sha256 "$PLAN_SHA256" \
  --prediction-audit "$PRIVATE_PREKEY_AUDIT_JSON" \
  --prediction-audit-sha256 "$AUDIT_SHA256" \
  --chronology "$PRIVATE_CHRONOLOGY_JSON" \
  --chronology-sha256 "$CHRONOLOGY_SHA256" \
  --typed-gold-sha256 "$TYPED_GOLD_SHA256" \
  --css-gold-sha256 "$CSS_GOLD_SHA256" \
  --output "$PRIVATE_FREEZE_JSON"
```

An invalid receipt created after a concurrent file mutation stays preserved
for audit; use a new private pre-key run rather than overwriting it. Replace
the placeholder in the planned
`arena_roster` template with that actual receipt SHA; write both the arena and
public roster files. The v3 scorer independently verifies the freeze receipt.
The generator makes no publish decision. Run deferred scoring, paired-CI and rank
commands only after this freeze, followed by the separate v3 publication
gate and full package parity audit.
