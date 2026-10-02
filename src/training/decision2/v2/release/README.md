# Decision 2.0 release engineering

One command takes a qualifying candidate from its scored checkpoint to a verified
**private** Hugging Face repository: build the package, reproduce the native
System One examples in separate processes, execute the card's own example,
upload, download the exact revision with the real `hf download`, re-hash every
file, reproduce the examples again from the download, and read everything back.
Adding a repository to the private "Decision 2.0" collection is a separate,
gated step. Nothing here makes a repository or collection public.

## Package layout (`dev2-package/1`, every size)

A package keeps the **exact scored model bytes at their checkpoint-relative
paths**, so `training.model.infer.checkpoint_fingerprint(<package>)` (or the
Kai native manifest) reproduces the scored identity from the downloaded bytes.
Each component uses its standard format: Transformers `save_pretrained`
backbone directories, PEFT adapter directories, safetensors weights and a
Hugging Face tokenizer.

```text
README.md                  product card (generated; see "Model card")
config.json                ROOT query file: vllm-sr-decision pointer, format_version 2
                           (the Hub's default download counter queries it and the
                           runtime reads it first); maps every model file by role
MODEL_MANIFEST.json        SHA-256 of every other file, parameter counts by component,
                           scored identity, origin, base binding, runtime and licence
LICENSE [NOTICE (upstream only)] [LICENSING.md]
assets/banner.png  assets/jevarena.png  assets/jevarena-types.png
assets/index-pareto.png  assets/index-areas.png
decision2/                 local System One runtime: __init__.py, api.py, <profile>.py
decision2/_vendor/         exact scored inference sources (hash-recorded in the manifest)
<model files by profile>
```

| Profile | Model files | Loads with |
| --- | --- | --- |
| `qwen-full` (0.8B–27B full; official-Qwen 0.6B causal) | `decision_config.json`, `decision_head.safetensors`, `[dec_residual.safetensors]`, `backbone/` (standard Transformers dir: `config.json`, `model*.safetensors`, index), tokenizer files at the root, `calibration.json` | vendored `training/model` modules; `AutoModel.from_pretrained(repo, subfolder="backbone")` and `AutoTokenizer.from_pretrained(repo)` also work |
| `qwen-adapter` (base-bound LoRA, e.g. 27B or 1.0 continuations) | as above but `adapter/` (standard PEFT dir) instead of `backbone/`; the base is **not** redistributed: `MODEL_MANIFEST.json` pins its repository, 40-hex revision and SHA-256 of every file the LoRA contract fingerprints | runtime downloads exactly those files at that revision (or takes `base_path`) and verifies them first |
| `kai-native` (0.6B Kai/Lex three-path encoder) | `native/` = the exact export tree verified by its own `MANIFEST.json` (encoder, Choice/Score paths, heads, tokenizer, native code) | vendored Kai runtime (`decision_runtime`, `decision_inference` from Kai `7185f514`); CPU or one ROCm GPU |
| `encoder-marker` (0.6B single-encoder exports) | `encoder/` = the export tree with its `MANIFEST.json` | layout defined; builder support is added with the first such candidate |

The runtime API is the same for every profile:

```python
from decision2 import Decision2          # after sys.path.insert(0, <package dir>)
model = Decision2.from_pretrained(dir)   # verifies every byte first; cpu or cuda:0
model.system_one(state=..., questions={...})  # {"model", "answers", "usage"}; no chat API
```

`verify_bundle(dir)` checks the exact inventory (only bytecode, HF local-dir
metadata and the Hub's `.gitattributes` are ignored), per-file SHA-256 and the
header parameter counts; `from_pretrained` additionally asserts the loaded
parameter count, the scored model identity and the calibration binding.
Model names are `Decision-2.0-<codename>-<size>B`; the codename follows the size
tier across generations (0.6B Kai, 0.8B Eos, 2B Sol, 4B Nox, 9B Lux, 27B Vega;
`layout.CODENAMES`). The size must match the loaded size tier (nearest tier
within the frozen 1.25 same-size ratio), or, with `"name_basis":
"loaded-parameters"` in the spec, the rounded loaded count inside that tier
(one decimal below 1B, whole billions from 1B; e.g. `Decision-2.0-Vega-26B` for
the ~27B tier's 25.75B text model), or, with `"name_basis": "base"`, the size
label of the base model (`name_base_model`, or the pinned `base.repo_id` of an
adapter; it must be in the declared weight lineage and in the loaded count's
tier). `Decision-2.0-Lux-9B` (Qwen3.5-9B base, 7,940,895,744 loaded) and
`Decision-2.0-Vega-27B` (Qwen3.8-27B base, 25,746,591,744 loaded) use it. The
repositories were released as `DEV2.0-<tier>` and renamed on 2026-10-02
(`layout.FORMER_REPOS`; the old IDs redirect): the guard refuses the former IDs,
and the successor gate accepts a current-revision gate receipt sealed under the
former ID of the same repository. Training state never enters a package.

## Release spec and commands

A release spec (`dev2-release-spec/1`, example: `specs/staging-kai1-dry-run.json`)
names the repository, profile, checkpoint and its expected identity, calibration,
input cap, direct weight origin, lineage licences (with pinned licence-file
hashes), the scored run binding and the card inputs. Release (non-staging) specs
also need the coordinator's gate receipt (`dev2-release-gate/1`).
Optional `"score_bias": {"path", "sha256"}` (Qwen profiles) packages per-level Score
logit offsets (`dev2-score-bias-v1`, from `training.model.infer --score-bias`) as
`score_bias.json`: bound to the model hash and, by value, to the offsets the scored
native manifest records (`score_bias.offsets`, `score_bias_sha256`), so a copy with
public-only `fit` provenance can be packaged (the screen refuses private paths).
Recorded as `MODEL_MANIFEST.json` `score_bias` (`file`, `sha256` of the packaged
file, `scored_sha256` of the applied file, `offsets`) and applied by the runtime
exactly as at scoring time. Without the key, packages and answers are unchanged.
Optional `"vendor_source"` and `"runtime_source"` name the decision2 tree of an exact
node mirror (`/data/dev2/src/<sha>-src_training_decision2/src/training/decision2`):
the first supplies the vendored scored inference sources (`training/model`), the
second the package runtime `decision2/*.py` (`v2/release/runtime`). Both default to
the builder's tree and are recorded under `MODEL_MANIFEST.json` `runtime`. A
card-only revision pins both to the mirror that built the revision it replaces, so
only card files change.

```bash
# local: push, then mirror the subtree to the node that holds the checkpoint
src/training/decision2/v2/common/mirror_to_node.sh --path src/training/decision2 node-a <commit>
# node: everything in one command (CPU shown; --gpu N --track T uses a leased GPU)
SRC=<commit>-src_training_decision2; S=/data/dev2/src/$SRC/src/training/decision2
$S/v2/release/release.sh --spec <spec.json> --src $SRC --work /data/dev2/runs/release/<id> \
  --cpu [--python <interpreter in the image>] [--mount <ro path>]... \
  [--parity typed-final:<goldfree prompts>:<sealed predictions>:<N>]... --upload \
  [--collect <gate.json>]
# Qwen3.5-family packages on a GPU: reproduce the scored kernel runtime (the image exposes FLA
# only through PYTHONPATH, which the isolated interpreter drops) with a copy of the scored run's
# persisted Triton autotune cache
$S/v2/release/release.sh ... --gpu <N> --track release --site /opt/decision-fla --require-kernels \
  --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR=<cache copy> --mount-rw <cache copy> ...
```

Individual steps (all write one receipt; the orchestrator calls them in order):

| Step | Command |
| --- | --- |
| build | `python3 -m v2.release.build --spec SPEC --output WORK/package/<repo-name>` |
| examples | `python -I -B v2/release/examples.py run --package PKG --output OUT [--device cpu] [--base-path B]` |
| repeatability | `python3 v2/release/examples.py compare A.json B.json --output R.json` (bit-identical required) |
| card example | `python -I -B v2/release/examples.py card --package PKG --reference A.json --output C.json` |
| scored parity | `python -I -B v2/release/examples.py parity --package PKG --panel NAME:PROMPTS:SEALED:N --output P.json` |
| hub | `<hf-cli python> -m v2.release.hub <ensure, upload, download, readback or collect> ...` and `python3 -m v2.release.hub tree ...` |

### Release gate and collection

The coordinator records the judgement part of the per-size gate in a decision
file (`dev2-release-decision/1`: model name, repository, checkpoint identity,
same-panel report SHA-256, paired-comparison SHA-256, `decided_by`, `rationale`)
and names it as `gate_receipt` in a `kind: release` spec; the builder refuses a
release spec whose decision names anything else. `release.sh ... --upload
--collect` then runs the whole chain and, only if every receipt passed,
`python3 -m v2.release.gate seal` writes `receipts/gate.json` binding the
decision to the uploaded revision and manifest (`gate evaluate --work W` prints
the six gate items with their evidence), and `hub collect` adds the repository
to the private "Decision 2.0" collection and reads it back.

While the coordinator's judgement is still pending (for example an independent
sealed confirmation), release engineering names a `status: draft` decision
(`prepared_by`, no `decided_by`) that binds the same identity, report and paired
comparison. It is enough to build, upload privately and verify (`--upload`
without `--collect`); `gate seal`, and therefore `--collect`, accepts only a
`status: final` decision with `decided_by`. Finalizing means: the coordinator
fills in the card's confirmation line in the spec, writes the final decision,
and reruns `release.sh ... --upload --collect` into a new work directory.

A size without a Decision 1.0 model (~27B) uses the **no-1.0 gate profile**
(coordinator 2026-09-29 11:55): the spec names `gate_profile: {"name": "no-1.0",
"reference": <card report key with role "reference">, "v3_share": 0.9,
"types": <v2.eval.gates types output for the scored run>}`, `card.paired` is the
candidate-minus-reference comparison, and gate item `1_near_first_tier_no_1_0`
replaces `1_beats_own_1_0`: v3 >= 0.9 x the reference peer's v3, the
human-transfer interval's upper bound >= 0 and every typed-FINAL type `OK`.
Its decisions also name `gate_profile` and `types_sha256`. The card then says
"There is no Decision 1.0 model at this size" and compares with the reference
peer where the own-1.0 model would appear.

A later revision of a collected release (for example a calibration-only
revision) needs its own final decision and `--upload --collect
--already-collected`. Collection items name a repository, not a revision, so
the repository is already in the collection when the pre-collect readback runs;
the flag makes that readback require the item instead of its absence, and
`gate seal` then binds the new decision to the new revision.

`hub download` is the real `hf download <repo> --revision <sha> --local-dir <fresh
dir>` with a fresh cache. `readback` checks the private flag, the exact revision,
each remote file's LFS SHA-256 or git blob id against the package, the Hub-parsed
card metadata, that every card link and image resolves, and collection
membership (staging repositories must not be in the collection). `collect` only
accepts a `Decision-2.0-*` release package with a gate receipt naming the exact
revision and manifest. `ensure`, `upload`, `readback` and `collect` refuse a
repository ID that the Hub resolves to a different repository (the old ID of a
renamed repository redirects to the new one), so a superseded spec cannot write
to a retired name.

## Model card

`card.py` writes a product card. README.md, in order: YAML metadata (`license`,
`base_model`, `base_model_relation`, `library_name: transformers`, tags); the
banner; `# Decision-2.0-<codename>-<size>`; one paragraph; an at-a-glance table
(parameters, context length, decision types, licence); **Highlights** (top
JevArena score of its size, only when true and "statistically level" when the
paired interval includes zero; the gain over the Decision 1.0 counterpart on
JevArena and, only when positive, on the Jev Decision Index, or for a size
without Decision 1.0 the gain over the family's next size when it is the
family's best; the median single-question latency from a pinned bench receipt;
many questions in one forward pass); **Quickstart**, code only (the pip line and
one Python block: `AutoModel` + `system_one`, which the verifier executes, and
one commented `pipeline("decision")` line); **Evaluation** (one table of
JevArena, human-labelled transfer and Index; JevArena overall and by decision
type; the Index against model size and by area, with the Index footnote);
**License**; **Citation**. No training details, limitations, NOTICE of our own,
attributions or `evaluation/` pages. `lint_readme` refuses internal vocabulary
(panel versions, post-key, Brier / ECE, mlx-diag, JevBench, training and
precision details, removed sections and files, native-runtime usage, revision
hashes) on top of the shared `lint`.

The banner and charts are PNGs rendered by `card_assets.py` (matplotlib, Inter,
white background, blue #30A0FC and yellow #FCB414, the vLLM-SR logo bottom-right
of every chart) outside the build. The banner's focal point is the codename in a
`#0A5BD8` → `#5CC8FF` gradient (Inter Display Bold), with the size in ink on its
baseline, the "DECISION 2.0" eyebrow, the tagline, the small logo top-left and
the translucent V-mark of the logo bleeding off the right edge;
`tests/test_card_banner.py` renders all six sizes and checks that no element
overlaps another or the V-mark (set `DEV2_CARD_FONTS` to the Inter TTF directory).
Its receipt `card-assets.json` holds input and output SHA-256
digests only. The spec pins the private Index input (`card.index`, schema
`dev2-card-index/1`, see `card_index.py`), the rendered assets directory
(`card.assets`) and the bench receipt (`card.speed`); the build re-checks that
the receipt names exactly the card's reports, Index input and weights before it
copies the PNGs. `python -m v2.release.card_index` builds the Index input from
the kit runs (one per released weights) and the board snapshot. Every point on
the size axis follows the board's served-parameter convention: a Decision 2.0
point takes the board's count for its own base (`BOARD_BASES`), and the
at-a-glance table keeps the loaded count. `load` refuses any other footnote than
`FOOTNOTE` (the kit edition, the snapshot date and the row-level training-data
audit). Index values therefore appear only on the published cards, never
in commits or records; tests use synthetic values (`tests/card_fixture.py`).
Internal release facts (gate items, every result below the counterpart, decision
IDs, revision and weights hashes) stay in the build receipt and release records.

Comparators pass a fail-closed licence filter (`licence.py`): CC BY-NC,
research-only, unknown and internal-only models are excluded. A peer measured
from a sibling of its Decision Index board artifact (e.g. the BF16 weights of an
FP8 entry) names `board_entry` and takes that roster entry's licence only if the
entry names it as its base model. Card metadata is `apache-2.0` only when every
upstream component of the weight lineage is Apache-compatible; otherwise `other`
with `LICENSING.md`.

Spec `card.text` keys: `description` (replaces the paragraph) and
`staging_notice`; unknown keys are refused.

`brand/` keeps the owl banners of earlier card revisions; current cards do not
use them.

## Tests

```bash
cd src/training/decision2 && python3 -m unittest v2.release.tests.test_release   # stdlib only
# in the pinned image, CPU only: tiny real qwen-full and qwen-adapter packages end to end
python3 -m v2.release.tests.cpu_integration --tokenizer <Qwen3 tokenizer dir> --work <new dir>
```
