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
LICENSE NOTICE ATTRIBUTIONS.md [LICENSING.md LICENSES/...]
assets/DEV2.0-<size>-owl-banner.png
assets/jevarena-v3-rank.svg  assets/jevarena-v3-model-task.svg  assets/jevbench-public231-rank.svg
evaluation/EVALUATION.md  evaluation/manifest.json
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
The model name must match the loaded size tier (`DEV2.0-<tier>`, nearest tier
within the frozen 1.25 same-size ratio). Training state never enters a package.

## Release spec and commands

A release spec (`dev2-release-spec/1`, example: `specs/staging-kai1-dry-run.json`)
names the repository, profile, checkpoint and its expected identity, calibration,
input cap, direct weight origin, lineage licences (with pinned licence-file
hashes), the scored run binding and the card inputs. Release (non-staging) specs
also need the coordinator's gate receipt (`dev2-release-gate/1`).

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
| banners | `python v2/release/banner.py --sources <1.0 headers> --fonts <dir> --output-dir brand/` (needs Pillow) |

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
accepts a `DEV2.0-*` release package with a gate receipt naming the exact
revision and manifest.

## Model card

`card.py` follows the Decision 1.0 product card: owl banner, what the model is
for, the three decision types, the same-panel score table, the v3 rank chart,
the model × task chart and the public-231 rank chart (all from the eval track's
`v2.eval.charts` on REPORT.json files only), an automatic **tradeoffs** table of
every per-type, per-task and public-tier result below the tier's own Decision
1.0 model, a runnable local System One example (executed by the verifier), and
short model details and limits. No Pareto chart, no internal gate ledger.
Comparators pass a fail-closed licence filter (`licence.py`): CC BY-NC,
research-only, unknown and internal-only models are excluded. Card metadata is
`apache-2.0` only when every upstream component of the weight lineage is
Apache-compatible; otherwise `other` with `LICENSING.md`.

## Banners

`brand/DEV2.0-{0.6B,0.8B,2B,4B,9B,27B}-owl-banner.png` use the Decision 1.0
composition: the tier's 1.0 mosaic owl (pixels unchanged; Kai, Eos, Sol, Nox,
Lux), a new owl with a ringed planet for 27B (`brand/sources`, generated in the
same style), the `DEV2.0` wordmark, a `DECISION 2.0` pill and the size in the
tier's 1.0 accent colour. `brand/BANNERS.json` records every source hash.

## Tests

```bash
cd src/training/decision2 && python3 -m unittest v2.release.tests.test_release   # stdlib only
# in the pinned image, CPU only: tiny real qwen-full and qwen-adapter packages end to end
python3 -m v2.release.tests.cpu_integration --tokenizer <Qwen3 tokenizer dir> --work <new dir>
```
