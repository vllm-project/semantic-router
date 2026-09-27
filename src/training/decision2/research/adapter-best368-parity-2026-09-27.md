# Decision 2.0 27B adapter-preserving package: private native parity dry run

Status: **technical parity PASS; release HOLD**. This is a gold-free packaging
check for the existing 27B BEST368 checkpoint. It does not transfer its
development scores to a release package, establish a JevArena result, or
authorize a Hugging Face upload.

## Frozen identities and provenance

| Evidence | SHA-256 or immutable revision |
| --- | --- |
| Scored checkpoint model identity | `d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2` |
| Native scored prediction manifest | `b2e33b1a4daa410ab6ab1d402692b296298fb29defa771c8c5cf6344e84b0760` |
| Exact CAL file | `e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78` |
| Upstream base repository | `Qwen/Qwen3.8-27B` |
| Upstream base commit | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Signed source reconstruction, isolated experiment only | `61925c09c` |
| Signed tokenizer screening correction | `a34df7997` |
| Exact signed source archive | `043ddecba5c7fc52bbc360279c085cf35952abe4819ac35ecc7a8384d4e2e2b9` |
| Private adapter package manifest | `e6ca0f51bf7f27c938f35cbdbc6802d72b10a1d3d33405765fe491c32e25110b` |
| Gold-free parity roster | `6b2e16cf55533d5486887f14217e85415d9b607bc4b1788a9dbbd41fa22ec32c` |
| Private parity receipt | `c23868b95324634aa0a1c2ef8fd4408e0e6118f0966ec957b51f4cc1771b4b2d` |

The package builder checked the original native scored manifest's model file
map, calibration file, context limit, inference source hashes, Torch and PEFT
versions against the exact checkpoint. The six frozen inference source files
matched the scored manifest byte for byte. Five of these six files have since
changed on the main Decision 2.0 development branch; `source.py` is unchanged.
The reconstruction commit is isolated and must **not** be merged as a source
rollback. The publication-only tokenizer correction can be selected
independently. It allows slash-leading public BPE merge tokens in
`tokenizer.json` while still rejecting absolute values in other JSON positions
and private paths, addresses and credential-like content.

The upstream repository returned the declared commit and all 28 pinned source
filenames through the Hugging Face CLI. A fresh CLI download of `config.json`
matched the existing snapshot's SHA-256. All 18 local safetensors shard hashes
matched their saved HF download metadata revision and LFS object identifiers.
This supports snapshot provenance; the later release audit must still verify
the exact external base dependency and its redistribution terms.

The package counts **25,688,227,840 loaded text-inference parameters**:
25,624,600,064 base text parameters, 58,363,904 LoRA parameters, and
5,263,872 decision-head parameters. It does not include the unused upstream
vision path or language-generation head. The candidate package omits the base
weights and binds them by repository commit and per-file hashes.

## Runtime evidence scope

| Package | Current parity runtime lock | Evidence in original native scored manifest |
| --- | --- | --- |
| Python | `3.12.13` | Not recorded |
| Torch | `2.12.0+git6bbd260` | Exact version recorded and matched |
| Transformers | `5.17.0` | Not recorded |
| PEFT | `0.21.0` | Exact version recorded and matched |
| safetensors | `0.8.0` | Not recorded |
| huggingface_hub | `1.31.0` | Not recorded |

The current lock was measured independently from the pinned parity runtime.
Its SHA-256 is
`438c6d105af900a229db1b77097fdfff9cb18cd7599edee25abf6373cbd3cefe`.
It does **not** prove that the four unrecorded package versions were used in
the historic scored run. The historic training provenance separately records
the same Torch version and HIP `7.2.53211`, but does not fill those four gaps.

## Gold-free native output comparison

The roster contains nine original prompt items and 27 Choice, Noul and Score
questions, including English, Chinese, Russian, a longer state and altered
Choice key order. No answer key or sealed evaluation label was loaded. Both
sides used the same immutable base, adapter, head, explicit criteria, CAL,
BF16 GPU forward path, one-item request batching and no truncation.

| Fixed parity gate | Observed |
| --- | ---: |
| Missing or invalid answers | `0 / 27` |
| Categorical mismatches | `0 / 27` |
| Maximum probability or Score drift | `0.0` (limit `1e-4`) |
| Source versus package answer digest | Identical: `7c72c5cfd51be4feb5de0ad11433e2b111a458220d16eae10694de55c47b4871` |

The private receipt says `passed: true`, and the parity process exited
successfully. This small synthetic roster tests only output preservation; it
does not estimate model accuracy, calibration, transfer or robustness.
An independent ordinary `import decision2` process also verified the staged
package, external source bytes, dependency lock, model identity and parameter
count after the GPU run.

## Required release integration

1. Keep the six scored inference modules as a versioned, immutable loader in
   the package. Add a reviewed external-base profile to the release packager
   and an arena adapter that imports **this package**, rather than silently
   replacing it with current development modules.
2. Re-evaluate the packaged model on every frozen, same-panel release split
   under a fully attested runtime lock. Do not inherit historic development
   scores merely from this 27-question parity result. Obtain historic
   dependency evidence if it exists; otherwise treat the new package run as
   the score-bearing baseline.
3. Complete source/rights review, authored sealed questions, overlap audit,
   multilingual review, candidate thresholds, model-card/charts inventory and
   downloaded-artifact verification before publishing. The package manifest
   remains `candidate-parity-pending` and no HF model or collection entry was
   created by this experiment.
