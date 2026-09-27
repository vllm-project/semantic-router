# Decision 2.0 27B BEST368: package-native development panels

Status: **three-panel development evaluation complete; release HOLD**. This
receipt supersedes any attempt to transfer the historic source-runtime scores
to the adapter-preserving package. The earlier 27-question gold-free parity
smoke passed, but the full paired panels below fail the stricter release
output-parity gate. No sealed FINAL labels, selector key, or Hugging Face
publication was used.

## Frozen model and execution

| Identity | SHA-256 or value |
| --- | --- |
| BEST368 text backbone, LoRA and decision-head fingerprint | `d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2` |
| Native CAL file | `e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78` |
| Clean sealed package `MODEL_MANIFEST.json` | `e6ca0f51bf7f27c938f35cbdbc6802d72b10a1d3d33405765fe491c32e25110b` |
| External immutable base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Package-native adapter | `decision2-peft-package-native-v1` |
| Signed collector source | `c47834ba15adf694a0c64812a4f26fd3baf4e51c` |
| Exact collector source archive | `35a7a34ce0ef8db100eaab6674009b971e3d65e1038baefffe982e755ea1f354` |
| Runtime image used for all three panels | `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54` |
| Private launch plan | `9de99a0c0f155f9b43427ef8c48e3e408179b43cb44d34659637151be64233f3` |
| Private score plan | `63d7a9f7886d47b0ef27a0963b525234c81a35892771f59bab18b6e1c8fcda62` |

Each panel used the same clean package, the same frozen per-type CAL, an
unchanged 4,096-token no-truncation limit, BF16 GPU backbone and FP32 head.
The signed local source archive was mirrored exactly into the runtime. Each
panel ran in a separate network-disabled container on an otherwise available
GPU; all three containers exited zero. Prompts contained only item ID, state
and native questions. The collector rejects explicit answer fields, preserves
question and option order, binds every row to input/model/package digests and
records invalid answers. Prediction and score files were private, and
predictions were scored separately with pinned scorer source. The manifest
reports **25,688,227,840 loaded text-inference parameters**, including the
external base text path, LoRA and custom head.

| Development panel | Prompt SHA-256 | Predictions SHA-256 | Score SHA-256 | Private score receipt SHA-256 | Scorer version and source SHA-256 |
| --- | --- | --- | --- | --- | --- |
| Typed DEV1,600 | `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` | `bbe2bcccb79380d385d74ca7ad3ac007d04cee77c281c0af8a0cdb8a7b5e2c12` | `eb903d873d7f0bb6f1180b1608c45e62a83a3854c9317185f28e749f1c034aa6` | `715ee8878c07bdc00d210d6bff9f13c18bc4d61a0736c91e447506d27f36f1c9` | `typed-decision-report/2`; `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc` |
| CSS pilot1,430, three tasks | `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda` | `e157b11f003cebc96877a2e183787ccbb32c0189a1ee7f0f3fdc633ce69005b8` | `f633d86d68d064bd2400c27c3cd9e2bd0e2c508de4ba5529aeb12eeed15ad5a6` | `14be030137428737998ce7740b7ef042b2e28211479548546ef75c3ee21ba304` | `css-transfer-score/2`; `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca` |
| Public 231-item subset | `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd` | `a2c9bbcaf5dd38eeb8b38e4647b114fc754f0c0f33d7df870c861904275694bc` | `689f0e344aa340d479d721b8c0a9d6d8e068ac886501eb081d9d9b37b39cdfde` | `2ef1d9c01d5f8927bd9674f726761706251764285cb2f1ae4627b612caaff19a` | `jevarena-jevbench-public-score/1`; `b0f3e77a705dde923e218d92841e333c04b37f8665e324a9760f691784e42557` |

These receipts are for **development and public-subset** panels. CSS pilot is
excluded from release transfer scoring. The public 231 is an independently
reproduced subset and is not an upstream closed-benchmark rank.

## Package-native results

| Panel | Package-native result | Native validity | Item latency p50 / p95 on this run |
| --- | --- | --- | --- |
| Typed DEV1,600 | **1,211/1,600 = 75.6875%**; Choice 791/800, Noul 258/400, Score 162/400 | 1,600/1,600; no truncation | 138.3 / 151.2 ms |
| CSS pilot1,430 | **845/1,430 = 59.09%** micro; median task macro-F1 0.62052 | 1,428/1,430; two over-budget answers counted as misses | 150.0 / 217.8 ms |
| Public subset231 | **200/231 = 86.58%**; easy 48/48, standard 69/72, hard 83/111 | 231/231; no truncation | 145.0 / 684.0 ms |

CSS task detail under the same scoring program: discourse **307/497**
(macro-F1 0.62052; two invalid), implicit hate **235/498** (macro-F1
0.47488), and SemEval stance **303/435** (macro-F1 0.68757). Typed DEV's
probability Brier is 0.19625 and ECE10 is 0.13790. The public subset's Brier
is 0.09549 and ECE15 is 0.03653. Latency is item-level inference time;
CSS/public scoring programs do not report throughput or cost. These latency
numbers should not be compared against other hardware or historical runs as
speed claims.

## Historical source versus current package

The old source predictions and current package predictions have identical
panel IDs, exact prompt bytes, the same 35-entry base/checkpoint fingerprint,
the same six scored inference-module file hashes, and the same CAL file. The
package additionally contains a versioned runtime API and manifest; its
adapter digest is therefore intentionally different. The fixed full-panel
parity gate requires zero categorical changes, p99 absolute probability/Score
drift at most 0.005 and maximum at most 0.02.

| Panel | Historic source score | Current package score | Categorical changes | p99 / maximum scalar drift | Private comparison receipt SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Typed DEV1,600 | 1,213/1,600 | 1,211/1,600 | 0 | 0.04978 / 0.19126 | `e8cc2a17918d10bd540f59de15023d83f94381da7d38aad603ee172b6908c688` |
| CSS pilot1,430 | 842/1,430 | 845/1,430 | 20 | 0.03296 / 0.20856 | `71341f21cb104b66268bf1cf4cf3faed91a14e3dc354386068f940834737bb66` |
| Public subset231 | 199/231 | 200/231 | 1 hard Score decision | 0.02499 / 0.06753 | `4fb57c60c3b088c0374fbb90ba7f1e4bbe08a7ac788f6fde7b8d8375f6b9cfc9` |

All three panels **fail** full-panel numeric parity. The 199→200 public
change and the CSS/DEV differences are runtime/package evaluation differences
of the same trained checkpoint, **not training gains**. Historical scores
must not be mixed with the new package-native scores or transferred into its
model card. The 27-question parity smoke was too small to detect this drift.

## Runtime provenance and bounded diagnosis

The original scored container was recovered read-only after the initial
parity note. Its exact image is
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
Reading package metadata inside that exact image now independently records
Python 3.12.13, Torch `2.12.0+git6bbd260`, Transformers 5.17.0, PEFT 0.21.0,
safetensors 0.8.0 and huggingface_hub 1.31.0. The current package image has
the same versions and matching SHA-256 for the inspected Qwen3.5 model,
PEFT LoRA layer, Torch functional and safetensors Python sources. The images
have different image identities and later layers; runs also used different
GPU instances on separate machines of the same device class and driver
version. Prompt and tokenizer bytes, option order, model weights, CAL,
Torch/PEFT versions, BF16/FP32 policy and selected performance environment
variables match. The historic source used explicit device synchronization
around forward; the package API omits those timing synchronizations. The
observed source-to-package drift is consistent with BF16 runtime or device
numerics, especially near decision ties, but a unique cause has **not** been
isolated. All 20 changed CSS choices had old top-two probability gaps below
0.054 (median 0.0099). No complete panel was rerun for this diagnosis.

The external-base release profile's CPU source/package contract matched the
same 35-entry checkpoint/base map and model SHA, and the earlier nine-item
native output parity smoke passed. Those checks do not override the failed
full-panel release parity. Formal assembled-bundle, rights, six-axis JevArena,
downloaded-artifact and HF-commit gates remain outstanding. The only current
pre-release model revision is the exact `package-sha256:` manifest digest;
claiming a future HF commit requires a separate downloaded-byte attestation.
