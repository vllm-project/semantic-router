# Lux 1.0 current-package JevArena v3 control: prospective lock

**Status before collection: LOCKED, not scored.** This is one own-1.0 control,
not a Decision 2.0 candidate. Earlier attempts produced no typed FINAL or
CSS15 predictions. The project has already opened these labels for unrelated
arms, so any result is explicitly a **post-key same-panel** comparison, never
an unopened blind test. This lock precedes the present control's predictions.

## Why this is a new protocol

The earlier Lux v3 attempt stopped when its current package differed by
`0.021284254` from a published Score probability, above the then-frozen
`0.020000000` ceiling. The old check remains a failed check; it is not silently
relaxed. A separate signed provenance audit established that the published
example was carried over byte-for-byte from an earlier Lux bundle and was not
bound by the current bundle's release/materials manifest. Its *requests* remain
fixed gold-free inputs, while its old recorded probabilities are not a valid
same-weight numerical reference for the current package. The prospective
current-bundle repeatability gate was signed before use and passed: two fresh
native processes, five Choice/Noul/Score answer slots, zero categorical or
numeric drift (limit `1e-6`), exact package revision and qualified runtime.
Its receipt SHA-256 is
`be088b2f19904dd8ead5017902bb794fdd0fb832c79cec4a29a61d7567f7e63f`.
No formal question or key was used in that gate. We reuse that completed
technical evidence rather than spending another GPU probe on the same control.

## Fixed identity and protocol

| Field | Locked value |
| --- | --- |
| Weight package | Own `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`, 7,940,895,744 deployed parameters. Current bundle SHA-256 `985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`; release manifest SHA-256 `f786ce80a159c716f22b7d9c652af765b06f9db083bf43a8b08e67c3be3e69c5`. Its manifest-listed files and local HF revision metadata must verify before use. |
| Native path | Published `DecisionModel.decide` through unchanged `inference/run.py` SHA-256 `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`; backend `lux`, exact input state/questions, no chat conversion, truncation, option removal or runtime override. Native source package must resolve the bundled FLA path. |
| Runtime | Previously qualified offline ROCm image `sha256:ce895822fc48bb6864911d4488a3946f3a18fd3dd2ec90c8a0a49b259145f2fb`; one live-confirmed idle GPU. Native output must report `runtime_matches_validated=true` and no runtime differences. |
| Typed FINAL | 1,600 original items, 2,000 answer slots; gold-free input SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`. |
| CSS15 | 6,547 original items and answer slots; gold-free input SHA-256 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`. |
| Prediction seal | Existing `jev_arena/seal_lux9b_peer.py` SHA-256 `ddc526425b6377f15ea37117c95a9315c7cb5b27a80eedeed5c8571ef2753`, plus a joint pre-score hash of both full prediction files and both seals. This sealer checks IDs, question counts, package revision and runtime. |
| Scoring | Fixed typed scorer SHA-256 `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`; CSS scorer SHA-256 `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`. Four-family typed macro accuracy `T`, median of 15 task macro-F1 scores `H`, total `100 × sqrt(T × H)`. All type/task scores and invalid slots are reported. |
| Optional same-panel peer | Already sealed JPT-9B v3, score 60.994, only if its exact frozen predictions are locally accessible with matching panel and scorer identities. Paired bootstrap: 5,000 draws, seed `20260927`, unchanged `jev_arena/compare_v3.py`. Otherwise report no paired interval. |
| Cost cap | One GPU, at most 0.5 GPU-hour / 30 minutes elapsed including model load and both collectors; no retrial, alternate checkpoint, input, image, adapter, threshold or calibration. |

## Stop and disclosure rules

1. Before GPU use, verify fresh inventory and per-card ownership/memory,
   source and image SHA, current package revision/all manifest files, available
   disk, both prompt hashes and the earlier passing repeatability receipt.
   Keep gold files outside every inference container. If any gate fails, stop
   before model load and record the reason and zero GPU-hours.
2. Run the complete typed FINAL and CSS15 gold-free inputs once each in two
   native processes using the same qualified runtime and package. A missing,
   malformed or over-budget answer is wrong; do not drop rows. A crash, OOM,
   incomplete file or time cap ends the run as incomplete, preserving partial
   evidence without score substitution.
3. Validate all 8,147 original IDs and 8,547 answer slots and seal both
   prediction files. Record hashes and UTC time in a joint gold-free freeze
   before opening either formal key. Then score once with the frozen scripts.
   This v3 result is a disclosed post-key same-panel control, not independent
   validation and not a Decision 2.0 improvement.
4. An older Lux public JevBench number is not reused as this run's public-231
   comparator without exact model revision, prompt, adapter and scorer hashes.
   Public JevBench is outside the v3 core and outside this GPU budget. No
   model publication or HF mutation occurs here.

Private run receipts retain raw prompts, predictions, logs, GPU timings and
host paths. Repository notes and the unified gist receive only aggregate
outcomes and non-sensitive digests.
