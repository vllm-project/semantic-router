# C1 event-3 card pass: six card-only revisions, and the `mirror_to_node.sh` race fix (2026-09-29/30)

Release worker (worktree `vllm-sr-dev2-release`, branch `xunzhuo/decision-2-training-release`), coordinator note
2026-09-29 23:40 UTC+8. Receipts are under [`dev2-c1-card-pass-2026-09-29/`](dev2-c1-card-pass-2026-09-29/), one folder
per size (`receipts/`, `extra/`, `logs/`, `package-text/`); the scripts that produced them are in its `ops/`.
Everything stays private.

## Result

| | New `main` | Manifest | Final decision (supersedes) | gate.json | Weights |
| --- | --- | --- | --- | --- | --- |
| DEV2.0-0.6B | `188eb4c822e8643034f8ca87d865a16a86f69952` | `42c31b04…` | [`453b4b7b…`](dev2-c1-card-pass-2026-09-29/DEV2.0-0.6B.decision.json) (`3fbe847b…`) | `bacf1e1f…`, 12 / 12 (successor profile) | 2 files, byte-identical to `b2131337` |
| DEV2.0-0.8B | `d4812ac6bb07333ec60d66091f08e1557933aa30` | `b568a92a…` | [`a97e063f…`](dev2-c1-card-pass-2026-09-29/DEV2.0-0.8B.decision.json) (`dd397e04…`) | `f8e25445…`, 6 / 6 | 2 files, byte-identical to `f458c34c` |
| DEV2.0-2B | `a47bdf895d982896b0e65c2bdc61330fe1ad2d7c` | `1ba35950…` | [`743c5b2e…`](dev2-c1-card-pass-2026-09-29/DEV2.0-2B.decision.json) (`de59a6c7…`) | `beca4c82…`, 6 / 6 | 3 files, byte-identical to `5ad3e9a3` |
| DEV2.0-4B | `197b70ca90759aca48410a9f94bb534e8ae2919f` | `8d3d618d…` | [`4d31cdab…`](dev2-c1-card-pass-2026-09-29/DEV2.0-4B.decision.json) (`bfd45785…`) | `5e62220a…`, 6 / 6 | 6 files, byte-identical to `452f1332` |
| DEV2.0-9B | `e51f9881b92f646cb0bd62b2876d4878cc8d16ec` | `e0199346…` | [`e833608c…`](dev2-c1-card-pass-2026-09-29/DEV2.0-9B.decision.json) (`bad97635…`) | `09a04c70…`, 6 / 6 | 10 files, byte-identical to `ae683196` |
| DEV2.0-27B | `c0dba600087f582a1033830a2276d22e96e6324c` | `0803580b…` | [`fd1b5ef0…`](dev2-c1-card-pass-2026-09-29/DEV2.0-27B.decision.json) (`76abd8e9…`) | `e6965d75…`, 6 / 6 (no-1.0 profile) | 2 files, byte-identical to `32e7e8b1` |

- **Every repository changed exactly `README.md` and `MODEL_MANIFEST.json`** against the revision it replaces (Hub
  revision diff, `extra/revision-diff.json`); every weight file keeps its size and LFS SHA-256.
- **Collection** "🎲 Decision 2.0" (the user's title, private): 0.6B, 0.8B, 2B, 4B, 9B, 27B. No item moved, nothing was
  added or removed (`extra/collection-order.json` after every run).
- **Storage:** 61.20 → 61.28 GB used of 100 GB. The +0.08 GB is another track's upload to the private training-data
  dataset during the pass (3.23 → 3.31 GB); the six repositories' usage is unchanged, since card-only revisions share
  every weight blob. Nothing was deleted.
- **GPU-hours:** 0.811 on node A GPU0 and GPU1 under the shared lease `owner.release-c1card` (details in §6).

## 1. Card text

Sources: the event-3 record `v2/eval/records/m4-c1-event3-prep-2026-09-29.md` §0000 (results, card lines, paired
differences and the regressions to disclose) and the JevBench record `v2/eval/records/jevbench-value-2026-09-29.md`
§§4 and 6. Each confirmation line is the event-3 record's verbatim line plus one plain reading sentence. The derivation
is [`ops/make_specs.py`](dev2-c1-card-pass-2026-09-29/ops/make_specs.py) (`--check` reproduces the specs byte for
byte); it asserts every replaced text before replacing it.

- **All six:** the note under the public-231 rank chart is the renderer's 17:15 wording ("public-only rerun (about a
  third of the official Intelligence inputs), not the official JevBench score; easy tier at ceiling; totals within
  about 10 items are not distinguishable"). It is the only change on the DEV2.0-0.8B card; DEV2.0-0.6B already had it.
- **DEV2.0-0.6B:** the confirmation now reads "Independent sealed confirmation, same input limit (previous
  revision)": the previous weights score 33.02 vs Decision 1.0 Kai at 8,192 tokens 22.14 (+10.89, 95% CI [+8.88,
  +12.64]). These are the event-2 predictions of revision `e61b2b44` (model files identical to `99c4e799`), rescored
  on v1.2; Kai at 1,024 tokens scores 17.79. "The current weights (released as revision `b2131337`) were not part of
  any sealed event."
- **DEV2.0-0.8B:** no new C1 line; the event-1 line (v1.1) stands.
- **DEV2.0-2B:** 45.70 vs Sol 1.0 16K 45.03 (+0.68 [−0.91, +2.22]), Decider 2B 42.45 (+3.26 [+1.32, +5.21]),
  This-That 1.2 42.81 (+2.90 [+0.85, +4.99]). "It is level with Decision 1.0 Sol and significantly above Decider 2B
  and This-That 1.2."
- **DEV2.0-4B:** 48.38 vs Nox 1.0 49.70 (−1.32 [−3.00, +0.35]), Decider 4B 49.65 (−1.27 [−3.05, +0.49]), Jet v6.2
  50.44 (−2.06 [−3.86, −0.18]). "It is level with Decision 1.0 Nox and Decider 4B and significantly below Jet v6.2: the
  post-key JevArena v3 lead over Decision 1.0 Nox does not carry over to C1."
  - New disclosure **JevArena-C1 (independent, human-labeled)**: the v3 lead over Nox (+6.68) does not carry over
    (−1.32 [−3.00, +0.35]) and C1 Noul is significantly lower (−3.21 [−6.28, −0.09]). Jet v6.2 is significantly ahead
    (−2.06 [−3.86, −0.18]), mainly on Score (−6.03 [−9.70, −2.05]), while this model leads it on Noul (+4.16
    [+0.41, +7.92]); Decider 4B is level. It lists the C1 tasks and languages below Nox 1.0 and below Jet v6.2, as the
    event-3 record gives them.
  - JevBench: the Decider 4B gap (171 vs 192, −21 [−32, −10]; hard 56 vs 73) now names the two hard-tier skills that
    neither JevArena nor our training covers: long policy documents with amendments and precedence (3 vs 10 of 19) and
    checking a quoted person's plausible but wrong conclusion (18 vs 30 of 46). Nox 1.0 has the same weakness (5 and
    22). The Nox 1.0 comparison gains its interval (−2 [−9, +5]).
- **DEV2.0-9B:** 53.77 vs Lux 1.0 51.97 (+1.80 [+0.55, +3.07]), Nimble v2 52.77 (+1.00 [−0.69, +2.75]), measured on
  the frozen package `53bac735` (weights identical). "It is significantly above Decision 1.0 Lux and level with Nimble
  v2." The card already had its within-noise JevBench line (178 vs 183, −5 [−11, +1]).
- **DEV2.0-27B:** 57.33 vs AutoJev-27B 58.17 (−0.84 [−2.50, +0.69]), Eikos-27B 59.37 (−2.04 [−3.66, −0.44]). "It is
  level with AutoJev-27B and significantly below Eikos-27B."
  - New disclosure **JevArena-C1**: level with AutoJev-27B although post-key v3 is significantly lower; Eikos-27B is
    significantly ahead, mainly on Choice (−3.01 [−5.35, −0.75]); the C1 tasks and languages below Eikos-27B.
  - JevBench: the Eikos gap (198 vs 212, −14 [−22, −6]; hard 79 vs 92) now names the same two skills (long policy
    documents 10 vs 15 of 19, quoted conclusions 29 vs 38 of 46).
- **Which regressions are listed.** Task and language lists are added only where the event-3 record shows a
  significant loss against a card peer: 4B vs Nox 1.0 (Noul) and vs Jet v6.2 (C1, Score), and 27B vs Eikos-27B (C1,
  Choice). 2B, 9B and 0.6B have no significant C1 loss against a card peer. The record's Russian figures for 27B (a
  strength) were left off: the record does not say which peer each belongs to, and the paired files have no
  per-language breakdown.

## 2. Specs, decisions and the builder change

- **Specs** `specs/dev2-{0p6b,0p8b,2b,4b,9b,27b}-card-c1.json`, derived from the current release specs. Only the gate
  receipt, a `_release.card_c1` note, the card text above and the source pins below change; checkpoints, identities,
  scored runs, runtime requirements, calibration and licences are copied. The released specs stay as they were, since
  their build receipts bind their SHA-256.
- **Final decisions** ([`ops/make_decisions.py`](dev2-c1-card-pass-2026-09-29/ops/make_decisions.py), `--check`).
  - Each copies the superseded final decision's judgement: identity, scored report, paired comparison, gate profile
    and evidence (0.6B successor: `current_revision` `99c4e799` and the eight evidence hashes; 27B no-1.0: the type
    check), calibration and licence decision.
  - Each adds the action, the C1 treatment (`c1`), the new disclosures, `card_revision.spec_sha256` and `supersedes`
    (the old final, the released revision, manifest and gate.json).
  - Decided by the coordinator (parent agent) under the 23:40 note; `decided_utc` 2026-09-29T15:40:00Z.
  - For 0.6B the successor profile keeps `99c4e799` as its comparison, because the R1–R7 evidence is against that
    run, so gate.json `supersedes` names `99c4e799`; the decision's own `supersedes` names `b2131337`, which this
    revision replaces.
- **Builder change: `runtime_source`** (`49bb605d4`, `v2/release/build.py`, tests in
  `v2/release/tests/test_vendor_source.py`).
  - The first previews found that the package runtime `decision2/qwen.py` always came from the building tree. That
    file gained Score-offset support in `c6e993f48`, after the 0.8B–27B releases, so today's tree would have shipped
    a changed runtime file in five repositories.
  - Three previews (0.8B, 2B, 4B) also stopped at the builder's scored-runtime check, because `training/model/infer.py`
    changed after they were scored and their specs did not pin `vendor_source`.
  - The new optional key names the exact mirror whose `v2/release/runtime` supplies `decision2/*.py`, as
    `vendor_source` does for `training/model`. Both default to the builder's tree, so existing specs build unchanged.
    `MODEL_MANIFEST.json` `runtime` records it.
  - Pins: 0.8B / 2B `33de83cea` (runtime and vendored sources), 4B `f8f52c695` (both), 9B / 27B `2926952c1` (runtime;
    their existing `vendor_source` stays). Each is the mirror that built the revision being replaced, which was
    checked against the released manifests' file hashes. 0.6B was built from a tree with today's runtime.
  - Release suite: 112 OK, 2 skipped.

## 3. Verification

`ops/card.sh <size> --gpu <0|1>` runs `release.sh --upload --collect --already-collected` from mirror `0a50a86b8` with
a fresh copy of the scored run's frozen autotune cache (its digest is checked first; every frozen cache kept its
digest). The post-checks are a Hub revision diff against the replaced revision, the collection order, the card HTTP
check, the card links, `gate evaluate` and storage. CPU previews built every card first and fetched the released
card text: against each released manifest only `README.md` changed, apart from `MODEL_MANIFEST.json`.

| | Parity, pre and post upload (0 changed answers) | Max drift | Download + re-hash | Loaded parameters | Card HTTP | Links |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6B | typed 200, CSS15 300, public 231 100, mlx-diag 100 (tolerance 0) | 0.0 | 33 files | 597,103,104 | 12 / 12 | 14 / 14 |
| 0.8B | 150 / 200 / 100 / 150 | 4.4e-16 | 30 files | 753,446,208 | 12 / 12 | 14 / 14 |
| 2B | 150 / 200 / 100 / 150 | 4.4e-16 | 33 files | 1,883,930,944 | 12 / 12 | 15 / 15 |
| 4B | 200 / 300 / 100 (no mlx-diag, as at release) | 1.8e-14 | 36 files | 4,208,383,488 | 13 / 13 | 17 / 17 |
| 9B | 200 / 300 / 100 / 100 | 4.4e-16 | 40 files | 7,940,895,744 | 12 / 12 | 15 / 15 |
| 27B | 200 / 300 / 100 | 0.0 | 31 files | 25,746,591,744 | 11 / 11 | 12 / 12 |

- Every run: examples bit-identical across processes before and after download, the card's Python block reproduced
  from the download, readback private with no card problems, gate sealed, collected readback passed.
- Anonymous requests to every repository's API, page and README return 401.

## 4. DEV2.0-27B on node A

The 27B spec names node-B paths, while this pass runs on node A GPU0 / GPU1. The inputs were therefore replicated to
the same paths on node A ([receipt](dev2-c1-card-pass-2026-09-29/receipts/nodeA-27b-replica.json)); the eval type
check records its run by path, so the paths had to be the same.

- **Run inputs:** the checkpoint, the formal run (report, seal, paired files, predictions, frozen cache
  `03b172f1…`), the three peer reports and the release inputs: 2,729 files. 2,699 were copied from node-A files with
  the same SHA-256 (the C1 event-3 and JevBench-analysis copies) and 30 small files were relayed from node B with
  `tar -k`. The sorted SHA-256 list on node A equals node B's byte for byte (`a2e95d72…`).
- **Base:** `Qwen/Qwen3.8-27B@1d4bf0f2` in `/data/dev2/hf-cache`, copied from node A's older cache (same revision) plus
  4 small files from the plain copy. All 28 pinned files and the pinned LICENSE match. The cache's `trees/<revision>.json`
  listing was relayed from node B after the first attempt (below).
- **Vendored sources:** mirror `35fa052d…` created on node A by the hardened mirror script.
- **First attempt failed before any upload** (17:04:49–17:10:17Z): the card-example step runs the README's Python
  block, which resolves the base with `snapshot_download` inside the offline container. huggingface_hub 1.31 needs the
  cached `trees/<revision>.json` for that, and node A's copy lacked it. After it was added, an offline check in the
  same image resolved all 28 files, and the retry (17:15:53Z–17:34:08Z) passed.
- Parity reproduced node B's scored predictions on node A (drift 0.0), as the C1 event-3 preflight had.

## 5. `mirror_to_node.sh`: per-mirror lock, verify or refuse, no nesting (`03224df4f`)

The race (0.6B release, 2026-09-29 18:52): two runs both saw the target absent, and the second `mv` moved its
staging copy inside the finished mirror. Root ignores the read-only bits, so nothing stopped it.

- **Lock:** the node-side step that creates or adopts a mirror holds an exclusive `flock` on
  `<target-root>/.locks/<mirror>.lock` (`DEV2_MIRROR_LOCK_TIMEOUT`, default 900 s; timeout exits 75 and removes the
  upload).
- **Existing target:** under the lock the target is re-checked. A finished mirror is reused, and an identical plain
  extraction is adopted. Anything else fails with "refusing to touch it" and is left untouched.
- **Every run** then checks the receipt (commit and git tree) and the content manifest recomputed over every file on
  the node, and refuses a mirror that holds nested `.incoming-*` entries.
- **Atomic placement:** a private `mktemp` staging directory next to the target is renamed with `mv -T`, which fails
  rather than moving into an existing directory. A mirror found already present skips the upload.
- The output line gains `action=created|reused|adopted|verified`.
- **Tests:** [`v2/common/tests/test_mirror_to_node.py`](../../common/tests/test_mirror_to_node.py), 8 tests against a
  local fake node: a six-way race (one created, five reused), a lost race, refusal of a differing or nested target,
  adoption, lock timeout and subtree mirrors. The race test fails on the previous script at exactly the nesting `mv`.
- **On node A:** three concurrent mirrors of `03224df4f` into a scratch root gave one `created`, two `reused` and no
  nested entry; a re-run gave `verified`; the scratch root was removed. This pass's own mirrors (`35fa052d`,
  `e933f088e`, `0a50a86b8`) used it.
- The interim rule of the 22:05 note ("verify and reuse an existing `/data/dev2/src/<SRC>`") is now what the script
  does; the coordinator can retire it once every track has merged `03224df4f`.

## 6. Incidents, GPU-hours, storage

- **Duplicate relay stream.** The first attempt to stream the 27B inputs from node B through the workstation (about
  50 KB/s) was running twice at once: the tool relaunched it. Both were stopped. Their partial tree (under
  `runs/27b/M3-A-soup`, created by that stream at 16:03:51Z and absent before) was removed before the replica in §4.
  No other track's files were touched.
- **Stale preview decisions.** The first previews wrote the then-current decisions into node A's decisions folder;
  after the source pins changed five of them, those six unsealed copies were removed and rewritten by the runs.
- **GPU-hours** (lease window × 1 GPU): GPU0 0.6B 192 s, 0.8B 195 s, 2B 246 s, 4B 390 s; GPU1 9B 473 s, 27B first
  attempt 328 s, 27B 1,095 s. Total **0.811 GPU-h**. Previews, the replica, the Hub readbacks and the mirror checks
  were CPU only. Both GPUs were idle when used (0%, 274 GB free); the GPU1 data co-tenant entries were `released`.
- **Storage:** 61.20 → 61.28 GB of 100 (`hf_headroom.sh`; see Result).

## 7. Commits

`03224df4f` mirror fix and tests · `e933f088e` specs, decisions, ops, replica receipt · `49bb605d4` builder
`runtime_source` and tests · `0a50a86b8` source pins and the preview check · this record, the run receipts and the
release-record updates, merged into `xunzhuo/decision-2-training` (merge-only).
