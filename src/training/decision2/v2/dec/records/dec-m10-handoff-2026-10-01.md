# Decoder Milestone 10 — hand-offs for the 4B successor candidate `m10-4b-LH` (2026-10-01)

Results: [`dec-m10-results-2026-10-01.md`](dec-m10-results-2026-10-01.md). `m10-4b-LH` passes successor items 1–7
against DEV2.0-4B (post-key v3 67.34 vs 63.15, +4.19 [+0.10, +9.88]). Three steps remain, each owned by another
track; nothing has been uploaded and C1 has not been opened.

## The frozen candidate (node A; identical copy on node F)

| Field | Value |
| --- | --- |
| Package dir (M6 staging layout) | node A `/data/dev2/runs/dec/formal/m10/pkg/m10-4b-LH` (file list `pkg/m10-4b-LH.sha256`; revision = its SHA-256 `da0d982fda1ac2d7fb6bc84437159d7d0920d8d4aeb0f38397d15e6c1836360a`) |
| Checkpoint | `m6/m10-4b-LH/checkpoint`: full FP32 `qwen3.5-text-endpoints-global-query-shared-bilinear-mlp`, prompt `decision2-segmented-options-global-query-v1` (the released runtime's head path); uniform soup of the three LH seeds, each a rank-128 LoRA merged into Qwen3.5-4B-Base `1001bb4d…` |
| Identity (`dec_fingerprint`) | `5fa2a6998ede6290e1c741c71b44d145b7564b622b0c8e4561ef8f7f69c4c489` |
| Calibration | `m6/m10-4b-LH/cal698-16k/calibration.json`, adopted by the 23:15 rule (`calibration-decision.json`): T choice .621 / noul .479 / score .209 |
| Parameters | 4,208,383,488 loaded (the same count as DEV2.0-4B) |
| Scored formal run | node A `/data/dev2/runs/dec/formal/m10/m10-4b-LH` (seal `f015d4f0…`; typed FINAL predictions `1d9189b1…`, public 231 `8543f7b8…`, `COLLECT.json` `f6e50377…`); mlx-diag `m10-4b-LH-mlx`; successor evaluation `formal/m10/successor/4b-m10-4b-LH.{json,md}` |
| Formal runtime | `v2.dec.infer_dec` with `v2/dec/adapter-spec-infer-dec.json` (calibrated), 16,384 tokens, image `dbe5f32b` (kernel), node F GPU3 with an isolated render node; persisted autotune cache staged frozen on node A at `formal/m10/m10-4b-LH-cache-frozen` (1,654 files, manifest `569e86f5…`) |
| Training data | the released 4B mixture minus the M7 quarantine group (`c385406e…`, 58,739 rows; exposure receipt 0 groups); own-Lux 1.0 targets; no new data source |

## 1. Release engineering: a release-format package (local only)

The C1 runner and the release pipeline need a `MODEL_MANIFEST.json` package. Build it locally from the frozen
package with `v2.release.build` (profile `qwen-full`, `max_input_tokens` 16,384, the adopted calibration; a BF16 copy
only with `bf16_copy` exact parity), then hand the package dir and manifest SHA-256 to the eval custodian. **Card
facts that change versus DEV2.0-4B:** the weights now descend from Qwen3.5-4B-Base through a merged LoRA, not from
Decision 1.0 Nox. The training data, own-Lux teacher and runtime family are unchanged. Upload only after item 8
passes and the coordinator approves.

## 2. Eval custodian: successor item 8 (C1 v1.2 post-key)

`c1-postkey.sh collect --spec <spec>` with `role: successor`, tier 4B, against the registered baseline (DEV2.0-4B
`452f1332`). Spec fields (draft; the package fields come from step 1):

- `adapter_spec`: `v2/dec/adapter-spec-infer-dec.json` with `extra`: `model_id`, `max_length` 16384,
  `calibration` = the package's calibration file, `source` = any existing path (unused by full checkpoints; the
  event-3 4B row used the Nox snapshot).
- `image`: `kernel` (`dbe5f32b`); `cache.frozen`: `/data/dev2/runs/dec/formal/m10/m10-4b-LH-cache-frozen`.
- `parity`: `exact`, stored `/data/dev2/runs/dec/formal/m10/m10-4b-LH/output/typed-final.predictions.jsonl`
  (tolerance 1e-4); `stored_collect`: its `COLLECT.json`; `files`: typed FINAL, public 231, `COLLECT.json`.
- `identity`: `5fa2a699…` (the release package's manifest identity must equal it).
- Node A's own 4B C1 runs used image `host2` with the node-A N4XF cache. Whether the kernel image reproduces this
  node-F run exactly on node A (possibly with `HIP_FORCE_DEV_KERNARG=1`, as M9's node-A readouts did) is for
  `--preflight-only` to decide.
- No new training data source, so no C1 content recheck beyond the custodian's standard check.

## 3. IX1: Index run on the frozen finalist (private)

Request an IX1 run of the release-format package from step 1 (or of this frozen checkpoint through the 2.0 engine
adapter, if IX1 prefers to run it before packaging). IX1 can pull the package from node A to nodes C / D with node
A's transfer key. Results go only to `decision2-program/private/` and the node's private directory. This record and
every public artifact carry no Index numbers, and nothing in M10 was selected or tuned on Index rows. The retention
probes excluded every item with a 13-gram in the suite (checked on node C).

## Remaining M10 work (this track)

- NT2 (wave 2: the N4XF recipe on Nox with the label-token readout) is read and gated as soon as it finishes. It can
  take the second finalist slot only under amendment 2's order.
- IB1 arms on the LH recipe start when a release-safe IB1 record lands (IB1-r2).
