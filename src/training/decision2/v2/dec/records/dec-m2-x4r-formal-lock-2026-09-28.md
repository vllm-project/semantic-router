# Formal lock: M2 4B retention arm X4R, seeds s1 and s2

Status: **locked before any post-key v3 or public231 prediction exists for
either seed.** Each is collected once; no checkpoint, calibration or adapter
change after this commit. Results are labeled **post-key same-panel**.

## Why they qualify (M2 prereg `143cfc214`, 4B formal rule)

Both retention seeds pass the development screen (P ≥ same-runtime Nox 1.0,
no typed-DEV type more than 3.0 points below Nox 1.0, H ≥ H(Nox 1.0) − .015,
all answers valid), so both seeds take formal runs. Development readout
(typed DEV + CSS pilot; node B; the Nox 1.0 reference here is Milestone 1's
node B readout, the Milestone 2 same-image control is recorded with the results):

| Arm | T | H | P | Choice / Noul / Score |
| --- | ---: | ---: | ---: | --- |
| Nox 1.0 | .6681 | .4123 | 52.48 | 462 / 229 / 378 |
| X4R-s1 | .6725 | .4319 | 53.90 | 457 / 233 / 386 |
| X4R-s2 | .6725 | .4437 | 54.62 | 484 / 222 / 370 |
| X4C-s1 (control recipe) | .6638 | .4547 | 54.94 | 471 / 224 / 367 |
| X4C-s2 (control recipe) | .6606 | .4488 | 54.45 | 474 / 225 / 358 (fails the Score floor) |

## Frozen candidate identities

| Item | X4R-s1 | X4R-s2 |
| --- | --- | --- |
| Source | `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68d` | same |
| Training | retention mixture `37fb8856…`, own-Lux KL 0.5 on 6,430 rows, code `163d40dab`, image `dbe5f32b…`, node B GPU0 | same, node B GPU1 |
| Selected checkpoint (matrix v1) | `checkpoint-0000987` (SELECT 627/700, family macro .882778) | `checkpoint-0000989` (633/700, .895741) |
| Adapter / head | `adapter_model.safetensors` `5b4631b4…`, `decision_head.safetensors` `559db9b6…` | `4468bc58…`, `fde6ff83…` |
| Inference identity `model_sha256` | `eb9ab20eb571155e01f638260cc8924b713d55c44e96a6a39bbf12bd3f4c5a77` | `52a22cb59a0f6ce88ec97e84efc8daecf4b6fbd54b354d6754c8877520ad46ce` |
| CAL calibration (file) | `d4da8d8b…`; T Choice .85828, Noul .80341, Score .39618 | `e87b58d9…`; .83525, .71714, .35469 |

Copies on node A (`/data/dev2/runs/dec/m2/formal-candidates/`) are
byte-identical to node B (per-file SHA-256 manifests `ce4d8fc7…`, `222a3848…`).

## Frozen protocol

1. The eval track's frozen runner (`v2/eval/run_same_panel.sh`,
   `same_panel collect`) on node A GPU5 (image `f83b1d10…`), adapter spec
   `v2/dec/adapter-spec-infer-dec.json` (8,192 tokens, the calibration above),
   panels typed-final, css15, public231, a persisted per-run autotune cache,
   from an exact mirror of the commit that adds this lock; a 20-item gold-free
   smoke first. s1 then s2.
2. `seal`, `report`, `compare` on node A against the adopted Nox 1.0 run
   (`/data/dev2/runs/eval/m1-adopt/nox1`: v3 56.470, public231 173) and
   against the same-limit control `nox1-8k` (Nox 1.0 package through the
   shared renderer at 8,192 tokens with its own temperatures, collected
   2026-09-28 on node A GPU5).
3. Reading (prereg): s1 is the candidate; it qualifies only if its v3 paired
   95% lower bound vs Nox 1.0 is > 0, s2's v3 point estimate is also above
   Nox 1.0, and no decision type collapses. Otherwise HOLD. Choice / Noul /
   Score, per-task CSS15, public231 tiers, calibration and invalid counts are
   reported either way.
