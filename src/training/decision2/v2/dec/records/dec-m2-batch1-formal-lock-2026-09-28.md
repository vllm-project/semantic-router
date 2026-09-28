# Formal lock: M2 batch 1 — B8F-s1, E8F-s1 (0.8B) and X4K-s1, X4K-s2 (4B)

Status: **locked before any post-key v3 or public231 prediction exists for
these four checkpoints.** Each is collected once; no checkpoint, calibration
or adapter change after this commit. Results are labeled **post-key
same-panel**. Qualification and order: prereg `143cfc214`, amendments 1
(`f316bad37`) and 2 (`181ec58a0`); development readouts are in amendment 2.

## Frozen candidate identities

All four were uploaded from node B to the private staging repo
`llm-semantic-router/dev2-dec-staging` at commit
`5421582dfed51dd3b19b5686128ed75e0f8525e3` and downloaded on node A; all 37
files match node B's SHA-256 list.

| Candidate | Start / mode | Selected checkpoint (SELECT) | `model_sha256` | CAL temperatures (Choice / Noul / Score) |
| --- | --- | --- | --- | --- |
| B8F-s1 | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2`, full fine-tuning on the full mixture `d1dc33fc…` | `checkpoint-0002030` (621/700, .868056) | `196f97bec42784b321ecfc492909a347c17a6041bf12ab5ac9eed7a5cfe1236e` | 1.13926 / 1.09218 / 0.77568 |
| E8F-s1 | `Decision-1.0-Eos-0.8B@363c4a5e`, full fine-tuning on the full mixture | `checkpoint-0001776` (592/700, .819722) | `5359b701c4a351f0f5130b34aa757494c4648f571092eaccf0912900a1a2281a` | 1.00520 / 1.01111 / 1.05724 |
| X4K-s1 | `Decision-1.0-Nox-4B@cde2a68d`, LoRA, control mixture + own-Nox KL 0.5 | `checkpoint-0000315` (620/700, .875185) | `83692099c1c7a919a2c0bbd922edb242d23b8a77e33727db25992fd91b29199b` | 0.87405 / 0.91964 / 0.39182 |
| X4K-s2 | same, seed 20260927 | `checkpoint-0000266` (619/700, .864630) | `cd8bb99bda698f9f8f3f8f2fa4756f024bdc2737218b8f7b0cb30275f125906b` | 0.89068 / 0.85589 / 0.48939 |

## Frozen protocol

Same as the X4R lock (`0ac081347`): the eval track's frozen runner on node A
GPU5 (image `f83b1d10…`), adapter spec `v2/dec/adapter-spec-infer-dec.json`
(8,192 tokens, the calibration above; full checkpoints carry their own
weights, the source path is recorded only), typed-final + css15 + public231,
persisted per-run autotune cache, 20-item smoke first, from an exact mirror of
the commit that adds this lock. Order: B8F-s1, E8F-s1, X4K-s1, X4K-s2.
`seal` / `report` / `compare` on node A against the adopted 1.0 run (Eos 1.0
`/data/dev2/runs/eval/m1/r4-eos1`, v3 42.547; Nox 1.0 `m1-adopt/nox1`,
56.470) and the same-limit control (`eos1-8k` 42.361, `nox1-8k` 55.689).
Reading (prereg): s1 qualifies only if its v3 paired 95% lower bound vs the
adopted 1.0 run is > 0, its second seed's point estimate is also above the
1.0 (E8F-s2 / B8F-s2 when complete; X4K-s2 here), and no decision type
collapses.
