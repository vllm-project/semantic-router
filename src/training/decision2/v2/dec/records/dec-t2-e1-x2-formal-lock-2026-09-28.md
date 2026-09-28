# Formal lock: E1-X2 (4B, own Nox 1.0 + own-Lux soft targets)

Status: **locked before any post-key v3 or public231 prediction exists for this
candidate.** Scored once; no checkpoint, calibration or adapter change after
this commit. Results will be labeled **post-key same-panel** (the v3 labels
were accessed earlier in the project).

## Why it qualifies

E1 prereg (`03172ca28`) formal rule for 2B/4B: dev proxy ≥ same-runtime 1.0
control + 3.0, positive paired lower bound, all answers valid.

| Development readout (typed DEV + CSS pilot) | T | H | Proxy | Paired Δ (10,000 draws) |
| --- | ---: | ---: | ---: | --- |
| Nox 1.0, same node B runtime | .668125 | .412259 | 52.4825 | — |
| E1-X0 control | .661250 | .458472 | 55.0604 | +2.58 vs Nox 1.0, [+0.34, +4.62] |
| **E1-X2** | .671250 | .469012 | **56.1092** | +3.63 vs Nox 1.0, **[+1.48, +5.61]**; +1.05 vs X0, [−0.36, +2.51] |

Threshold 52.4825 + 3.0 = 55.48; X2 passes with 3,030/3,030 valid answers. X0
does not (55.06). The X2 − X0 factor effect is "suggestive" under C1 rules
(Δ ≥ +1.0, interval crosses 0), so the formal run tests a candidate, not a
confirmed factor.

## Frozen candidate identity

| Item | Value |
| --- | --- |
| Source | `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68dbaa557ea65dc458104d410a0802ee259` (4,208,383,488 decision parameters) |
| Training | E1-X2 full run, code `a4c6f0000`, image `sha256:ce895822…45f2fb`, node B GPU2, 466 updates, 2,552 s; provenance `0f081f13…9860c38` |
| Selected checkpoint | `checkpoint-0000408` (SELECT 635/700, family macro .897130; matrix v1 rule) |
| Adapter / head / config | `adapter_model.safetensors` `255dc4e3…1acc1a3cc`, `adapter_config.json` `c9ff755d…edac54`, `decision_head.safetensors` `cfc9c268…d045dcd0f`, `decision_config.json` `44a44ce0…d85436` |
| Inference identity | `model_sha256` `de41a699e3a34d93234015e5ae92794d192bf78b85b4d6f6dd2a8646c2e4d855` (source + adapter + head + tokenizer) |
| CAL calibration | `calibration.json` `71bab01e…6bb6890`; temperatures Choice 0.84166, Noul 0.75063, Score 0.45590 (fit once on CAL700) |

The Score temperature sharpens a CAL fitted only on five-level quantized-median
rows; its transport to v3 Score probabilities is a disclosed risk (it changes
no argmax).

## Frozen formal protocol

1. Collection with the eval track's frozen runner (`v2/eval/run_same_panel.sh`,
   `same_panel.py collect`) from an exact mirror of the commit that adds this
   lock, adapter spec `v2/dec/adapter-spec-infer-dec.json` (`v2.dec.infer_dec`,
   max length 8,192, the calibration above), panels `typed-final`
   (`e2a4a86b…`), `css15` (`7a527357…`) and `public231` (`642d3fac…`). Runs on
   node B GPU0 (track allocation) with image `ce895822…` from a hash-verified
   copy of the gold-free prompts only; no gold is present on node B.
2. `same_panel seal` (gold-free) on node B; the sealed run directory is copied
   byte-identically to node A, where `same_panel report` and `compare` read the
   frozen gold, against the eval track's adopted Nox 1.0 run
   (`/data/dev2/runs/eval/m1-adopt/nox1`: v3 56.470, public231 173).
3. Pre-stated reading: **qualified 4B first-release candidate** only if v3 gain
   over Nox 1.0 ≥ +3.0 with paired 95% lower bound > 0 and public231 ≥ 173;
   otherwise HOLD. Report Choice/Noul/Score, per-task CSS15, tiers, calibration,
   invalid counts and parameters either way. No HF upload in Milestone 1; no
   second 4B candidate is scored on v3 in Milestone 1.
