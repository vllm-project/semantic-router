# Decoder Milestone 8-small — amendment 2: the 4B M8 Noul and Score floors (2026-09-30)

Written ≈10:10Z (18:10 UTC+8), before any line point was built or read and before any D-arm training job (the D
chains were still waiting for the A20r labels). Preregistration [`dec-m8s-prereg-2026-09-30.md`](dec-m8s-prereg-2026-09-30.md)
(`c99cf9b7f`), amendment 1 (`ada76c46e`). Development readouts so far: the references `2b-I` / `08b-I` (typed DEV,
CSS pilot, HT-DEV v2) and the control seeds' early HT-DEV v2 collections; no candidate point.

**Reason.** The 4B M8 preregistration (`a629a6ce2`, written ≈17:45 UTC+8, after this milestone's freeze at ≈17:30)
defines the "Score and Noul floors" of the assignment concretely, and the assignment asks to mirror its design where
sensible. Its λ (1.0) already equals this milestone's.

**Change (gates only):** a point must additionally pass

1. **Noul floor:** typed-DEV `rule_precedence` c ≥ c_I − 0.01·n (the 4B rule; stricter than the −0.03·n type floor,
   which stays);
2. **Score floor:** Score5-typed-DEV (800 prompts, `8e35bfff…`) read at 16K on the same path as every point; its
   **check half has no COLLAPSE, and no WARN unless I's check half has WARN** (the eval's `dev_readout` block,
   scored on node A where the gold is; `m8s-relay.sh s5`).

Everything else is unchanged and kept as preregistered, with the differences from the 4B design disclosed:

- **Starts:** the released soup itself (the assignment's "top-ups of the released weights"), not each soup member.
- **Human rows (D2):** this milestone's frozen source partition (public human-labeled sources, including QA / evidence
  rows) with a 1:1 human / typed token split; the 4B uses the narrower 9B M6 "human-rated" rule on a proportional
  slice. The TRAIN files are locked (part 1) and the control seeds have trained on them.
- **Line:** α ∈ {1, ½, ⅓} (4B: {1, ⅔, ⅓}).
- **Early rule:** HT-DEV v2 vs C-s1 not FLAG plus SELECT700 within 0.03 (4B: SELECT700 within 0.02 only).
- **Teacher runtime:** the pinned kernel-path logit collector of A20r's CAL fit (parity bit-exact on CAL698); the 4B
  converts rows to prompts for the formal collector (parity on 80 typed-final prompts). Both are A20r's scored
  runtime at T = 1.

Code: `m8s_rules.py` (`floors` Noul rule, `score_floor`, `score5t` subcommand), `m8s-lines.sh` (Score5 readout),
`m8s-relay.sh s5`; 12 tests pass.
