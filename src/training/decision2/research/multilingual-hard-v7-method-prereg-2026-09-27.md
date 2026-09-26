# Hard multilingual DEV v7: prospective editorial method

Status: method frozen before any v7 case, prompt, target, reviewer packet or
model inference is created. This follows the independently sealed r6
`BLOCK_FOR_INFERENCE` result. r6 remains unchanged and ineligible.

## Candidate boundary

Author a fresh, private DEV-only panel of at most 18 independent base cases,
with English, Chinese, Spanish and Japanese variants. Preserve balanced
Choice/Noul/Score coverage (six bases and 24 localized rows per type if all
18 pass preflight). Do not reuse r6 wording or relabel the ambiguous r6
question in place. Facts and source text belong in a separate private
casebook; the public code contains only rendering/verification logic. The
mechanical oracle must derive keys from structured facts. Gold-free prompts,
targets and a manifest with exact source/revision hashes are frozen before
any blind review or model response.

## Design repairs fixed in advance

1. Every temporal rule must state whether issue, expiry and revocation dates
   are inclusive **in all four languages**. Include at least one same-day
   expiry and one next-day revocation to test the boundary. The reviewer must
   be able to cite the exact localized rule and evidence for both.
2. At least one and at most two of the six Choice bases must genuinely
   resolve to HOLD because no candidate has all required evidence. No single
   A/B/C answer or native option position may occupy more than two of six
   bases. Option-order assignment must be frozen independently of model
   outputs. Do not add decorative failures solely to hit a quota.
3. For Score, use an unambiguous localized term for a mitigation **addition**
   in Chinese and explicitly distinguish it from subtracting a penalty.
   Include positive-credit and zero-credit cases, and paired facts that
   defeat mark-count or level-frequency shortcuts. Keep all numeric and
   ordinal thresholds consistent across languages.
4. Avoid a single repeated table voice. Vary evidence presentation only
   where the native typed adapter can faithfully represent it, and require
   every material record to affect the decision under a counterfactual
   deletion or substitution. Check a shallow heuristic suite before blind
   review: never-HOLD, fixed-label, newest-record, largest-number, mark-sum
   and keyword-only baselines.

## Acceptance sequence

First run structural/oracle tests, native parser preflight, and exact/near
source checks against TRAIN, SELECT, CAL and visible development references.
An available gold-free sealed source-ID denylist may be checked without
opening FINAL. These screens do not prove semantic non-overlap.

Next obtain genuinely independent gold-blind reviews. A separate qualified
bilingual or native reviewer is required for each non-English language;
a non-native multilingual-only review is insufficient for this gate. Each
reviewer must mark answer, supporting rule, naturalness, ambiguity and
translation fidelity. Seal all row and 18-base group judgments and hashes
before comparing with the private oracle. Any material ambiguity, wrong
localization, unavailable answer, or dominant shallow shortcut blocks the
entire frozen candidate; the author may design a later version but may not
edit the failed packet.

Only a clean editorial pass permits a separately preregistered native model
diagnostic. This small DEV slice can identify weaknesses but cannot enter
the release JevArena total, select a final checkpoint, justify training on
evaluation rows, or support a broad multilingual claim. No v7 casebook or
model result exists at this preregistration.
