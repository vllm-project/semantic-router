# IB3 preregistration — amendment 2: G4 drops the two grounding families; one passage-swap redesign (2026-10-01)

Committed **after G4 on run `d1` (pass 3) and before any screen or review sample was drawn**; no IB3 row has been shown
to a reviewer. The prereg (`b30bcae5c`) and amendment 1 (`bcaaa70f7`) are unchanged except where this amendment says
so. Only G4 receipts and counts were read. The prereg said a failing family is dropped with no redesign; as IB2's
amendment 2 did for its priority families, this amendment allows **one** redesign for the grounding family, because
grounding is a priority family of this job and no other licence-clean grounding source is left. The gate is unchanged.

## A. What G4 found on `d1` pass 3 (accuracy vs cross-validated majority; gate = majority + 0.05)

| Family | Rows | State removed | Option only | Hypothesis only | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `wpd` | 3,708 | .494 vs .485 | .492 vs .485 | — | PASS (shape-cell baseline .432) |
| `phiu` | 1,730 | .496 vs .475 | .491 vs .475 | — | PASS (shape-cell baseline .466) |
| `esci` | 9,986 | .500 vs .500 | .500 vs .500 | product .515 vs .500 | PASS |
| `mqa` | 8,762 | .500 vs .500 | .500 vs .500 | proposed answer .543 vs .500 | PASS |
| `fdial` | 2,546 | .479 vs .477 | .484 vs .477 | **response .743 vs .477** | **FAIL** |
| `haluqa` | 554 | .500 vs .482 | .495 vs .482 | **answer .601 vs .482** | **FAIL** |

Both grounding families leak through the response / answer alone even after the first-person, length and form cells:
FaithDial's unsupported responses are chit-chat in a style its supported responses lack, and HaluEval's hallucinated
answers differ in form from the HotpotQA spans. `haluqa` leaves IB3 (its unsupported answers are exactly the leaking
ones, so no construction from this file holds the answer fixed).

## B. One redesign: `fdial2` (passage-swap twins; the response is held fixed)

- Base: FaithDial train utterances whose BEGIN set is exactly {Entailment}; response as in `fdial` (the labelled
  original response, or `response` when `original_response` is null).
- **yes** = {knowledge K_i, previous_turn, response R_i} (the knowledge the response was judged entailed by);
  **no** = {knowledge K_j, previous_turn, response R_i}, where K_j is the knowledge of another wizard turn of the
  **same dialogue** whose normalized text differs from K_i (hash order `ib3-fdial2-other-v1:<key>`); an utterance with
  no such K_j is dropped. Same question as `fdial`.
- So **every response appears once as yes and once as no**: the response alone (G4 hypothesis view) is at chance by
  construction, and the decision needs the passage. "No" is checkable: R_i is supported by K_i, and K_j is a different
  sentence of the dialogue's topic.
- Cell = the utterance (twins); group = dialogue; ≤ 2 utterances (4 rows) per dialogue (hash order); cap 6,000 rows.
- G4 gates `fdial2` like every other family (hypothesis field `response`); a failing `fdial2` is dropped with no
  further attempt.

## C. Run `d2`

Run **`d2`** builds the five families `wpd`, `phiu`, `fdial2`, `esci`, `mqa` from the same pinned raw files (`maud`
left by amendment 1; `fdial`, `haluqa` left here; the four unchanged families are produced by unchanged code), and
**every audit runs again on `d2` from the start** (G0 and G0u with fresh positive controls, G1, G2, G3, pass 1,
re-scans, the sample pass, G4 on every family, G5–G8, leak guard). The review sample is drawn from `d2` only; with
F = 5 surviving families stage R takes `max(18, ceil(216 / 5))` = 44 rows per family (220). `d1` stays on the node as
the record of amendments 1 and 2 and is not published.
