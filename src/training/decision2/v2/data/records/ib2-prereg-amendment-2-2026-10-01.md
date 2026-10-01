# IB2 preregistration — amendment 2: G4 drops on run `c2`, one frequency-matched redesign each (2026-10-01)

Committed **after G4 on run `c2` (pass 2) and before any screen or review sample was drawn**; no IB2 row has been
shown to a reviewer. The prereg (`fff14438e`) and amendment 1 (`0a2c8a2b9`) are unchanged except where this amendment
says so.

## A. What G4 found on `c2` (pass 2 TRAIN; accuracy vs cross-validated majority, gate = majority + 0.05)

| Family | Rows | Failing view | Accuracy | Majority | Verdict |
| --- | ---: | --- | ---: | ---: | --- |
| `fc_rel` | 5,642 | — | .497 | .496 | PASS |
| `fc_ready` | 2,578 | — | .518 | .481 | PASS |
| `ytspam` | 1,456 | — | .501 | .483 | PASS |
| `argq` | 5,152 | — | .498 | .492 | PASS |
| `hover` | 4,020 | — | .600 | .600 | PASS |
| `fc_sel` | 5,901 | option only | .874 | .246 | **FAIL** |
| `fc_args` | 4,552 | option only | .718 | .325 | **FAIL** |
| `gsm` | 5,438 | hypothesis only (the stated answer) | .564 | .493 | **FAIL** |
| `qasc` | 5,268 | option only | .321 | .126 | **FAIL** |
| `arc` | 2,839 | option only | .313 | .250 | **FAIL** |

`cnli` has **no row left** after G0 and G2: ContractNLI's NDAs share boilerplate 13-grams with Index or panel rows, so
every remaining document group was dropped. Contracts stay uncovered.

As the prereg says, the five failing families leave IB2 as built. Their mechanisms are visible from the construction:

- `fc_sel`: distractors were drawn uniformly over distinct function names, while the gold is a frequently called
  function, so the names alone reveal it; the gold also used the conversation's own description, the distractors the
  most frequent one.
- `fc_args`: replacement values were drawn uniformly over distinct values and Glaive repeats a few argument strings
  (for example the same body measurements) thousands of times, so frequent strings mark the gold.
- `gsm`: "no" rows stated a calculator intermediate, whose magnitude and form differ from final answers.
- `qasc`, `arc`: the published option texts alone identify the key above chance. IB2 does not rewrite published
  options, so these two are **not** redesigned; knowledge MCQ stays uncovered in IB2.

## B. One redesign each (new family names; the same G4 gate; a failing redesign is dropped, no further attempt)

- **`fc_sel2`** — as `fc_sel` (§2.1) except: distractors are drawn from the list of **first-turn call instances** of
  all conversations (one entry per call, hash order `ib2-fcsel2-pool-v1`, walked from `hash("ib2-fcsel2-start-v1:" +
  key)`), so a function appears as a distractor in proportion to how often it is called; **every option, the gold
  included, uses the function's most frequent description**. The unrelatedness rules (no shared content token with the
  gold name or the request; not listed in the conversation; distinct descriptions) are unchanged.
- **`fc_args2`** — as `fc_args` except: a replacement value is drawn from the list of **all call instances' values**
  of that function and parameter (hash order `ib2-fcargs2-pool-v1`, walked from `hash("ib2-fcargs2-start-v1:" + key +
  parameter)`), still of the same type, different from the gold and not grounded in the request; and **at most 2 rows
  per (function, gold argument object)**, so no gold string is frequent.
- **`gsm2`** — as `gsm` except: a "no" row (`hash("ib2-gsm2-v1:" + key)` odd) states the **final answer of another
  GSM8K train problem** (hash order `ib2-gsm2-pool-v1`, walked from `hash("ib2-gsm2-start-v1:" + key)`, first one that
  differs numerically), so yes and no rows draw the stated answer from the same distribution. Every problem gives one
  row; yes / no equal.

## C. Run and review

Run **`c3`** builds the eight families `fc_rel`, `fc_sel2`, `fc_args2`, `fc_ready`, `ytspam`, `argq`, `hover`, `gsm2`
from the same pinned raw files (the unchanged families are byte-identical to `c2`), and **every audit runs again on
`c3` from the start** (G0 with fresh positive controls, G1, G2, G3, pass 1, rescans, pass 2, G4 on every family,
G5–G8, leak guard). The review sample is drawn from `c3` only; with F = 8 surviving families stage R takes
`max(18, ceil(216 / 8))` = 27 rows per family (216). `c2` stays on the node as the record of this amendment and is
not published.
