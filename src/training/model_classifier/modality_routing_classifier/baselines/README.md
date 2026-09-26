# Fixed-policy controls — modality routing

Simple untrained baselines for AR / DIFFUSION / BOTH. For #3857.

## Results

Test split, 759 rows (AR 300 / DIFFUSION 300 / BOTH 159).

| control | accuracy | AR recall | DIFFUSION recall | BOTH recall |
| --- | --- | --- | --- | --- |
| lexical rules | **0.9025** | 0.9300 | 0.9133 | 0.8302 |
| majority class (AR) | 0.3953 | 1.0000 | 0.0000 | 0.0000 |
| random by class prior | 0.3572 ±0.0173 | 0.3957 | 0.3957 | 0.2086 |

Lexical precision: AR 0.8404, DIFFUSION 0.9352, BOTH 0.9851.

Train is 0.9031, test 0.9025 — the rules generalize.

## Run

```bash
python3 export_modality_dataset.py --max-samples 6000
python3 -m pytest baselines/test_fixed_policy_controls.py -q
python3 baselines/fixed_policy_controls.py --split test --json baselines/results_test.json
```

Export is deterministic (70/15/15, seed 42). Priors and majority class come
from train, never from the split being measured. Rules were written against
train only.

## What this shows

**1. The floor is 0.9025, not 0.3953.** Thirty lines of regex get 90%. A trained
model has to beat that, not the majority-class number.

**2. BOTH is mostly boilerplate.** 61 of 159 BOTH examples share four templates
built on "consists of both text and multiple images...". One regex on that
phrase gets 61/159 with no false positives.

**3. About a sixth of BOTH has no image request in it.** *"How do I install a
ceiling fan?"* is labelled BOTH but looks exactly like AR. The label means "this
answer would be better with pictures", not "the user asked for pictures". So a
model scoring well there is learning which topics suit images, not reading the
request.

`test_unsignalled_both_falls_through_to_ar` pins this: add a rule that catches
those and the test fails.

## Caveats

- Split is balanced on purpose (2000/2000/1055). Real traffic is mostly AR, so
  these floors apply to this eval set only.
- Rules are English-only, the baseline model is multilingual. Non-English
  examples make the lexical control look worse than it is.
- DIFFUSION examples are scraped image prompts, full of style words a regex
  catches easily. Real requests would not look like that.
- No cost controls. Cheapest/strongest do not apply when the classes are
  capability requirements, not price points — that belongs in #3198 once #3856
  gives per-request token counts.
