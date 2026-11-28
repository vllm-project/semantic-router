# Search and evaluation

Run these commands from `tools/recipe-lab` with the same workspace and installed configuration as the development loop. The benchmark adapter runs the stock agent, obtains independent strict grades and exports evidence. The Lab chooses assignments and verifies the receipt files. The [worked scenarios](SCENARIOS.md) introduce the development loop. Python, PyYAML and pytest are sufficient to run its synthetic regression tests offline.

## Freeze a development objective

Before opening development scores, save a `recipe-lab-objective-v1` JSON and its digest. Add `"objective": {"file": "objective.json", "sha256": "…"}` to the development matrix manifest passed to `headroom`. The declaration fixes the study, arm order and policy hashes, per-task controls, target and guard tasks, hashes of held-out task IDs, repeats, baseline, threshold values and any external checks. The report distinguishes thresholds measurable from the matrix and checks that require other evidence. Older matrices remain readable. The [objective fixture](tests/test_headroom.py) gives a complete example.

## Evaluate a skill revision

Casebook `sources` point to repository material for applying or testing a mechanism. The digests in `historical_evidence_sha256` identify the original operator-held reports or configurations. The measured figures retain their original denominators and deployment assumptions.

```bash
python recipe_lab/loop.py --workspace "$WORK" skill-replay \
  --point "$DEV_CARD" 2 --before-skill "$OLD" --after-skill "$NEW" \
  --revision-source "$DEV_COMPARISON" > "$WORK/packet.json"
```

The integer sets how many attempts are visible at the checkpoint. Give each agent its own skill and the same exported point. Each agent chooses `continue` or `keep_parent`. Save both choices in a file with the packet's `packet_sha256` and `choices[study][before|after]`. Each choice records its `choice` and `skill_sha256`. Repeat the command with `--decisions FILE`. The Lab freezes the paired choices before scoring the rest of the verified development history. Read lost advancing candidates and counterfactual remaining episodes separately. This tests one early-stop decision. A new task family provides independent evidence for transfer.

## Compare three search methods

Use one workspace for the study. First run `search-preflight` on the common tool-window edit space, development allocation, independent holdout allocation and proposal/episode budgets described in the [command reference](REFERENCE.md#check-whether-a-matched-budget-search-comparison-can-start). Supply `--target-prefixes` for the allocated development tasks and `--candidate-trials-per-method`. Enrollment requires distinguishable options beyond the number one method may try.

1. Run `search-enroll` with the same preflight arguments and `--allocation FILE`. The file declares `schema: recipe-lab-search-allocation-v1`, the study name, the protocol digest and `methods`. For each of `existing_skill`, `random` and `guided`, assign as many unique, ordered trial study names as the common trial limit. This reserves future slots while allowing the method to choose its next candidate after reading the previous result.
2. Use the ordinary `schedule` command under each reserved trial name. The random method follows the seeded representative order from preflight. Every method uses the same paired target, guard, repeats and per-trial episode limit. Finish started trials. Unused slots and studies ending without valid grades are reported separately. Capture each proposer call's provider response ID and usage at execution time. A method can retain the parent when it finishes.
3. Run `search-freeze` with the same preflight arguments and `--selections FILE`. Use `schema: recipe-lab-search-selections-v1`, `protocol_sha256`, `direct_strong_sha256`, `holdout_repeats`, and a `methods` object. Each method declares its `policy_sha256`, ordered `trial_cards` and `usage`. Each file reference is a sibling filename and its SHA-256. A method selecting a changed policy needs an advancing development trial. The policy of the direct control identifies its model and deployment settings.
4. Give the adapter the returned `holdout_assignments`. They interleave the three frozen choices, fixed SR and the direct strong model. Before launching a run, write its `assignment_id`, `instruction_sha256`, `policy_sha256` and `controls_sha256` into its execution binding. Capture a distinct episode fingerprint, elapsed time and the independent outcome for every attempt. Keep native receipts and exporter provenance.
5. Run `improver-eval --study NAME --manifest FILE`. Use the audited matrix schema from `headroom`, with `split: holdout`, the `freeze_sha256`, and an explicit `invalid_attempts` array. Every row includes the freeze and instruction hashes plus pinned `binding` and `outcome` files. The outcome has the assignment ID, episode digest, seconds, assessment and either the boolean strict grade or the invalid reason. Both evaluation inputs and the stored verdict can be rechecked. A second, different matrix for the same study is rejected. When two methods deliver the same policy, their search costs remain comparable.

The usage manifest has `schema: recipe-lab-search-usage-v1`, `enrollment`, `method` and `calls`. Each call pins a sibling `recipe-lab-proposal-call-v1` receipt containing the enrollment, method, trial study and provider response. The response supplies its ID and prompt, completion and total tokens. A capture with no proposer calls uses an empty list. Completeness of exported calls, runtime timestamps and native grader correctness belong to the adapter.

The scheduler checks the chosen configuration and the method's remaining whole-episode budget before dispatch. An invalid attempt consumes a slot in that budget. A task already in progress keeps the agent's normal termination conditions. The proposer or its calling adapter controls token spending while a proposal is generated. Final accounting verifies the exported usage against the enrolled ceiling.

The [CLI regression](tests/test_improver_eval.py) builds the input files, enrolls all three methods, completes their paired schedules, freezes the choices, exports a five-arm synthetic holdout and reads the final result twice. It can be run with `python -m pytest tests/test_improver_eval.py`.
