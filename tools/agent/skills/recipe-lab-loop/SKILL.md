---
name: recipe-lab-loop
description: Use when improving routing accuracy through the recipe-lab evidence loop - reading misroute attribution, proposing recipe candidates, evaluating through the replay/live tiers, and recording keep/revert decisions in the ledger.
---

# Recipe-lab improvement loop

Drive the closed loop with `tools/recipe-lab/recipe_lab/cli.py` (installed as
`recipe-lab`): attribution -> proposal -> replay-tier search -> live verdict
-> ledger. sr-bench owns evaluation; this loop only orchestrates it. Read
`references/command-reference.md` for the command surface,
`references/evidence-interpretation.md` for reading outcomes, and
`references/red-lines.md` before proposing anything.

## Ground rules

- Never touch a stack you did not start; run `vllm-sr status` first.
- Credentials stay in environment variables, never in YAML, args, or logs.
- frozen-test is quarantined: proposals and agents see calib/dev only.
- A `RANKED` verdict is a ranking, not a conclusion. A `PASS` is a
  recommendation, not a promotion - promotion is human + Standard holdout.
- Change one variable per candidate; re-run the current step after it.
- INCONCLUSIVE and no-evidence are evidence problems: gather more evidence,
  never record a guess.

## Steps

1. Validate the experiment contract: `recipe-lab plan --spec spec.yaml`.
   Success: normalized spec JSON. Failure: fix the spec per the contract error.
2. Read the evidence: `recipe-lab next --spec spec.yaml --ledger ledger.jsonl
   --output context.md` (add `--diagnosis diag/diagnosis.json` when a
   diagnosis artifact exists). Read the untrusted-material header, then the
   contract summary, best replay candidate, last live verdict, attribution
   summary and the citable insight-ref list it exports (G3).
3. Propose one candidate: edit one recipe variable in a config revision and
   cite the motivating evidence with `--insight-refs`, using the exact refs
   from `next`'s citable list (`rec-<id>`, `bench:<run_id>`,
   `attr:<path>@sha256:<hex>`, `diag:<path>@sha256:<hex>#<anchor>`).
   When you authored the candidate config yourself (not a proposer output),
   evaluate it with `--proposer agent:<id>` in step 5; the id is lowercase
   `[a-z0-9._-]` (e.g. `agent:claude-loop`) and is recorded verbatim in the
   ledger provenance.
4. Validate the candidate: `vllm-sr config validate` on the revision.
   Failure: fix the config; never evaluate an invalid candidate.
5. Evaluate in the replay tier: `recipe-lab run --spec spec.yaml --split dev
   --cases <split.jsonl> --bank bank.jsonl --candidates <config>
   --proposer agent:<id> --ledger ledger.jsonl --router-bin ... --mock-backend
   ...` against the answer bank. Records land in the ledger.
6. For the final gate: run sr-bench preview/live for baseline and candidate,
   then `recipe-lab live-verdict --spec spec.yaml --baseline-run-id ...
   --candidate-run-id ... --guard-measurements measurements.jsonl
   --candidate cand-001.json --ledger ledger.jsonl` (add `--split frozen_test
   --final` only for the promotion verdict).
7. Read the outcome with `recipe-lab report --ledger ledger.jsonl --output
   report.md`; stop when the objective is met, the budget is exhausted, or
   evidence is repeatedly incomplete.

Run `make check` after changing recipe-lab code; skill changes follow the
`tools/agent` structure checks and public-skill sync tests.
