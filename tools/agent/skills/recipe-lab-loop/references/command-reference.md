# recipe-lab command reference

All commands print a single structured JSON document on stdout
(allow_nan=False, UTF-8); progress and logs go to stderr; no command reads
stdin. Exit codes: `0` success (for `live-verdict`: PASS), `1` FAIL verdict
(`live-verdict` only), `2` usage, contract or tool error, and INCONCLUSIVE
for `live-verdict`.

A usage error (unknown command, missing or conflicting argument) is itself a
JSON receipt on stdout, so there is always a document to parse:
`{"schema": "recipe-lab-cli-error-v1", "verdict": "INCONCLUSIVE",
"reason": "..."}`, exit 2. Contract and tool errors exit 2 with the reason on
stderr.

## Contract

- `recipe-lab plan --spec S`
  Load and validate the experiment contract and print it normalized; no
  experiment runs. Contract errors exit 2 with the reason on stderr — fix
  the spec and re-run. Every later command re-checks the same contract, so
  a spec that passes this command is the single source of truth for a round.

## Dataset identity and splits (stage 2)

- `recipe-lab hash --file F`
  Frozen dataset identity: `sha256:` + hash of the raw bytes (A4). The spec's
  workload records this hash; every command that touches the dataset re-hashes
  and refuses drift (exit 2).
- `recipe-lab split --cases C --output-dir D [--group-key source_id
  --ratios 0.4,0.3,0.3 --seed 42]`
  Deterministic group-level three-way split (B): `calib.jsonl`, `dev.jsonl`,
  `frozen_test.jsonl`. Same input + same flags -> byte-identical output.
  `frozen_test` is quarantined: only `live-verdict --final` may cite it.

## Case conversion (X3)

- `recipe-lab cases to-srbench --cases C --output O [--benchmark recipe-lab]`
  Deterministically render a cases JSONL as an sr-bench dataset
  (`case_id`->`id`, `prompt`->`messages`, `truth`->`answer`). Same cases ->
  byte-identical output, so the rendered dataset is freezable by hash.
  Signal-fact truths have no sr-bench answer to grade against and exit 2.

## Evidence intake

- `recipe-lab attribute --spec S [--split calib|dev] --records R
  [--cases C] --output-dir D`
  E1/E2/E3 misroute attribution; writes `attribution.jsonl`, `summary.json`
  (the summarize() result, read by `next`, G3) and `report.md`.
  The split manifest is re-hashed (A4); optional `--cases` verifies the
  attribution input against the split identity. `frozen_test` is refused.
- `recipe-lab attribute-session --records R --output-dir D`
  Session-aware agentic attribution over graded records that carry session
  evidence (`conversation_id` / `turn_index` / `session_action` /
  `selection_reasoning`, plus the record timeline and lifecycle fields).
  Writes `session-report.md` and `session-events.json`; evidence classes are
  `lane_flap` (adjacent model-bearing requests of one conversation on
  different models), `tool_loop_interrupted` (a failed selection after a
  lane switch inside a conversation) and `selection_failed` (a rejected
  selection, the P1 consumption of `lifecycle_state=failed` records, which
  are never deduped away).
- `recipe-lab collect-replays --url http://127.0.0.1:8080 --cases C
  --output R --family d3`
  Pulls router_replay records from the management API and joins them with
  labeled cases. Skipped rows are counted (`no_prompt`/`unlabeled`/
  `duplicate_prompt`), never guessed.
- `recipe-lab bank build --cases C --models M1,M2 --endpoint U --output B`
  Direct per-case/per-model queries graded by `expected_contains`.
- `recipe-lab bank apply --bank B --records R [--output R2]`
  Joins correctness by (case_id, routed model); missing entries stay
  `no_evidence`.

## Proposal and replay tier

- `recipe-lab propose-thresholds --base-config C --signal S --deltas=-0.1,0.1
  [--recipe R | --public-model M] --output-dir D`
  Deterministic single-variable threshold candidates (baseline dropped,
  clamped to [--lo, --hi]). Both proposers read the canonical v0.5 layout
  (`recipes[].signals`, `recipes[].decisions`). A document with several
  recipes (the multi-objective ones) needs `--recipe <name>` or
  `--public-model <name>` to name the one to edit; without either, a
  single-recipe document is assumed and several are a usage error.
- `recipe-lab propose-rule-thresholds --base-config C --signal S
  [--from V] --values V1,V2 --output-dir D [--decisions D1,D2
  --allow-partial --op gte --lo L --hi H --list-sites]
  [--recipe R | --public-model M]`
  Moves every rule-level predicate site of one signal to each target value,
  including sites under NOT branches (C; the BUG-3 golden case moves four
  `difficulty gte 2.5` sites to 1.75 in one shot). Each candidate config
  ships a `<stem>.candidate.json` sidecar with kind/ref/change_summary/
  provenance/sites that `run` records in the ledger.
- `recipe-lab run --spec S [--split calib|dev] --cases C
  --bank B --candidates D [--baseline-config B] --proposer P --ledger L
  --output-dir D --router-bin BIN --mock-backend M [--mock-extra-args "..."]
  [--validate [--validate-bin vllm-sr]] [--attribution-ref REF]
  [--insight-refs "..."] [--round N]`
  One router lifecycle per candidate; RANKED records land in the ledger.
  The split identity is re-hashed before any stack work (A4); the trial
  deadline is `budget.max_minutes_per_trial` (A5, timed-out candidates get
  `status=timeout` and never rank). `--validate` pre-flights each candidate
  with `<bin> config validate`; rejected candidates never reach the ledger,
  all-rejected exits 2. `--candidates` accepts a directory, a comma-separated
  list, or a single config path — the agent's own proposal entry: pass one
  config plus `--proposer agent:<id>` so the ledger records who proposed it.

- `recipe-lab cases from-replays --records R --output O [--granularity
  session|prompt|request] [--provenance P] [--duplicate-warn-pct N]`
  AG3: build route-replay cases from router replay records. The default
  granularity is `session` when the records carry conversation identities,
  else `prompt` (the J3 convention). Dedup rules and their effects are
  written into the provenance JSON; a duplicate share above the warning
  threshold (default 50 percent) warns toward a coarser granularity.

## Insight references (stage 2)

`--insight-refs` (comma-separated, on `run`, `ledger append` and
`live-verdict`) merges provenance links into
`candidate.provenance.insight_refs`; unverifiable references exit 2:

- `rec-<id>` - an existing ledger record_id (G1: every record carries one)
- `bench:<run_id>` - a benchmark run id (format-checked)
- `attr:<path>@sha256:<hex>` - attribution artifact, hash-verified
- `diag:<path>@sha256:<hex>#<anchor>` - diagnostic artifact with an anchor

## Live tier

- `recipe-lab live-verdict --spec S [--split dev|calib|frozen_test] --final?
  --baseline-run-id A --candidate-run-id B [--guard-measurements M]
  [--candidate J] [--insight-refs R]
  [--ledger L] [--srbench-bin vllm-sr --srbench-url U --srbench-store P]`
  Compares two sr-bench runs, applies guard floors, records the three-state
  verdict with gates and bench_refs. `frozen_test` requires `--final`
  (the promotion verdict is its only consumer, B).
- `recipe-lab verify --spec S [--split dev|calib|frozen_test] --final?
  (--baseline-run-id A | --baseline-manifest M) --candidate J
  [--candidate-targets T1,T2] [--repetitions N] [--work-dir D]
  [--guard-measurements M] [--insight-refs R] [--attribution-ref REF]
  [--ledger L] [--run-cache C]
  [--srbench-bin vllm-sr --srbench-url U --srbench-store P]`
  Drives sr-bench end to end for one candidate (I): preview routing first,
  then AB/BA repetitions (each pair re-runs the baseline's frozen manifest
  and submits a fresh candidate plan), per-pair compare, and the shared
  three-state verdict; any incomplete pair is INCONCLUSIVE. Exit codes
  follow the verdict (0 PASS / 1 FAIL / 2 INCONCLUSIVE or error). Repeated
  invocations reuse cached reference-arm run IDs (M4). `frozen_test` and
  transfer workloads require `--final` (M5).
- `recipe-lab score-candidate --workspace W --candidate C --proposer P
  --baseline-run A --baseline-target T --candidate-target U
  --active-runtime-hash H [--candidate-run R] [--guard-case ID]
  [--min-speedup-pct X] [--wait-seconds S] [--vllm-sr BIN]`
  Scores one deployed candidate against a saved sr-bench baseline run
  through the installed `vllm-sr benchmark` worker: the agent's own
  proposal, submitted as `--proposer agent:<id>`, with no spec. The plan must
  reproduce the frozen baseline and match the active runtime hash before
  anything is dispatched. Prints one JSON receipt
  (`recipe-lab-sr-bench-run-v1`, also saved under `<workspace>/sr-bench`);
  the exit code follows the verdict.
- Guards (A3): floors come only from `spec.gates`. `--guard-measurements`
  is a JSONL of measurements only - `{"name": "pii-recall-floor",
  "candidate": 0.97, "baseline": 0.96}`; a `floor` key in the file is a
  contract bypass and exits 2. `floor: baseline` in the contract resolves
  against the measured baseline value. A gate without a usable measurement
  is unevaluated (verdict INCONCLUSIVE); undeclared measurement names are
  reported but never veto.
- Constraints (E): `objective.constraints` declares economic conditions
  (`{name, metric, op gte|lte, bound: "baseline"|<number>, tolerance_pct}`);
  distinct from gates, which are safety vetoes. Verified sr-bench compare
  cost fields: `total_spend_usd`, `cache_neutral_cost_usd`, `cost_usd`,
  `evaluation_cost_usd` on the baseline_targets/candidate_targets entries.
  Unknown usage is null -> the constraint is INCONCLUSIVE (never zero).
  Constraint evidence lands in the record's `constraints` field.

## Reporting

- `recipe-lab ledger show|append|top`
- `recipe-lab report --ledger L --output R` - ledger-only replay with a
  consistency check.
- `recipe-lab next --spec S --ledger L [--diagnosis diag/diagnosis.json]
  --output C [--tail N]` - full agent context export (G3), byte-stable for
  the same inputs. Sections: untrusted-material header; contract summary
  (objective, replay_metric, constraints, gates, split names + hashes,
  budget used/remaining); ledger tail; best replay candidate (ranked by
  `objective.replay_metric` through the same `best_replay` selection as
  `ledger top`, A2); last live verdict with reasons, gates, constraints and
  bench_refs; latest attribution summary read through the newest record's
  hash-verified `attribution_ref`; optional diagnosis highlights (dead/rare
  decisions, unreachable threshold sites); and a citable insight-ref list -
  paste those refs verbatim into `--insight-refs` when proposing (step 3).
- `recipe-lab provenance --ledger L --output M` - recipe-lab-run-v1 manifest.
- `recipe-lab record-draft --ledger L --output D` - promotion draft from the
  latest PASS live record (human checklist included).

## Agentic tier (sub-issue 6)

- `recipe-lab profile-agentic --reports R [--reports R2 ...] --output P`
  AG5: aggregate per-model single-target sr-bench run reports into the
  agentic profile (per-model task success / cost / turns / truncation).
- `recipe-lab bootstrap-agentic --spec S --profile P [--template T] --client C
  [--recipe R | --public-model M] [--lambda X] [--local-model M] --out D
  [--validate]`
  AG5: generate the agentic candidate from the agent recipe template, the
  profile and the client identity contract; every lane's choice rides a
  sidecar with its rule and evidence (unmeasured-lane default rule when the
  profile has no lane data). The template is the canonical v0.5 document
  (default: `config/recipes/agent/config.yaml`); the rewrite edits one recipe
  in `recipes[]` - its decisions take the lane models and the recipe carries
  `candidate_requirements`. A document with several recipes takes `--recipe
  <name>` or `--public-model <name>` to name the one to rewrite.
- `recipe-lab propose-session-policy --config C [--diagnosis D]
  [--replay-summary S] --output-dir D`
  AG6: cause-keyed one-variable session-protection candidates (identity
  headers, stability weight, switch margin, agent identity header); drift
  releases and rejected selections come back as handoffs, not knobs.
- `recipe-lab run-agentic --tasks T --target U --out D [--only ID]
  [--retry-policy P] [--max-turns N] [--timeout-s S] [--env K=V]`
  AG2: drive a real agent client (headless Claude Code) over a task file
  through one target - one working directory and session per task; the
  local proxy journals identity headers and per-request ids.
- `recipe-lab judge-agentic --run-dir D [--judge-command C] [--dataset-name N]`
  AG2: judge a run-agentic directory with the SWE-bench official harness ->
  sr-bench-shaped per-task rows the paired machinery and AG7 gates consume.

## Notes

- Replay tier needs no LLM backend: the router and signal model run on CPU.
- A store with prior sr-bench execution evidence refuses autostart; start it
  explicitly with `SR_BENCH_STORE=<path> vllm-sr benchmark serve`.
- The router's benchmark binding supports `X-SR-Bench-Max-Inference-Calls: 1`
  only (`pkg/extproc/benchmark_config_binding.go`).
