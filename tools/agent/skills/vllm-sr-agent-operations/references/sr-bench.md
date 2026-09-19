# sr-bench evaluation loop

Use sr-bench to compare single models and MoM on shared cases, then improve a
recipe with measured quality, cost and latency. Discover commands with
`vllm-sr benchmark --help`; use the [product guide](https://vllm-sr.ai/docs/benchmarking/sr-bench)
for adapters, dataset preparation, manifests and accounting details.

## Choose the scope

**Profile controls the questions:** Smoke checks the pipeline cheaply; Quick/dev
supports tuning; Standard is a disjoint holdout for a frozen candidate. Select
benchmarks relevant to the capability change and inspect the planned case count
and limits. A slice is not a full benchmark score. Whole agent tasks may consume
many generation, simulator and judge calls.

**Mode controls execution:** Preview diagnoses routing without generating answers;
Live measures capability, usage and latency. Replay estimates a route from saved
single-model answers only when the service finds compatible evidence. It does not
replace live validation or measure new latency.

For a new mixture, discover actual providers/entrypoint and verify basic delivery
first. `benchmark setup` checks adapter prerequisites; use its install/sandbox
options only when needed. CLI and Dashboard must point to the same service/store.
Remote dataset paths are worker-local paths, not uploads. Discover registered
targets and datasets before preparing duplicates; credentials stay server-side.

## Run one improvement cycle

1. Create an experiment to group the work. Define the capability objective, allowed
   quality loss, cost basis and time/call/output budgets. Use Smoke preview and
   bounded live Smoke to verify routing, final-answer grading and accounting.
2. Freeze a Quick/dev dataset and run the relevant single-model baselines plus
   the current MoM. Link those runs to the experiment. Reuse compatible saved
   baselines in later iterations rather than regenerate them.
3. Inspect dev failures and route/cost distributions. Make one coherent recipe
   change through [config validate/plan/apply](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md),
   then verify its active revision and update the registered MoM target's
   `config_hash` to match. Existing runs retain their frozen target definitions.
   Preserve the previous recipe for recovery.
4. Use `benchmark candidate-plan BASELINE_ID --target REGISTERED_MOM` to inherit
   the exact baseline cases and request protocol; select Preview or Live and
   attach the experiment. Its output wraps the manifest; extract `.manifest`
   before passing it to `run` or `preview`. Preview the candidate first. If useful, discover replay
   combinations with `replay-options`; otherwise proceed to bounded live work.
5. Compare the live candidate with the strongest observed single on the same
   cases using `comparison-options` and `compare`. Keep or revert according to the
   stated objective. Repeat on dev when warranted; freeze the chosen recipe
   before the Standard holdout. Do not tune on holdout failures.
6. Inspect the same experiment in Dashboard and deliver its runs, recipe and
   report. For a functionality demonstration, a Smoke cycle may exercise the
   whole workflow, but explicitly leave capability/holdout qualification pending.

Use command-specific help for experiment create/attach/delete, candidate planning,
run submission and reports. Experiment membership organizes evidence; it neither
starts evaluation nor makes incompatible runs comparable. Deleting an experiment
removes its grouping, not run evidence; active runs must be resolved first.

## Preserve comparability

- Freeze cases, targets, grader/source versions, effective request parameters,
  prices and recipe identity. Target request overrides take precedence over run
  defaults; keep fixed judges/simulators consistent across candidates.
- Preview accepts supported chat messages/tools and optional session context.
  Learning preview is a read-only state snapshot, not a promise of the next live
  choice. Live state is not automatically isolated/reset between candidates.
  Keep Learning when it is the policy being tested and disclose state differences.
- Replay/Compare discovery is authoritative. Follow bounded pages when needed;
  an unfinished or size-limited scan does not prove no compatible result exists.
  Never change cases, parameters or Learning merely to force replay eligibility.
- A terminal full baseline can supply a candidate's frozen protocol even after
  failure. Compare requires an explicit outcome for every planned cell; failures
  count as incorrect and remain visible. Missing or ungraded results are not zero.
- Route only on user-request evidence, never answers, benchmark names or split
  labels. Keep tuning and holdout separate. Disclose previously inspected tasks
  or labels.

## Run reliability

Inspect the plan before dispatch; use bounded deadlines, call/output limits and
cost policy. Submit with a stable idempotency key. After a lost acknowledgement,
look up that run before acting; do not substitute a new key or repeat generations.
`cancel` stops actual work; Ctrl-C only stops the CLI wait.

Inspect failures and saved receipts before recovery. `recover-plan`/`recover`
create explicit child attempts for eligible cells; they do not rewrite a parent
or make an incomplete run complete. Usage reconciliation and supported regrading
operate on retained evidence, not fresh generations. Discover their specific
contracts only when that failure occurs. Poll durable progress with a deadline;
notify on meaningful completion, failure or required action, not every counter.

## Read the outcome

Check planned/completed/scored/failed denominators, final-channel scores, model
and config identity, four token buckets, subject versus judge/simulator spend,
latency and elapsed time. Unknown usage is not zero. `capability_only` permits
unpriced evaluation but cannot support savings claims. With cache effects, report
observed cost and the separate cache-neutral estimate; neither is a GPU invoice.

Savings are `100 × (1 − candidate subject cost / baseline subject cost)`, with
complete compatible accounting. The baseline is the best observed single over
the same aggregate, not a per-question oracle; quality ties use the lowest known
cost. Show signed quality/cost changes, uncertainty and benchmark coverage.
A small dev win or zero observed difference does not establish equivalence.
Keep essential limitations with the result and detailed evidence accessible.
