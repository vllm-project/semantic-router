# Recipe Lab walkthroughs

The examples use Bash from the checkout root. Set `WORK` to your experiment directory and initialize the shared commands below. Supply the paths and model aliases described in each example. To try the decisions with synthetic data first, run the offline tests at the end of this page.

```bash
mkdir -p "$WORK"
lab() { python tools/recipe-lab/recipe_lab/loop.py --workspace "$WORK" "$@"; }
```

## Serve two models before you have a dataset

A team has one fast text model and one stronger model that can call tools. Its client sends office questions, software design requests and occasional tool calls. It needs one public model name today. Speed provides a starting policy for routine text, reasoning and tools across subjects. Topic-specific guarantees, such as sending every legal question to the stronger model, require a separate rule and examples that exercise it.

Use the v0.5 provider configuration for the two connected models. Set `PROVIDERS` to that file, `FAST` and `STRONG` to its provider model names, `PUBLIC` to the name clients will request, and `ROUTER` to the Router executable paired with your `vllm-sr` CLI. The provider entries must declare actual capabilities and context and output limits. This example binds tools to the strong backend.

```bash
python tools/recipe-lab/recipe_lab/speed_draft.py \
  --providers "$PROVIDERS" --fast "$FAST" --strong "$STRONG" \
  --tool-model "$STRONG" --public-model "$PUBLIC" \
  --vllm-sr vllm-sr --router "$ROUTER" --out "$WORK/draft"
```

Use a fresh draft directory. Read `draft/bindings.yaml` for the model order in each lane and `draft/manifest.json` for the template and validators used. A status of `statically_validated` means the CLI and Router accepted the generated configuration. With the configured decision model and generation backends available, start an instance using this exact file.

```bash
vllm-sr serve --config "$WORK/draft/config.yaml"
```

Once the service is ready, run these probes with the same environment. They exercise the public name that the client will use.

```bash
vllm-sr route probe --config "$WORK/draft/config.yaml" --model "$PUBLIC" \
  --prompt 'What does HTTP stand for?' \
  --expect-recipe speed --expect-decision fast --expect-selected-model "$FAST"
vllm-sr route probe --config "$WORK/draft/config.yaml" --model "$PUBLIC" \
  --prompt 'Design a multi-region checkout cache. Compare two invalidation protocols and prove which prevents stale prices after an acknowledged update.' \
  --expect-recipe speed --expect-decision reasoning --expect-selected-model "$STRONG"
```

For a client that advertises tools, also let it execute a harmless function and return the tool result on the next request. Check which backend actually answered both turns. For example, attaching tool definitions to a routine question can rule out a text-only fast backend. Finally, send a request outside both models' declared capabilities and verify the refusal. These are deployment checks. Record independently graded tasks when you want to measure quality or latency. [Speed probe examples](../../config/recipes/built-in/latest/mom-v1/probes.yaml)

## Tune Speed using an existing sr-bench development run

Suppose your team already benchmarks a Speed deployment and its two direct models. Use the sr-bench service and store that own those results. The native CLI discovers the local stack, or accepts `SR_BENCH_URL` and `SR_BENCH_STORE`. Get the run ID and target IDs from `vllm-sr benchmark show "$DEV_RUN"`. The [sr-bench guide](../agent/skills/vllm-sr-agent-operations/references/sr-bench.md) covers creating a shared live baseline. Register the routed target with `capture_recipe: true`. Authorize the benchmark worker to read the Router's debug trace on this test instance. The worker checks the active configuration hash in that trace on each routed request. A successful chat response alone cannot provide that evidence. Keep the original recipe and the direct strong model in your comparison design if both matter for adoption.

Use the benchmark CLI and worker from the same release as the Router. When registering a target on the worker host, use `vllm-sr benchmark --store "$WORKER_STORE" target register --help` for that store. Verify the target is visible via `vllm-sr benchmark --url "$SR_BENCH_URL" target list`. A local registration in another store cannot be used by this worker.

While the parent is active, point `PARENT_ORIGIN` at its management API. Use the Speed YAML that was deployed. For a CLI-managed stack, also supply `--deployed-config "$PARENT_RUNTIME_YAML"` when the runtime file printed by `vllm-sr serve` has different bytes. Retain its adjacent native provenance file. Choose a new output filename.

```bash
recipe-lab suggest-depth --workspace "$WORK" --config "$PARENT_YAML" \
  --public-model "$PUBLIC" --strong "$STRONG" --fast "$FAST" \
  --dev-run "$DEV_RUN" --parent-target "$PARENT_TARGET" --fast-target "$FAST_TARGET" \
  --endpoint "$PARENT_ORIGIN" --out "$WORK/candidate.yaml"
```

Read the JSON `observations`. A move needs a comparable opening request that used the strong reasoning branch, a correct and faster whole task on the direct fast backend, and no competing conditional branch. A slower or unsuccessful fast result supplies a protected observation. The threshold must leave a protected score on the strong lane. If the result is `keep-parent`, retain the original package and use the recorded observations to decide whether more development evidence is worthwhile. For later tool turns, use the [development ledger](REFERENCE.md#speed-depth-evidence-for-the-development-ledger).

Check `output_policy`, the requested output limit and both finish reasons in the proposal receipt. A fast model that reached `length` was graded under that output reservation. To assess its ability to finish naturally, use a separate, comparable run under the deployment's native output policy where supported. Keep the original measurements available.

If the result is `proposed`, deploy the exact output to a test stack, run a public route probe on both sides of the proposed change and register its sr-bench MoM target. Read the new `active_runtime_hash` from that Router's `/api/v1/config/hash`. Set the minimum acceptable speedup in percent before running the comparison. This example uses the original routed target as the baseline.

```bash
recipe-lab run --workspace "$WORK" --candidate "$WORK/candidate.yaml" \
  --proposer agent:recipe-tuner --baseline-run "$DEV_RUN" --baseline-target "$PARENT_TARGET" \
  --candidate-target "$CANDIDATE_TARGET" --active-runtime-hash "$ACTIVE_HASH" \
  --min-speedup-pct "$MIN_SPEEDUP_PCT"
```

Supply `--deployed-config "$CANDIDATE_RUNTIME_YAML"` for a materialized candidate in the same way as for the parent. The receipt names the submitted run. Reuse it with `--candidate-run "$CANDIDATE_RUN"` to read the evidence again or compare against the direct strong target in the same baseline. Exit 0 means `PASS`, 1 means `FAIL`, and 2 means `INCONCLUSIVE` for a valid invocation. Case times are summed on jointly correct tasks. With the default guards, losing any baseline success fails the comparison. The command's wait limit controls how long the client watches the run. The worker follows the benchmark's own execution policy.

Freeze the chosen candidate and the adoption criteria before revealing any holdout grades. Prepare a separate full baseline on the unseen cases, then repeat `run` with its ID and targets. Record both comparisons and deliver the package that satisfies the criteria. The development verdict alone cannot establish a holdout gain.

## Adapt a hand-tuned rule to a new deployment

A service owner has a reference recipe that switches to a smaller model late in a tool session. The current parent recipe sends these requests to the strong model. Before committing to another set of whole-task runs, the owner wants to know which calls the proposed rule would have touched locally.

Set `CONFIG` to the exact deployed parent YAML and `ANCHOR` to the reference YAML. Set `PUBLIC`, `STRONG` and `FAST` to this deployment's public model name and its two provider model names. Set `RUN` to a development run exported by the benchmark adapter and `ARM` to its recorded arm name. The export includes a frozen task, an independent grade, DSH receipts and the request history. Keep the source files at their recorded paths. [Run layout and import contract](REFERENCE.md#optimize-a-deployed-v05-recipe)

```bash
lab record --run "$RUN" --arm "$ARM" --config "$CONFIG" --split dev \
  --public-model "$PUBLIC" --strong "$STRONG" \
  --hypothesis 'Measure late tool turns on the current parent' > "$WORK/parent.json"
PARENT=$(python -c 'import json
from sys import argv
with open(argv[1], encoding="utf-8") as receipt:
    print(json.load(receipt)["record_id"])' "$WORK/parent.json")
plan() { lab plan-dev --base "$PARENT" --fast "$FAST" --anchor "$ANCHOR" "$@"; }
plan > "$WORK/plan.json"
```

Read `next.action`, `prefix_evidence` and `ranked_probes`. A reproducible planner fixture reports that parent calls 2 and 3 account for 3 seconds of a 12.5-second task. Those 3 seconds tell you where a change could matter. Delivering a different reply may change how many tool calls follow. If the action is `diagnose_reference`, check the reference's upstream model identities, limits and trigger against the current deployment.

When `next.action` is `probe_public_route`, materialize the candidate once. Deploy it on your test gateway. Send requests below and inside the threshold, exercise a tool round trip and test eligibility limits. Set `PROBE` to the normalized public-probe receipt your harness captured for this candidate.

```bash
plan --out-candidate "$WORK/candidate.yaml" > "$WORK/proposal.json"
# After probing the deployed candidate
plan --probe-receipt "$PROBE" > "$WORK/verified-plan.json"
```

If your isolated gateway changed provider connection settings, pass the actual deployment YAML using `--probe-config`. If the action is `find_distinct_reproducible_guard`, collect fresh parent records on a different task that the baseline can solve repeatedly. Import them with `record` and repeat the last command with one `--prefixes RECORD_ID` per measurement. If the action is `plan_paired_development`, create a v2 schedule card binding the verified plan and candidate. [Plan inputs and probe receipt](REFERENCE.md#choose-a-configuration-and-development-task-before-starting-dsh) and [paired card and dispatch](REFERENCE.md#schedule-a-paired-development-experiment).

## Decide whether a faster candidate is safe to keep

Suppose one development task is a slow five-day planning problem and another checks a storage behavior the current recipe consistently gets right. The first is the optimization target. The second is the guard. If you already have a graded development matrix, `screen-dev` can help select that pair. A large same-task time difference between two successful policies makes a useful target. Expensive mandatory grading makes a poor choice for frequent repetitions.

```bash
lab screen-dev --manifest "$DEV_MANIFEST" --max-seconds 600
lab schedule --card "$CARD"
```

Here `DEV_MANIFEST` points to the audited matrix and `CARD` is the frozen development card for the selected pair. The 600 seconds is a reference for selecting tasks. Each DSH run retains its normal termination conditions. The card budgets whole attempts, including invalid attempts.

For each `run` status, give the adapter the exact task, config fingerprint and `next.assignment_id` **before** it launches the worker. Import the independently graded result. Set `ATTEMPT` to the receipt that binds this assignment to its measurement, then request the next slot. Once the study reaches a terminal decision, read the summary.

```bash
lab schedule --card "$CARD" --attempt "$ATTEMPT"
lab decision-summary --card "$CARD"
```

Start with the strict grades on the guard and target. If the guard loses a previously reproducible success, `drop` retains the parent. If the same task alternates between paired wins and losses, `defer` can request a repeat even when the other task completed faster. For an `advance` decision, confirm the frozen configuration on previously unused tasks against the original SR and the direct strong model. To evaluate how well the **search method** selects recipes, enroll the three methods before the first trial. [Search enrollment and independent evaluation](SEARCH.md#compare-three-search-methods)

## Run the examples with synthetic receipts

With Python, PyYAML and pytest available, these tests exercise the planner, paired scheduling and a frozen three-method comparison in temporary workspaces. They require no serving endpoints or benchmark downloads.

```bash
python -m pytest -q \
  tools/recipe-lab/tests/test_planner.py::PlannerTest::test_plan_publish_anchor_and_duplicate_predicate \
  tools/recipe-lab/tests/test_planner.py::PlannerTest::test_ready_plan_enrolls_separately_measured_guard_and_freezes_feedback \
  tools/recipe-lab/tests/test_schedule.py::ScheduleTest::test_opposite_quality_pairs_require_repeat_before_latency_promotion \
  tools/recipe-lab/tests/test_improver_eval.py
```
