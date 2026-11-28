# Recipe Lab command and evidence reference

Use the [project overview](README.md) for a first run and the
[worked examples](SCENARIOS.md) for complete workflows. This reference
describes the inputs, output fields and deployment checks for each step.

## Deliver a user report

Open `report.md` beside the generated draft's `config.yaml`. It identifies the template, model bindings and static checks. For `suggest-depth` and `run`, use the `report` path in the JSON result. The `loop compare`, `loop decision-summary`, `loop improver-eval` and `loop plan-dev --probe-receipt` commands print the report location to standard error so their standard output remains usable as canonical evidence. Each workspace report has a sibling `evidence.json` that you can use again.

To hand off a multi-round study, select its receipts in execution order and give the deployment objective in plain language. For example, after evaluating the candidate from a two-model draft:

```bash
recipe-lab report \
  --background "Keep the baseline's task successes and reduce time on shared successes." \
  --evidence ./speed-draft/manifest.json \
  --evidence "$DEVELOPMENT_EVIDENCE" \
  --evidence "$COMPARISON_EVIDENCE" \
  --out ./handoff.md
```

Use the receipts for the same study and identify the evaluated baseline. Include rejected and inconclusive rounds when they affected your choice. The report reads the files you select and requires no model calls. Static validation establishes configuration compatibility. Route probes establish behavior for the requests exercised. Whole-task comparisons support conclusions for the measured tasks, models and conditions. Use a frozen holdout when making claims about performance beyond development.

## Draft a two-model Speed recipe

Use this entry point when you have two connected text models and no labeled
dataset. Supply their aliases from your own canonical provider config. The
strong model handles reasoning and the fast model handles routine requests.
The tools lane needs an explicitly tool-capable backend. The manifest records
the checks that ran. Business quality requires separate task-level evaluation.

From the checkout root, using Python with PyYAML and the CLI and Router intended
for deployment:

```bash
python tools/recipe-lab/recipe_lab/speed_draft.py \
  --providers providers.yaml --strong team/strong --fast team/fast \
  --public-model my/assistant --vllm-sr /path/to/vllm-sr \
  --router /path/to/router --out ./speed-draft
```

Use a new output directory. By default the generator assigns tool calls to
the strong model. If the fast backend is the only one that supports tools,
add `--tool-model team/fast`. The provider config must declare actual model
capabilities and known input-context and output limits on `providers.models[]`.
Use verified limits for the deployed endpoint. Use its existing secret references
and provide only configuration you are authorized to include in the output.
The generator retains the provider definitions, including their connection
settings. It does not copy their external assets. Where local files are needed,
use absolute paths that will resolve in the deployment. Use the deployment's
CLI, catalog and Router together.

The draft uses the installed `mom-v1/speed` policy. It preserves the policy's
questions, priorities and fallback. Each lane has a preferred backend. If only
one backend can accept tool requests, the generator adds it as the second
candidate in the lane whose preferred backend lacks tools. SR filters candidates
against the full request and then uses their configured order. A simple request
can therefore select the fast lane and still reach the strong model when a
client attaches tool definitions. This also permits fallback when another hard
eligibility limit excludes the preferred candidate. If neither model qualifies,
the gateway rejects the request. The decision model serving the template's
questions is separate from the two generation models.

If the caller omits its output limit, each lane uses the template's default
bounded by the declared limits of every candidate in that lane. A shorter
fallback model can reduce this default for the preferred model too. An explicit
caller output limit is preserved and used in the eligibility check.

For an agent such as DSH that may stream one response for several minutes, add
`--agent-streaming` to the generation command. Each Speed decision then sets
`reliability.total_timeout: 0s`. The gateway can keep an active call open past
the listener's whole-call deadline. The configured idle timeout still closes a
stalled stream. This policy applies to all calls on this public recipe, including
non-streaming calls. Client cancellation, provider limits and task deadlines
still apply. The manifest records the change.

Review the resulting `config.yaml`, `bindings.yaml` and `manifest.json`.
With your normal SR runtime available, start the stack using
`vllm-sr serve --config ./speed-draft/config.yaml`. Test the public model name. Use
`vllm-sr route preview` to inspect the decision and `vllm-sr route probe`
to exercise the backend. For the fast and reasoning cases assert the selected
decision and model and the completed backend response. For a tools deployment,
actually execute a harmless tool and send its result back; separately send an
input neither backend supports and verify no generation backend was invoked.
When a signal reads tool output, send two otherwise identical histories through
the public chat endpoint, one with a passing check and one with an outstanding
change. Record the decision and actual backend for each. In Router revision
`e50115865`, management preview counts tool results but omits their text from
SystemOne classification. Replay representative agent histories through the public
endpoint before using preview results to choose a recipe.
For a semantic signal guarded by a tool-count threshold, measure classification
time on requests below the threshold too. In this revision the Router evaluates
referenced signals before it combines their predicates. A rarely selected branch
can add classification time to most requests.

When moving a Speed draft to the pinned `vllm-sr/vela3-fast` 3.0 package
(`7cbefd4b`), check its `noul` questions. Put the meanings of yes and no
in `question.instructions` and omit described `question.choices`. This
package rejects descriptions on the individual binary options. Inspect the
SystemOne answer for an actual score and check the backend reached by the
public request. A successful chat response can hide a rejected classifier
request if the recipe falls back to another decision. Recheck the existing
thresholds with representative examples when switching models.
Find request examples in the [maintained probes](../../config/recipes/built-in/latest/mom-v1/probes.yaml).
Measure answer quality and task completion time against both direct deployments
before claiming an improvement.

Run the offline regression suite with `make test-calibration`. To exercise the
generator's command-line path with your installed validator, set
`RECIPE_LAB_V05_CLI` and `RECIPE_LAB_V05_ROUTER` to the respective executables,
then run `python -m pytest tools/recipe-lab/tests/native_speed_draft.py`.

For an isolated, live wiring check, use
`python tools/recipe-lab/tests/live_speed_smoke.py --help`. Its `fixtures`
and `probe` modes exercise the real gateway with two identifiable responders,
including text streaming, a tool-result round trip and rejection before
dispatch when neither backend qualifies. The separate `probe-optional-tools`
mode checks OpenAI Chat backends where only the strong backend accepts tools.
An available tool makes that backend eligible even when a greeting selects the
fast lane. With `tool_choice: none`, these short requests go to the fast text
backend. The probe checks that the backend receives no tool definitions,
`tool_choice` or `parallel_tool_calls` and returns a complete text response.
Run it on a dedicated stack so counts belong to the probe.

## Optimize a deployed v0.5 recipe

### Speed depth evidence for the development ledger

The [sr-bench walkthrough](SCENARIOS.md#tune-speed-using-an-existing-sr-bench-development-run) consumes a native run directly. Use the ledger path below when your runner already exports graded development measurements and per-call observations, including turns after the opening request.

`recipe-lab loop --workspace "$WORK" plan-dev --base "$PARENT" --fast "$FAST" --depth-evidence depth.json --out-candidate candidate.yaml` considers one binary SystemOne signal on a priority recipe with reasoning, competing and fallback decisions. The proposed edit preserves the other rules and provider bindings. Add `--prefixes "$OTHER_PARENT_MEASUREMENT"` for independent guard repetitions.

The `recipe-lab-depth-evidence-v1` JSON object identifies the `base_config_sha256`, `public_model` and `signal`. Each entry in `observations` names its development `measurement`, `task`, `trace_sha256` (the exact requests JSONL), and integer `call`. Supply the observed `score` (0 to 1 or explicitly null), `matched_other_decisions` (conditional decisions other than the depth branch), and `fast_eligible` for that request. Keep the native observations and independently graded source runs so the plan can be replayed. The planner checks that the predicted old decision and backend agree with the recorded parent.

For the verified schedule, use the normal `--probe-receipt` flow. A depth receipt also binds the `parent_source_sha` and includes paired parent and candidate responses to the same wire request. Exercise a changed score, a protected higher score, competition, an unavailable score, a tool round trip and applicable capability limits. Record the observed score, other matched branches, backend eligibility, decision, selected backend, returned identity and completion state on each side. A plan that verifies the probe and reproduces a distinct guard can enter the [paired development schedule](#schedule-a-paired-development-experiment).

Use `recipe_lab/loop.py` from this checkout with Python and PyYAML. The lab
stores one append-only JSONL ledger per workspace. Keep the frozen source
configuration, task packet, per-task active configuration bindings, independent
grader output and DSH receipts at their recorded paths so another agent can
verify them. A failed task with a valid independent grade is a measurement.
A verifier error or an incomplete packet needs investigation before comparison.

Run the remaining examples from `tools/recipe-lab`. Set `WORK` to the
experiment workspace and create it before redirecting command output there.
An arm is one configuration or direct backend under evaluation. Import a
frozen DSH arm and save the returned measurement
ID. Use the configuration that was actually deployed. For the run layout,
`RUN/dsh/frozen.json` describes the task set, expected checks, pinned inputs,
runner and author limits. Each `RUN/dsh/runs/TASK/ARM/main/` contains
`protocol.json`, `provider.json`, `result.json`, `grade/score.json` and
`requests.jsonl`. Each `RUN/dsh/bindings/TASK/ARM.json` identifies the public
model, deployed source SHA-256, active runtime fingerprint and client window.
The importer supports a single replicate per packet. Each request needs a
dispatch and response receipt, including an error receipt for failed streams.
DSH's terminal `model_endpoint_error`, a reported runner or grader error,
or a verifier timeout invalidates the arm for quality comparison. On old ledgers, `next`
identifies the affected records under `excluded_measurements` and omits their
comparisons. A recovered request error or the task author's deadline can still
produce a valid measurement.

If the frozen task packet declares `split`, import uses that allocation.
An explicit `--split` must agree. For older packets without this field, pass
`--split holdout` for sealed evaluation data. The historical default is
`dev`. When reading existing ledgers, `next` withholds measurements and
comparisons that conflict with an allocation recorded in the frozen packet.

```bash
python recipe_lab/loop.py --workspace "$WORK" record \
  --run "$RUN" --arm speed --config "$CONFIG" --public-model "$PUBLIC" \
  --strong "$STRONG" --hypothesis "Measure tool-result traffic on the strong model"
python recipe_lab/loop.py --workspace "$WORK" next > "$WORK/next.json"
```

`next` exports development measurements, exact record IDs, per-task failed
checks, model counts, stop reasons and related routing cases. The
`strong_tool_results` count includes calls whose last input message has role
`tool` and whose selected model matches `--strong`. Supply additional
observations with repeated `--symptom` arguments. The frozen split option
`record --split holdout` keeps those measurements out of proposal exports.

### Choose a configuration and development task before starting DSH

Run `plan-space` on the installed v0.5 configuration to inspect the target
recipe, provider limits, active routing decisions, signal dependencies and
supported edits. This works before a dataset is available. An existing
hand-tuned config can serve as the first reference to reproduce. Supply other
configs with repeated `--reference` or inspect a repository catalog with
`--reference-library /path/to/config/recipes`. Use `--reference-entry PUBLIC CONFIG`
to disambiguate a multi-entry reference. A built-in template without providers
is listed with its selected recipe, decisions and declared model slots.
Its `bind_reference_template_models` status means you can inspect its structure;
bind the template to providers before transferring an executable change.
For deployed references, foreign model bindings are matched by upstream model ID.
When the model families differ, add
`--anchor-models SOURCE_STRONG SOURCE_FAST` to `plan-space` and `plan-dev`.
These are the reference config's provider names in role order. A supported
handoff is rendered against the current deployment's providers, limits and
routes. Check it with fresh gateway probes and development tasks.
For a parent with an existing tool window or first-tool model rule, supply its
pinned `--edit-space FILE` to generate the allowed variants through the same
editors as `propose --space`. Expanding an eligible fast window from 1–3 to
1–8 exposes counts 4–8. Complex or narrowing edits stay available for paired
gateway probes. Use `--anchor-entry PUBLIC CONFIG` for a multi-entry anchor.

```bash
python recipe_lab/loop.py --workspace "$WORK" plan-space \
  --config "$CONFIG" --public-model "$PUBLIC" --strong "$STRONG" --fast "$FAST" \
  --anchor "$HAND_TUNED" > "$WORK/space.json"
python recipe_lab/loop.py --workspace "$WORK" plan-dev \
  --base "$DEV_PARENT" --prefixes "$OTHER_DEV_PARENT" --fast "$FAST" \
  --anchor "$HAND_TUNED" --out-candidate "$WORK/candidate.yaml" > "$WORK/plan.json"
```

`plan-dev` includes the base record's tasks. Supply another `--prefixes` for
each additional verified development record of the same deployed configuration.
Its `prefix_evidence` shows the exact calls that matched, the original
successful backend replies, rate limits, missing data, token-fit estimates and
the union of model-wait intervals. `same_observed_predicate` groups references
that matched identical positions on the available inputs. A priority override,
model capability or effective request parameters can still change delivery.

The nominated anchor gets the first probe when it exposes a measurable change
on a naturally completed, officially passing parent. Other probes are ordered
by deployability and the fraction of parent wall time covered by complete
strong-model requests, with absolute model-wait time as a tie-breaker.
Multiple observations of a task contribute their combined exposure and elapsed
time. A task with a failed parent repeat cannot supply a latency ranking.
These times refer to the recorded trajectory. A new reply can change later
tool calls and task duration. `--out-candidate` writes the selected draft once
and requires a materializable recommendation. Omit it to obtain a diagnostic
report even when the planner recommends collecting a better prefix.

Read `next` in the report to find the candidate, task, affected calls and
next check. Probe the public gateway at the branch boundary and verify the
returned backend, tool behavior and qualification. Confirm a distinct passing
guard before funding matched DSH runs. If the target predicate misses, inspect
the visible history. If the predicate hits and the same model answers, inspect
priority and fit. After a paired run, `record` and `compare` as usual and rerun
`plan-dev`. It reads matching, verified dev comparisons. Use `--comparison ID`
for another same-parent comparison, and `--schedule-card FILE` to release
assigned measurements after their paired study completes. Compare latency
only for joint successes.

After probing, repeat `plan-dev` with the same inputs and
`--probe-receipt <public-probes.json>`, omitting `--out-candidate`. Bind the receipt
to the emitted configuration and the public model. Include the backend mode,
boundary dispatch roles, selected and returned models, actual tool round trip and
unsupported-input result. A passing fixture leads to a real-backend probe.
If the isolated instance used different upstream addresses or authentication,
provide its exact YAML with `--probe-config FILE`. The planner verifies its
fingerprint and compares the remaining configuration with the candidate.
A passing real probe leads to a distinct guard check or paired development
planning when the supplied parent measurements already reproduced that guard.

Once the plan recommends a paired development run, save the full `plan-dev`
JSON. In a `recipe-lab-schedule-v2` card, set `signature` and
`signature_sha256` to that report and its file digest, and set `candidate`
to the exact generated YAML. Omit the v1 window `space` field. Use the
recommended target, a verified guard, the hypotheses, the three decision
consequences, repeat count and whole-episode budget as in the paired example
below. `schedule` verifies the original development evidence and probe
before issuing the first assignment. Commands that use assigned development
measurements accept `--schedule-card` for the completed study.

An agent can edit one recipe mapping from the measured configuration and
compose a full candidate using the public entrypoint. For example:

```bash
python recipe_lab/loop.py --workspace "$WORK" propose --base "$SPEED_ID" \
  --recipe edited-recipe.yaml --out candidate.yaml
```

This command refuses a shared target and preserves its sibling recipes,
provider settings and entrypoints. Run the installed CLI and Router validators,
start a new gateway from the candidate, then re-run the independently graded
tasks with the same DSH settings. Use `record` on that run with
`--based-on "$SPEED_ID"` and a hypothesis explaining the change. The
comparison checks the frozen tasks, grader, harness, client window and
configuration outside the measured recipe.

```bash
python recipe_lab/loop.py --workspace "$WORK" compare \
  --baseline "$SPEED_ID" --candidate "$CANDIDATE_ID" \
  --disposition keep-baseline --reason "Explain the observed trade-off"
```

The comparison records strict wins and losses and includes durations only for
joint successes. Omit the disposition for a descriptive comparison. Use
`advance-to-holdout` to test a development candidate with no lost strict
passes and a measured quality or paired-time gain. Validate the final choice
on predeclared independent tasks against the direct and routed baselines.

### Measure task-level headroom before searching

Run a read-only analysis on a predeclared development matrix. An adapter for
the benchmark must audit each episode against its independent strict grade,
elapsed task wall time, frozen controls, actual policy and physical run
identity. It must distinguish an official task timeout followed by a valid
grade from a provider quota outage, broken judge or interrupted measurement.
Only episodes assessed as valid enter a complete matrix. Mark infrastructure
and grading failures in the source study, then rerun the missing cells. The
command verifies the normalized matrix and its fingerprint. The adapter is
responsible for the truth of its evidence assessments and fingerprints.

```bash
python recipe_lab/loop.py --workspace "$WORK" headroom \
  --scope dev --manifest "$WORK/dev/manifest.json" --mde-pair speed direct-glm \
  > "$WORK/dev/headroom.json"
```

The JSON manifest declares the complete expected matrix before normalization.
For example, `{"study":"pilot","split":"dev","tasks":["task-1"],"arms":["speed","direct-glm"],"repeats":[1,2],"scores":"rows.json","scores_sha256":"…"}`.
The sibling scores file is a JSON array. Each episode has these fields.

```json
{"study":"pilot","split":"dev","task":"task-1","arm":"speed","repeat":1,
 "strict":true,"seconds":14.2,"assessment":"valid","controls_sha256":"…",
 "policy_sha256":"…","episode_sha256":"…"}
```

Use lowercase 64-character SHA-256 digests. Pin the same task, grader, DSH
settings and comparable run conditions across arms and repeats for each task.
Pin the deployed fixed policy for each arm and a distinct physical execution
for each cell. Use the original run identity when a measurement is reused in
a report. Keep development and retired studies in separate manifests. Use
`--scope retired-holdout` only after formal retirement of that evaluation set.
The manifest and every row must agree with the selected scope. The command
writes exclusively to stdout and its output never enters the proposal ledger.

For repeated development loops, screen existing runs for routing opportunities
and expensive mandatory work before picking tasks. Use the same development
matrix format, with one or more fixed arms. Supply DSH start-to-exit time in
`seconds` and copy `termination` from the solver receipt. When available,
supply `model_requests` and `model_request_union_seconds` together. The
union counts overlapping model requests once within each episode.

```bash
python recipe_lab/loop.py --workspace "$WORK" screen-dev \
  --manifest "$WORK/dev/manifest.json" --max-seconds 600
```

The default reference is 600 seconds. Read `routing_candidates` for same-task
quality contrasts and timing differences on jointly successful repeats across
distinct policies. A slow direct backend and a fast successful recipe make
an informative candidate. Include a reverse contrast or a cheap control to
catch regressions. The output retains the observations behind each contrast;
a single pair is a hypothesis. A long run with only one measured policy goes
to `investigate_slow`. Check long generations, excess model turns and
agent-selected tests before reserving more whole episodes.

Exclusion for fixed cost requires an independent benchmark-adapter audit.
The manifest can map a task in `required_cost_receipts` to a sibling JSON
file using `{"file":"cost-A.json","sha256":"…"}`. The file binds `study`,
`split`, `task` and `controls_sha256` to the matrix. Include a plain-language
`requirement`, the frozen `protocol_sha256`, `required_for_every_attempt: true`,
`stage` (`required_solver_work` or `official_grading`) and `observations`
with distinct `run_sha256` and measured `seconds`. The adapter verifies the
underlying requirement and timings. When even the shortest observed mandatory
work exceeds the reference, the task is listed in `excluded_tasks`. A
successful solution faster than the claimed mandatory solver time flags a
conflict for review. Official grading is timed separately. A full suite
voluntarily launched by DSH is part of the observed strategy.

For a task with an independent grading-cost audit and no solver run, put its
task ID and audited controls hash in the manifest's `unmeasured_controls`
mapping. A manifest containing exclusively unmeasured tasks declares empty
`tasks`, `arms` and `repeats` lists and omits the scores file and hash.
The screen can evaluate the fixed cost while reporting zero solver attempts.

Freeze the chosen development cohort before evaluating candidates. Preserve
each attempt's grade and cost. DSH runs tasks to natural termination.

The fixed-policy comparison counts strict passes first. Among equally accurate
arms it minimizes total time on the cells all tied arms passed. Remaining ties
follow the declared arm order. L0 applies that rule once per task across all
repeats, then measures the hindsight selection against the best fixed arm.
The report lists gained and lost passes, paired wall time on joint successes
and a second L0 calculation constrained to preserve each baseline pass.
Treat these as empirical comparisons over the measured arms. A new recipe
needs an independent evaluation to establish a gain.

Each task includes the observed time range and sample coefficient of variation
over repeats, with a separate coefficient on successes. `--mde-pair` selects
two fixed arms in advance. The pilot latency MDE uses mean paired differences
per independent task and reports approximate seconds per jointly successful
task. Its reporting threshold is eight jointly successful tasks with at least
two joint repeats each and observable between-task dispersion. It imposes no
constraint on running experiments. Repeats improve within-task estimates and
do not increase the number of independent tasks. Stage and request-level
counterfactuals and a time breakdown need additional evidence beyond these
task scores.

### Edit and inspect a tool-result window

Start from a verified development measurement whose recipe already has this
window. Declare the public entrypoint, recipe, decision, bound signals and
allowed inclusive upper limits in a YAML file. For example:

```yaml
schema: recipe-lab-window-space-v1
action: tool_result_window_upper
base_config_sha256: "<SHA-256 of the measured parent config>"
public_model: vllm-sr/real-ab
recipe: speed
decision: early_tool_fast
lower_signal: observed_tool_results
upper_signal: beyond_early_results
allowed_upper: [3, 8]
```

The decision must conjoin the positive lower count signal with the negated
upper count signal. Both count `assistant_tool_cycle`. A window ending at
eight therefore uses `NOT count >= 9`. A shared upper signal, a mismatched
parent or an out-of-range value fails validation. Create the candidate at a
new output path. Use the actual CLI and Go Router validators on that file.

```bash
python recipe_lab/loop.py --workspace "$WORK" propose --base "$PARENT" \
  --space space.yaml --upper 8 --out candidate.yaml
python recipe_lab/loop.py --workspace "$WORK" signature --base "$PARENT" \
  --prefixes "$TRAFFIC" --space space.yaml --upper 8 --candidate candidate.yaml \
  > signature.json
```

`PARENT` identifies the deployed configuration to edit. `TRAFFIC` identifies
one compatible development measurement supplying the observed request
prefixes. They can differ. The report pins both identities and the candidate.
For each task it reports observable and unknown requests, the old and new
predicate hits, and the changed calls. Validate changes in model selection on
the serving gateway using identical requests for both configurations. Record
the effective parameters, selected and returned models, and any refusal or
fallback. A fixture backend can establish what SR sent. Independently graded
whole tasks with the deployed models are needed to measure quality and time.

### Change the model for an existing first-tool decision

Use a measured parent with a static decision for requests that offer tools,
allow tool use and contain no visible assistant history. Declare the deployed
signal names and the tool-capable backend names from the provider registry.
This action accepts the three guards shown by the signal names below as the
direct children of an AND with `on_unknown: no_match`. The recipe must already
enforce declared capabilities and known context limits. Each allowed backend
must declare text and tools (including the Router aliases `chat` and
`tool_use`) and positive context and output limits.

```yaml
schema: recipe-lab-first-tool-space-v1
action: first_tool_model
base_config_sha256: "<SHA-256 of the measured parent config>"
public_model: vllm-sr/real-ab
recipe: speed
decision: first_tool_turn
tools_signal: has_tools
disabled_signal: tool_disabled
assistant_signal: assistant_history
allowed_models: ["strong", "fast"]
```

Run `propose --base "$PARENT" --space opening.yaml --first-tool-model fast --out candidate.yaml`
through the same CLI. It changes the sole static `modelRefs[0].model` on that decision.
The Router sees the history supplied for each request. A resumed client that
omits earlier messages can satisfy the condition again. Validate the candidate
on the installed CLI and Router, then probe with and without tools, explicit
tool disabling and existing assistant history. A smaller destination can be
ineligible for a long request or a large output allowance. Include such a
request when checking the deployment's rejection or fallback behavior.
`signature` and v1 schedule cards verify a tool-result window edit.
For a candidate supported by `plan-dev`, use its verified report and
a v2 card as described in [planning](#choose-a-configuration-and-development-task-before-starting-dsh).

### Schedule a paired development experiment

Freeze a JSON experiment card with this shape. Paths to the space, candidate
and signature are relative to the card. Use real measurement IDs and hashes.

```json
{
  "schema": "recipe-lab-schedule-v1",
  "study": "tool-window-dev",
  "based_on": "<development parent record ID>",
  "signature": "signature.json",
  "signature_sha256": "<SHA-256 of signature.json>",
  "space": "space.yaml",
  "candidate": "candidate.yaml",
  "hypotheses": ["Earlier handoffs reduce elapsed time.", "Handoffs add repair turns."],
  "prediction": "The wider window preserves the solved storage task and helps the planning task.",
  "guard_task": "mvcc-lsm-compaction",
  "target_task": "production-planning",
  "decisions": {
    "advance": "Confirm the gain on further development tasks.",
    "revise": "Review the changed calls and repair turns.",
    "drop": "Retain the parent and record the task regression."
  },
  "budget_episodes": 4,
  "repeats": 1
}
```

Run `python recipe_lab/loop.py --workspace "$WORK" schedule --card card.json`.
The first call pins the card and enrolls the study. When the status is
`run`, the adapter saves the returned `next.assignment_id` in the new
run's binding **before launching the worker**. Give the task and exact
configuration to the benchmark's normal runner. Let the agent
finish under the task's original conditions. Import the independently graded
run with `record`. For an incremental single-task packet, use the original
frozen task and controls; the card and deployment fingerprint bind the task
to this experiment. Return the receipt with `schedule --card card.json
--attempt attempt.json`. For example:

```json
{"task":"mvcc-lsm-compaction","repeat":1,"role":"baseline",
 "assignment_id":"<ID from this slot>","measurement":"<record ID>"}
```

For an audited endpoint, runner or independent grader failure, supply
`invalid_run` (the run root containing `dsh/`) and its physical `arm`
in place of `measurement`. The adapter must capture same-boot start and
end receipts bound to the assignment, frozen controls and worker outcome.
Invalid attempts consume a budget slot. Their elapsed worker time has a
separate timing source from DSH's solve time. Preserve incomplete evidence
for investigation. A retry gets its own assignment if budget remains.
Reimporting the same receipt is idempotent. The command rechecks the stored
measurements before returning the next slot. Its budget applies to whole
attempts across both configurations.
The paired summary keeps strict wins, losses, joint-success time and total
attempt time distinct. A development advance calls for independent evaluation.

Read back the original card and its audited decision with the same workspace.

```bash
python recipe_lab/loop.py --workspace "$WORK" decision-summary --card card.json
python recipe_lab/loop.py --workspace "$WORK" next --schedule-card card.json
```

Both commands leave the schedule untouched. Before enrollment the summary
shows the prediction and fixed-prefix changes. During a run it reports progress.
After a terminal decision it returns the parent, candidate and evidence hashes,
fresh per-task strict results, paired elapsed time, invalid costs, next action
and conditions for reopening the experiment. Routing counts describe each
observed trajectory. The request-level comparison requires matching body bytes
and one completed backend observation per arm within the same task and repeat.
An empty intersection leaves the backend change unknown.

To investigate a strict failure with runner termination `native_output_limit`,
read `terminal` on the development case or completed schedule attempt. It
identifies the final client call, explicit request limits, finish reason, and
model-reported completion and reasoning tokens. Reasoning is included in the
completion total. An empty `explicit_request_limits` means the receipt has no
usable explicit limit. Missing finish or usage fields and the unobserved
upstream limit appear as JSON `null`. The case reference gives checks to make
before proposing an edit. A length finish alone does not trigger it.

Supply one `--schedule-card` per study to pass completed decisions to the next
proposer. The unpaired list always withholds assignment-bound measurements.
A changed source receipt invalidates readback.

### Check whether a matched-budget search comparison can start

Use `search-preflight --protocol protocol.json` through the same CLI before
comparing an existing skill, random search and a guided agent. A protocol
specifies `schema: recipe-lab-search-preflight-v1`, the study, development
parent (`based_on`), prefix record (`prefixes`), sibling space file and its
SHA-256, integer `seed`, and an identical budget for each method
(`episodes` and `proposal_model_tokens`). Proposal tokens measure the
optimizer's work. They do not set limits on the task-solving agent.

The `split_plans` object has a `dev` and a `holdout` entry. Set either to
`null` until its split is allocated. A frozen entry supplies a sibling JSON
`file` and its `sha256`. Each split file contains only its
`schema: recipe-lab-search-split-v1`, study, split, the ordered methods
`["existing_skill", "random", "guided"]`, and an `instruction_sha256` array
of unique task fingerprints. Use common allocations for all three methods.
The preflight checks overlap with the named development parent and the other
allocation. The study owner must also exclude tasks used for earlier tuning.

The existing protocol and command remain valid. In report v2 a historical
prefix result with both split plans can reach `protocol_ready`. Top-level
signatures still refer to the original `prefixes` record. Earlier v1 reports
called this paperwork state `ready_for_comparison`.

To measure action differences on the target workload, pass one or more
verified development records and the number of distinct candidates each
method may try. For example, with two trials per method:

```bash
python recipe_lab/loop.py --workspace "$WORK" search-preflight --protocol protocol.json \
  --target-prefixes "$TARGET_DEV_RECORD" --candidate-trials-per-method 2
```

The target receipts must belong to the frozen dev allocation and the exact
parent configuration identified by `based_on`. The historical prefix arm
can retain its original configuration. The report identifies missing tasks,
unknown coverage and the number of distinct new predicate signatures.
Once the target records cover the allocation, the same dev instructions can
serve as their own prefix source. The holdout is checked by allocation hashes.
The seeded representatives in `comparison` then use the target prefixes.
If two trials cover both distinct new actions, all methods can try the same
set. Testing selection among final configurations requires more distinct
choices than trials. An omitted trial count leaves that question open.

When the allocation hashes raw instructions and the solver hashes
rendered prompts, add `--target-dev-manifest PATH` and
`--target-dev-manifest-sha256 SHA` from the independent dev packet export.
Use `schema: recipe-lab-target-dev-manifest-v2`, the same `study`, `split: dev`,
a `template` object with `file` and `sha256`, and a `tasks` list of objects
with `instruction_sha256` and `instruction_file`. All file paths are relative
to the manifest. The UTF-8 template accepts the placeholder `{instruction}`;
it preserves line endings and whitespace in both files. The frozen task packet
must record the same template path and digest in `solver_files`. Preflight
reconstructs and checks each rendered prompt against the measured hash.
The legacy SWE v1 manifest uses `prompt-template.txt` and per-task `repo` and
`statement_file` to reconstruct the original solver's prompts. For solvers
that receive the raw instruction directly, omit the manifest and preflight
compares the instruction hashes directly.

`target_actions_distinguished` describes observed predicates. Use the
study's separate grader receipts and existing budget for execution decisions.
A preflight reports `search_executed: false`. It leaves
`paid_search_authorized` null because budget authorization belongs to the
study owner.

For frozen development objectives, historical skill replay, prospective
three-method enrollment and final scoring see [Search and evaluation](SEARCH.md).

## Live verdict against sr-bench runs

`verify` drives sr-bench end to end for one candidate on a frozen split;
`live-verdict` replays the verdict for two runs that already exist. Both
record one live-tier ledger entry whose `bench_refs` cite the sr-bench run
IDs - the evaluation numbers themselves are never copied into the ledger.

```bash
# end to end: preview the routing distribution, then AB/BA pairs
recipe-lab verify --spec spec.yaml --split dev --cases splits/dev.jsonl \
  --candidate candidate.json --ledger ledger.jsonl \
  --srbench-bin vllm-sr --srbench-url http://127.0.0.1:9000 \
  --srbench-store ~/.srbench-store --candidate-targets target-a,target-b

# verdict for two recorded runs (guard floors from spec.gates, A3)
recipe-lab live-verdict --spec spec.yaml --split dev \
  --baseline-run baseline-id --candidate-run candidate-id \
  --candidate candidate.json --ledger ledger.jsonl --json
```

The three-state rule (INCONCLUSIVE > FAIL > PASS): incomplete, timed-out or
hash-mismatched evidence is INCONCLUSIVE; a breached guard floor fails the
verdict even when the objective metric wins; PASS requires the objective
improvement, every declared guard pass, every cost constraint within its
bound, and enough paired samples. `verify_promotion_evidence` replays a
recorded PASS from its ledger record for audit.

### Score one deployed candidate

`score-candidate` scores an authored v0.5 source YAML that is already
deployed against a saved baseline run. A coding agent submits its own
proposal through it (`--proposer agent:<id>`); no spec is needed.

```bash
recipe-lab score-candidate --workspace lab/ \
  --candidate config.yaml --proposer agent:claude-loop \
  --baseline-run baseline-id --baseline-target single \
  --candidate-target recipe --active-runtime-hash <sha256> \
  --guard-case case-17 --min-speedup-pct 5
```

The receipt is one JSON line and is saved under `<workspace>/sr-bench`.
`--candidate-run <id>` re-opens a submitted run instead of dispatching it
again. The exit code is 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
