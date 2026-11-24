## Reports, manifests and promotion drafts

The ledger is the single replay source for every human-facing artifact: the
same ledger renders byte-identically, and mixed dataset hashes are flagged as
not directly comparable.

```bash
recipe-lab report --ledger ledger.jsonl --output report.md
recipe-lab provenance --ledger ledger.jsonl --output manifest.json
recipe-lab record-draft --ledger ledger.jsonl --output promotion-draft.md
recipe-lab next --spec spec.yaml --ledger ledger.jsonl --output ctx.md
```

`report` also renders explicitly selected receipts from outside the ledger
(`--evidence path.json`, repeat in workflow order, `--background` for the
operator's own context); unknown or malformed schemas are refused and the
output is published atomically - a failed render never replaces an existing
report. `provenance` emits a `recipe-lab-run-v1` manifest whose vocabulary
aligns with the upstream provenance schemas (record IDs, ledger hash,
bench_refs). `record-draft` produces a docs/records draft from the latest
PASS live record for human review; drafts are never merged automatically.
`next` exports the agent context with the untrusted-material header and the
citable refs described in the report section above.

## Speed depth receipts

`recipe-lab suggest-depth` prints and saves one
`recipe-lab-sr-bench-depth-v1` receipt: `status` (`proposed` / `keep-parent`
/ `inconclusive`), the run and target identity (`dev_run`, `parent_target`,
`fast_target`), `parent_sha256`, `deployed_source_sha256`,
`parent_runtime_hash` and `output_policy`, the per-case `observations`
(`role` `move` / `protect` / `skip`, both finish reasons, the comparable
opening-request hash and whole-task seconds; case IDs only, never task
text), and when a threshold separates the observations `before`, `after`,
`candidate` and `candidate_sha256`. `report --evidence` renders the receipt
in supplied order; the saved receipt never changes shape even when Markdown
delivery fails (`report_error` on the receipt, evidence still on disk).

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
change. Record the decision and actual backend for each. Management preview
counts tool results but omits their text from SystemOne classification; replay
representative agent histories through the public endpoint before using preview
results to choose a recipe.
For a semantic signal guarded by a tool-count threshold, measure classification
time on requests below the threshold too. The Router evaluates
referenced signals before it combines their predicates, so a rarely selected branch
can add classification time to most requests.

When moving a Speed draft to the pinned `vllm-sr/vela3-fast` 3.0 package,
check its `noul` questions. Put the meanings of yes and no
in `question.instructions` and omit described `question.choices`. This
package rejects descriptions on the individual binary options. Inspect the
SystemOne answer for an actual score and check the backend reached by the
public request. A successful chat response can hide a rejected classifier
request if the recipe falls back to another decision. Recheck the existing
thresholds with representative examples when switching models.
Find request examples in the [maintained probes](../../config/recipes/built-in/latest/mom-v1/probes.yaml).
Measure answer quality and task completion time against both direct deployments
before claiming an improvement.

Run the offline regression suite with
`python -m pytest -q tools/recipe-lab/tests/test_speed_draft.py tools/recipe-lab/tests/test_user_report.py`.
To exercise the generator's command-line path with your installed validator, set
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
