# Routing and model evaluation details

Use the [operations skill](../SKILL.md) for discovery, installation, and initial
verification. Set management origin, inference base URL, Recipe, and entrypoint
from the selected deployment. Do not copy a model name from an example.

## Routing evidence and branch coverage

`route preview` evaluates Router signals and decisions without a generation
backend call. It can use learned signals or stateful selection; do not assume
all previews are deterministic. Request a trace and assert the expected Recipe,
decision, and algorithm for representative inputs.

The installed CLI can verify ordinary branches without a source checkout. For
example, after checking the exported `mom-v1` manifest still declares this
`balance` simple-text case, use the deployed entrypoint and listener origins:

```bash
PROMPT='Answer briefly with a short definition of a readiness probe.'
vllm-sr route preview \
  --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --token-env VSR_MGMT_TOKEN --timeout 300 \
  --prompt "$PROMPT" --trace --json > preview.json

vllm-sr route probe \
  --config config.yaml --base-url "$INFERENCE_BASE_URL" --model "$ENTRYPOINT" \
  --api-key-env OPENAI_API_KEY --timeout 300 --prompt "$PROMPT" \
  --max-completion-tokens 8192 \
  --expect-recipe balance --expect-decision simple \
  --expect-algorithm multi_factor > probe.json
```

These credential flags name environment variables; they never take token values.
Use the selected deployment's actual variable names when different. Management
and inference credentials are independent, and an unset variable sends no bearer
header. The 300-second budget matches the exported `mom-v1` probe policy; cold
learned signals and long-context cases can exceed the CLI preview default of
15 seconds. Choose a timeout appropriate to the case and retain timeout failures
as evidence. Inspect `preview.json` for the expected Recipe, decision, algorithm,
selection status, signal errors, and trace; HTTP success alone is not a route
assertion. Repeat with other cases from the selected manifest.

The completion budget is separate from the HTTP timeout. `8192` is an example,
not a guarantee: choose a limit that fits the backend context window after the
actual input, and allows both reasoning tokens and the final answer. Omitting
`--max-completion-tokens` leaves the backend default unchanged. A reasoning model
can exhaust that default while returning HTTP 200, correct routing headers,
`content: null`, and `finish_reason: length`. Keep this failed receipt; repeat
with an explicitly larger supported budget when the test scope permits. Do not
disable reasoning to hide incomplete delivery or claim the first request passed.
This explicit budget applies to CLI probes. Ordinary Playground requests omit
token-limit fields and use backend defaults; when those defaults truncate an
answer, inspect the backend generation settings and available context budget.

`route probe` makes a real OpenAI-compatible request through Envoy. Its receipt
contains status, latency, routing headers, response body, and assertions. Use
`--expect-selected-model` for the routing identity and `--expect-response-model`
for the backend's separately calibrated top-level `model` field. An absent or
unstable backend identity limits that assertion, not the need to verify real
delivery. Use installed help for supported Recipe and decision assertions.

For an expected successful HTTP status, `response.body.delivery` must also pass:
every choice needs final assistant text, a structurally valid function tool call
with JSON object arguments, or an explicit refusal, plus a recognized terminal
finish reason. Reasoning alone, empty or malformed choices, and any `length`
finish fail, including partially generated answers. A `content_filter` finish
passes only with an explicit refusal. The assertion records each finish reason
and delivery kind without repeating reasoning; the original response remains in
the receipt. A refusal or tool call proves delivery, not answer quality or tool
execution. Explicit non-2xx `--expect-status` cases check the expected rejection
without demanding an assistant completion.

Build cases for every relevant remaining branch, boundaries, fallback, and
unsupported-input behavior. Candidate-selection algorithms can select from a
pool, so test membership or policy outcomes appropriate to the contract instead
of demanding one fixed winner. Failed delivery is not a passing route probe,
and an excluded baseline lane is not covered by a derivative's passing cases.

Preserve the verified bundle and its original probes. Export copies exact bytes:
a probe manifest can retain authored repository-relative `routing_assets` paths.
Treat an exported manifest as test inputs and expected assertions; it does not
install a runner or make those paths valid in the export directory. Use `--prompt`
for plain `query` cases or `--messages` for complete OpenAI message arrays. Do not
use `display_prompt` as a substitute for the real payload, discard tool fields,
or skip materializing declared padding, generated text, and image fixtures while
claiming that case passed.

Current route CLI commands have no top-level `tools` or `tool_choice` options.
For those shapes, use the discovered Router preview HTTP operation and its
OpenAPI request schema. Write the exact request, including the deployed `model`,
to `preview-request.json`; export the nonsecret `PREVIEW_URL` as that discovered
operation URL, including its trace option when supported. This standard-library
request keeps the token out of process arguments and preserves error responses:

```bash
python3 - <<'PY'
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

request = urllib.request.Request(
    os.environ["PREVIEW_URL"],
    data=json.dumps(json.loads(Path("preview-request.json").read_text())).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
token = os.environ.get(os.environ.get("ROUTER_TOKEN_ENV", "VSR_MGMT_TOKEN"), "")
if token:
    request.add_header("Authorization", "Bearer " + token)
try:
    response = urllib.request.urlopen(request, timeout=300)
except urllib.error.HTTPError as error:
    response = error
with response:
    Path("preview-response.json").write_bytes(response.read())
    print("preview HTTP status:", response.status)
    raise SystemExit(0 if 200 <= response.status < 300 else 1)
PY
```

The same payload shape must reach the inference API when testing actual delivery;
use its discovered request contract and the inference credential, not the
management token. Keep tool execution and backend context limits within the
requested test scope. Full manifest materialization and repository-native
conformance are optional contributor workflows requiring their own installed
tools or checkout; they are not fresh-install prerequisites. If using that
harness, inspect its path/filter options and explicitly select the deployed
config/DSL, Recipe, and entrypoint. Keep adapted probes separate, record the case
IDs and actual assertions exercised, and report baseline and adapted coverage.
Do not rewrite digest-bound resources or claim full bundle conformance from a
filtered result.

## Requested workloads and benchmarks

Run benchmarks when requested or when the agreed optimization objective requires
them. Discover routing workloads with `benchmark catalog` and the installed
workload validation/run help. Compare paired baseline and candidate runs using
the same inputs and preserve their manifests and receipts.

For model-quality evaluation, begin with:

```bash
vllm-sr benchmark intelligence list
vllm-sr benchmark intelligence plan --help
```

Use the exact dataset and runner revisions reported by the installed catalog.
Materialize frozen sources as the plan requires, keep them clean, and supply
credentials through named environment variables. Do not substitute rolling data
or change the suite's modality/subset under the same score label. Use the
installed `run` contract only after the plan's prerequisites are satisfied.

Live exact-answer grading reports incomplete final answers as unavailable for
grading; reasoning output remains observed evidence rather than a final answer.

Physical and virtual models follow the same evaluation contract. Evaluate a
virtual model through its actual routed endpoint so route failures, retries,
model mix, latency, and cost are observable. Never synthesize its score from
member-model scores. A partial run or `--sample-limit` supplies smoke evidence,
not a full Intelligence score.

For optimization, capture the baseline first, make one coherent change, and
compare the agreed quality, cost, latency, and reliability gates. Retain the
candidate only when the evidence supports the objective without violating hard
constraints; otherwise use the [configuration recovery path](configuration-loop.md).
Keep raw outputs private and preserve secret-free receipts with runtime and
config identity so the comparison can be reproduced.
