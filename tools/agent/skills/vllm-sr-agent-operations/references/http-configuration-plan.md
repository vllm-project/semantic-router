# Prepare a configuration change over HTTP

This example prepares one decision-description change for review through the
Router management API. It stops after a config plan; nothing is saved or
activated. The [configuration workflow](configuration-loop.md) covers a later,
separately authorized activation.

Use Bash, curl and Python 3 in an empty working directory. Use a disposable
Router with the secret-free fixture below, a converged persisted/active
configuration, and no concurrent writers. Follow the
[installation skill](../SKILL.md) to start a new stack; do not restart an existing
stack to try this example. Set `ROUTER_ORIGIN` to its management origin. If bearer
auth is enabled, supply `ROUTER_TOKEN` through your environment with the built-in
`viewer` role; it permits discovery, reads and planning, but blocks writes.

A viewer's config read may redact secrets. Do not use that redacted document as
a whole-config replacement for a deployment containing real credentials. This
example has no provider secrets and tests preparation, not backend delivery.

## 1. Use a fixture with something to preserve

For a new disposable stack, use this as `config.yaml`. Both backend addresses
are synthetic; no model server or inference request is needed for this task.
The second decision makes accidental replacement of the decision list visible.

```yaml
version: v0.3
listeners:
  - name: example-http
    address: 127.0.0.1
    port: 8888
    timeout: 120s
providers:
  defaults:
    model: general-model
  models:
    - name: general-model
      provider_model_id: general-model
      backend_refs:
        - name: general-local
          endpoint: 127.0.0.1:18081
          protocol: http
          provider: vllm
          weight: 100
    - name: specialist-model
      provider_model_id: specialist-model
      backend_refs:
        - name: specialist-local
          endpoint: 127.0.0.1:18082
          protocol: http
          provider: vllm
          weight: 100
routing:
  strategy: priority
  decisions:
    - name: default-route
      description: Generated during setup to route all requests to the default model.
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: general-model, use_reasoning: false}]
    - name: specialist-route
      description: Preserve this unrelated decision.
      priority: 50
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: specialist-model, use_reasoning: false}]
```

For a running disposable stack, proceed only if its canonical config has these
same two decisions and the expected old description. Always build the candidate
from the fresh API response, not from this launch file, which may omit defaults.

## 2. Discover the contract and record the baseline

Run the following blocks in the same Bash session. These requests read data;
`curl --fail-with-body` stops on HTTP errors rather than treating them as success.

```bash
set -euo pipefail
export ROUTER_ORIGIN="${ROUTER_ORIGIN:-http://127.0.0.1:8080}"
request() {
  if [[ -n "${ROUTER_TOKEN:-}" ]]; then
    set -- -H "Authorization: Bearer $ROUTER_TOKEN" "$@"
  fi
  curl --fail-with-body --silent --show-error "$@"
}

request "$ROUTER_ORIGIN/ready" > ready.json
request "$ROUTER_ORIGIN/api/v1?capability=config" > discovery.json
request "$ROUTER_ORIGIN/openapi.json?path=/api/v1/config/plan&method=POST" > plan-contract.json
request "$ROUTER_ORIGIN/api/v1/config/schema?view=section&path=routing.decisions" > decision-schema.json
request "$ROUTER_ORIGIN/api/v1/config/hash" > before-hash.json
request "$ROUTER_ORIGIN/api/v1/config/versions" > before-versions.json
request -D baseline.headers "$ROUTER_ORIGIN/api/v1/config" > baseline.json
```

Inspect discovery and the filtered OpenAPI: the plan request is an object with
`yaml` (a string containing the candidate document) and `mode` (`replace` here).
It is not a bare candidate object or a JSON-encoded string containing that object.
`GET /api/v1/config` returns the canonical document as JSON, with an ETag header.

## 3. Prepare and inspect the local candidate

```bash
python3 - <<'PY'
import copy
import json
from pathlib import Path

load = lambda name: json.loads(Path(name).read_text())
assert load("ready.json")["ready"] is True
identity = load("before-hash.json")
hashes = [identity.get(k) for k in (
    "source_config_hash", "generated_runtime_hash", "active_runtime_hash")]
assert all(hashes) and len(set(hashes)) == 1
assert identity["activation_status"] == "active"
baseline = load("baseline.json")
decisions = baseline["routing"]["decisions"]
assert len(decisions) >= 2
matches = [i for i, d in enumerate(decisions) if d["name"] == "default-route"]
assert len(matches) == 1
index = matches[0]
old = "Generated during setup to route all requests to the default model."
new = "General requests use the default model"
assert decisions[index]["description"] == old
candidate = copy.deepcopy(baseline)
candidate["routing"]["decisions"][index]["description"] = new
for name, document in (("baseline.pretty.json", baseline), ("candidate.json", candidate)):
    Path(name).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
# JSON is also a YAML document: preserve the full config without another parser.
Path("plan-request.json").write_text(json.dumps({
    "yaml": json.dumps(candidate), "mode": "replace"}) + "\n")
print(f"default-route.description: {old!r} -> {new!r}")
PY

diff -u baseline.pretty.json candidate.json > candidate.diff || [[ $? == 1 ]]
cat candidate.diff
```

Review the diff: only `default-route.description` should change. The complete
second decision and all other fields must remain. A merge request also replaces
lists wholesale, so a patch containing only the target decision could remove
others. This example deliberately submits the complete candidate with `replace`.

## 4. Check the exact candidate without applying it

```bash
request -H 'Content-Type: application/json' \
  --data-binary @plan-request.json \
  "$ROUTER_ORIGIN/api/v1/config/plan" > plan.json
request "$ROUTER_ORIGIN/api/v1/config/hash" > after-hash.json
request "$ROUTER_ORIGIN/api/v1/config/versions" > after-versions.json

python3 - <<'PY'
import json
from pathlib import Path

load = lambda name: json.loads(Path(name).read_text())
plan = load("plan.json")
etags = [line.split(":", 1)[1].strip()
         for line in Path("baseline.headers").read_text().splitlines()
         if line.lower().startswith("etag:")]
assert len(etags) == 1 and plan["current_etag"] == etags[0]
assert plan["valid"] is True and plan["changed"] is True
assert plan["mode"] == "replace" and plan["candidate_etag"]
assert load("before-hash.json") == load("after-hash.json")
assert load("before-versions.json") == load("after-versions.json")
print("Baseline:", plan["current_etag"])
print("Candidate:", plan["candidate_etag"])
print("Server plan passed; observed config identities and history unchanged.")
print("Prepared for review only. Nothing saved or activated.")
PY
```

Stop here. Retain `plan-request.json`, `candidate.json`, the reviewed diff and
responses together: the plan returns identities, not a field-level diff. The
local comparison establishes the intended change; `valid: true` alone does not.
ETags identify bytes, so do not compare the candidate ETag with a hash of your
locally formatted JSON and assume a mismatch means different configuration.

A successful config plan includes the mutation path's validation and hot-reload
compatibility checks, not activation or a routed-request test. The read/plan and
before/after checks are separate observations, not a transaction or continuous
monitor. If the baseline differs or a check fails, stop and inspect the response;
do not apply the candidate or infer that the failure is an API defect.

## Recipe validation is a narrower alternative

An agent may instead discover `GET /api/v1/config/recipes/default` and
`POST /api/v1/config/recipes/validate`. For the same description-only task,
construct the request as `{"name": "default", "routing": <complete routing block>}`,
keeping every other decision and field. Omit read-view metadata such as
`entrypoint_source`, `source` and `deletable`.

The default recipe's read view includes a synthesized recipe-level description.
A nonempty recipe-level `description` is rejected on mutation; omit it. Named
recipes can set their recipe-level description. This restriction does not apply
to decision descriptions within `routing`. An explicitly supplied `entrypoints`
field can change entrypoint bindings: omitting it preserves them for this task.

Recipe validation returns `valid`, `name` and `action`; it returns neither a
candidate ETag nor the full-config plan's hot-reload check. Record the GET ETag
as the read baseline, preserve the exact submitted payload, and compare the
complete intended result separately. Do not describe recipe validation as a
config plan or as proof that the candidate is ready to activate.
