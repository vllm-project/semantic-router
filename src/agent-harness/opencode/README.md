# vSR for OpenCode

This package is under development for OpenCode 2.2.0. Router and Escalate are
implemented with configurable model tiers, session-scoped decision metadata and
versioned RPC diagnostics. Guard, Redact, Shield, Check and the terminal UI are
later implementation stages.

From this directory, install dependencies and check the package:

```bash
npm ci
npm run check
npm test
npm pack
```

For development, configure a local package directory in `opencode.jsonc`:

```json
{
  "plugins": [
    {
      "package": "/absolute/path/to/semantic-router/src/agent-harness/opencode",
      "options": {
        "runtimeUrl": "http://127.0.0.1:8790/v1/decisions",
        "runtimeModel": "",
        "runtimeTokenEnv": "VSR_RUNTIME_TOKEN",
        "routing": {
          "tiers": [
            { "model": "your-provider/light", "efforts": ["low", "high"] },
            { "model": "your-provider/middle", "efforts": ["low", "high"] },
            { "model": "your-provider/strong", "efforts": ["low", "high"] }
          ]
        }
      }
    }
  ]
}
```

Start an existing `vllm-srun` runtime separately. `runtimeModel` selects one model
when an endpoint serves several. Remote URLs use the same `/v1/decisions` API.
Set the named environment variable in the OpenCode server's environment if the
endpoint needs a token. Credentials, URL query tokens and embedded URL passwords
are not accepted as options. No token is returned in diagnostics or decision
snapshots. Redirects are refused so authorization cannot move to another host.
A remote decision runtime receives the text supplied for classification.

The server registers RPC `vsr.opencode.v1` with `snapshot`, `diagnose` and
`feedback` methods. `diagnose` explicitly queries runtime capabilities; loading
the package alone sends no content to a model. Import the exported `./rpc`
definition to call these through an OpenCode client. Snapshots are scoped to the
requested session. Only decision metadata is retained, not raw prompts, commands
or output. Decision counts and feedback are currently memory-only and bounded
by retention settings; durable calibration is a later stage. A new server has a
new snapshot instance ID, allowing clients to discard obsolete revisions.

Options also include `apps`, `waits`, `maxDecisions` and `maxSessions`. Apps default
to enabled except Check; waits are milliseconds (`1000`, `2000`, `3000`) or
`"done"`. Controls for apps assigned to later stages reserve their eventual
contract. Unknown options are
rejected rather than silently ignored. Retention defaults to 500 decisions per
session and 64 sessions, with least recently used session eviction.

Router applies to new `general` subagent calls. Configure three distinct models
in ascending capability order from one provider, and use that ladder for the
main session's model too. Replace the example IDs with models enabled in your
OpenCode provider configuration. Optional `efforts` lists name supported native
variants in ascending order; omit them to use each model's default. Variants
must appear in OpenCode's model catalog. Explicit child models, resumed children,
agents with their own model and models outside this ladder retain their choices.
The main session is never changed. A child receives its selection at creation;
provider retries retain that selection.

Difficulty and reasoning-step scores use a 0–2 scale. Defaults are
`easyBelow: 0.65`, `hardFrom: 1.5`, `minConfidence: 0.5` and
`downgradeConfidence: 0.6`. `routing.goal` is `"balanced"`, `"save"` (uses the
ordinary confidence threshold for downsizing) or `"quality"` (no downsizing).
Low confidence, invalid/disabled models or unavailable classification preserve
the host's original choice. `routing.deadlineMs` defaults to 1500 and includes
queue time. Tasks over 32,000 UTF-16 code units skip classification.

Escalate classifies admitted human prompts when their turn actually executes.
Wrong-answer/different-answer probability at least `feedbackThreshold: 0.7`, or
repeat probability at least `repeatThreshold: 0.8`, adds a re-check reminder.
The next eligible child consumes the escalation once, including simultaneous
launches. Its tier is one above the ordinary pick, capped at the strongest tier
and never below the parent's tier. Its effort rises one step and, on the same
tier, never below one step above the parent's declared effort. At the ceiling it
stays at the highest supported effort. If task classification is unavailable
after pushback was detected, the parent's tier supplies the baseline. Queued
prompts and synthetic background messages do not inherit another turn's
escalation. Interrupting/deleting a session cancels pending local requests.

Snapshots distinguish the planned model from `observedModel`, child session and
observed token/cost facts read after completion. Missing observations remain
absent; a failed spawn is recorded separately. No estimated savings are claimed.
Prior human text is held transiently in bounded memory for repeat detection;
it is never included in decision records or diagnostics. Classification sends
that text and child tasks to the configured runtime.

The HTTP client serializes requests and includes queue time in deadlines. A
network failure opens a 60-second breaker; HTTP errors, malformed replies and
caller cancellation are reported separately. Discovery tries
`/v1/systemone/models`, falling back to `/v1/models` only on HTTP 404. The selected
model and its advertised question types are checked before use. Responses over
2 MiB or missing requested answers are rejected. Cancelling a request stops the
local wait; it cannot guarantee a remote model stops computing.
