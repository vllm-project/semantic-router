# vSR for OpenCode

This package is under development for OpenCode 2.2.0. The foundation provides a
decision-runtime client, validated options, session-scoped decision metadata and
versioned RPC diagnostics. The six apps and terminal UI are not wired yet.

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
        "runtimeTokenEnv": "VSR_RUNTIME_TOKEN"
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
`"done"`. These controls reserve the eventual six-app contract; they do not imply
that the corresponding apps are implemented at this stage. Unknown options are
rejected rather than silently ignored. Retention defaults to 500 decisions per
session and 64 sessions, with least recently used session eviction.

The HTTP client serializes requests and includes queue time in deadlines. A
network failure opens a 60-second breaker; HTTP errors, malformed replies and
caller cancellation are reported separately. Discovery tries
`/v1/systemone/models`, falling back to `/v1/models` only on HTTP 404. The selected
model and its advertised question types are checked before use. Responses over
2 MiB or missing requested answers are rejected. Cancelling a request stops the
local wait; it cannot guarantee a remote model stops computing.
