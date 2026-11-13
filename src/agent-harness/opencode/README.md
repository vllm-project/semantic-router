# vSR for OpenCode

This package is under development for OpenCode 2.2.0. Router, Escalate, Guard,
Redact and Shield are implemented, with session-scoped decision metadata and
versioned RPC diagnostics. Check, the terminal UI, setup and durable calibration
remain later implementation stages.

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

Guard covers OpenCode's built-in `shell` tool. It decomposes shell chains while
preserving quoted text, substitutions, heredocs and pipeline boundaries. Fixed
rules flag catastrophic deletion, disk/device writes, recursive system-wide
permission changes, fork bombs and downloads fed to interpreters, including
common wrappers and nested shell strings. These rules stay active without a
runtime. They are a bounded shell analysis, not a complete shell sandbox.

Commands exceeding 32,000 UTF-16 code units or 12 parts require native approval.
Other parts are classified sequentially within one shared budget. A review score
at least `guard.reviewThreshold` (default 0.9) requires native approval before
execution. Existing native deny/ask rules remain in force. Rejection or
interruption prevents execution, and snapshots distinguish held, approved and
declined commands. A command that was approved and exits nonzero remains
approved. Checks and outcomes are scoped to session, assistant message and call.

`waits.guard` defaults to 3000 ms. With bounded waits, unavailable classification
records a skipped check and leaves native permissions in control; fixed rules
and coverage limits still require approval. `"done"` uses a 60-second maximum
and requires native approval for any part it could not check, including offline,
malformed, unsupported and timed-out responses. Cancellation aborts the local
decision request and prevents the pending command from running. Saturating the
bounded pending-check capacity rejects new shell attempts until calls finish.
Manual shell sessions and commands executed through other custom tools are
outside this adapter's coverage. Classifier framing and pure shell rules are
ported locally from the Claude plugin, with source revisions in the code.

Redact covers text returned by native `read` and `shell`, including nonzero shell
exits and background shell notifications. Fixed rules mask known credential
formats, private keys, authorization headers, password-bearing URLs and config
assignments before text reaches either the decision runtime or generation model.
Numbered `read` lines retain their prefixes while their credential values are
masked. Other PII uses the runtime's `pii` preset and `redact.piiThreshold`
(default 0.7). Dates and domain names stay visible, and IP-like version numbers
are preserved. Runtime codepoint spans are validated against the exact input,
converted to UTF-16 offsets, expanded to complete ASCII words where needed and
merged over overlaps. Invalid spans make coverage unavailable.

Shield covers text returned by `read` and `webfetch`. It frames classifier input
as file/page contents and uses `shield.injectionThreshold` (default 0.9). Flagged
content is delivered as a quoted JSON string with untrusted-data guidance.
Shell output is not shielded, and web output is not redacted, matching the
baseline tool coverage. Other custom tools and ordinary human prompts are
outside these output adapters. Checks rewrite model-visible content; they do
not erase original files or the host's output artifacts.

Both apps inspect after tool completion and at the outgoing context boundary.
The second boundary also covers native error results and synthetic background
notifications. Classification runs in overlapping windows (initially 2000 UTF-16
units, at most 8000, with a 256-unit overlap), without splitting surrogate pairs.
Each app gets its own total wait budget, including HTTP queue time. Defaults are
3000 ms for Redact and 2000 ms for Shield; `"done"` has a 60-second maximum per
app. The model scan stops at 128,000 units. Fixed secret rules still scan the
entire text up to the separate 1,000,000-unit ceiling.

| Condition                                        | Bounded wait                                                                                  | `"done"`                                      |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------- |
| Timeout, offline, unsupported or malformed reply | Keep fixed masking and completed checks; deliver remaining text with an incomplete-check note | Withhold the content                          |
| Model scan reaches 128,000 units                 | Mark partial and retain fixed masking over the full text                                      | Withhold the content                          |
| Text exceeds 1,000,000 units                     | Withhold the content                                                                          | Withhold the content                          |
| Non-text attachments                             | Retain attachments with an unchecked note                                                     | Withhold attachments and text                 |
| Cancellation                                     | Abort local HTTP and withhold pending content                                                 | Abort local HTTP and withhold pending content |

Decision metadata includes masking counts and checked/total UTF-16 coverage of
the classifier input, with explicit partial/withheld outcomes. No raw findings
are included. Already-inspected output is cached in bounded memory (128 entries,
8 MiB text budget); changes to controls select a new cache key. This is transient processing
memory, not a persistent log. Coverage stays incomplete when the host appends an
image-omission explanation or separates attachments from their tool text.

After Redact emits a `[vsr:…]` tag, native `write` and `edit` calls cannot add it
as real data. Recognized `apply_patch`/`patch` additions are checked per file/hunk
as well. Edits may preserve existing tags without adding copies. Full-file
writes containing an active tag are rejected; use focused edits to preserve
masked values. This protection is session-scoped, with active tags inherited by
children. It covers these file-writing tools, not arbitrary writes through shell
programs or custom tools. Disabling Redact disables this write-back protection.

The HTTP client serializes requests and includes queue time in deadlines. A
network failure opens a 60-second breaker; HTTP errors, malformed replies and
caller cancellation are reported separately. Discovery tries
`/v1/systemone/models`, falling back to `/v1/models` only on HTTP 404. The selected
model and its advertised question types are checked before use. Responses over
2 MiB or missing requested answers are rejected. Cancelling a request stops the
local wait; it cannot guarantee a remote model stops computing.
