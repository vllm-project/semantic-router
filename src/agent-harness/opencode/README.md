# vSR for OpenCode

This package is under development for OpenCode 2.2.0. Router, Escalate, Guard,
Redact, Shield and Check are implemented, with a native terminal decision panel
and durable app/wait controls, and set up through OpenCode's `/connect` and a
`vsr-setup` skill, and calibrated from your own answers with `/vsr-calibrate`.
Full combined qualification remains a later implementation stage.

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

## Set up the decision runtime

vSR needs a decision runtime: `vllm-srun serve` on this machine, or a Router
listener that publishes a System One model.

1. **No runtime yet?** Ask the agent to use the `vsr-setup` skill. It looks at
   the machine, installs `vllm-srun` pinned to the commit vSR was tested against
   in `~/.local/share/vsr/venv`, and starts vela3-lite on port 8790 on the GPU
   or CPU, for this login or as a user service. OpenCode asks before each
   command it runs. It is the same runtime the Claude Code plugin installs, so
   one runtime serves every agent on the machine.
2. **Connect it.** Run `/connect` and pick **vSR decision runtime**: **Runtime
   without a token** for `http://127.0.0.1:8790/v1/decisions`, or **Runtime with
   a token** for one that needs a key. The form asks the `/v1/decisions` URL and,
   when the runtime serves several, the model. OpenCode keeps the token in its
   own credential store, and vSR switches to the connected runtime at once,
   without a restart. A connected runtime takes the place of `runtimeUrl`,
   `runtimeModel` and the token variable below; disconnect it to return to them.
3. **Check it.** Press **F9** in a session. vSR lists the runtime's models and
   asks one case per enabled app through the path that app's hook takes (a
   force push for guard, a support ticket for redact, a page with hidden
   instructions for shield, an easy task for router, pushback for escalate, an
   invented summary for check). Each line says whether the app decides it as it
   does on vela3-lite and how long it took, with the median. An app with no
   answer means the runtime is offline, refuses the token, or is slower than the
   app's wait (F7).

A remote runtime receives everything the enabled apps check: each read and shell
output, fetched page, prompt, child task, shell command and answer.

To remove vSR: take the plugin out of `opencode.json`, remove the credential in
`/connect`, and, if `vsr-setup` installed a runtime, stop it
(`systemctl --user disable --now vsr-runtime` for the user service) and delete
`~/.local/share/vsr`, unless the Claude Code plugin still uses it.

Without `/connect`, the options name the runtime. `runtimeModel` selects one model
when an endpoint serves several. Remote URLs use the same `/v1/decisions` API.
Set the named environment variable in the OpenCode server's environment if the
endpoint needs a token. Credentials, URL query tokens and embedded URL passwords
are not accepted as options. No token is returned in diagnostics or decision
snapshots. Redirects are refused so authorization cannot move to another host.
A remote decision runtime receives the text supplied for classification.

The server registers RPC `vsr.opencode.v1` with `snapshot`, `diagnose`, `verify`,
`controls`, `feedback` and the `calibration*` methods. `diagnose` queries runtime capabilities and
`verify` asks F9's cases; loading the package alone sends no content to a model. Import the exported `./rpc`
definition to call these through an OpenCode client. Snapshots are scoped to the
requested session. Of a decision's text only its subject is kept: the command,
child task, prompt or output source (file, address or command, never the output
itself), on one line, at most 200 characters, with fixed secret patterns hidden.
The session view is memory-only and bounded by retention settings; the
calibration log below is kept in plugin storage. A new server has a
new snapshot instance ID, allowing clients to discard obsolete revisions.

Options also include `apps`, `waits`, `maxDecisions` and `maxSessions`. Apps default
to enabled except Check; waits are milliseconds (`1000`, `2000`, `3000`) or
`"done"`. Unknown options are rejected rather than silently ignored. Retention defaults to 500 decisions per
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
observed token usage and host cost estimates read after completion. Missing observations remain
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

## A Router listener's switches and rules

Pointed at a Router listener that publishes a System One model, vSR reads the
listener's listing (`GET /v1/systemone/models`) at the first prompt, which the
first check waits for, and again at a prompt a minute or more later; loading the
plugin calls no runtime. Its `vsr` profile (`apis.systemone.vsr` in the Router's
config, or Default, On or Off in the Dashboard under Config › Global Config › vSR
agent plugins) turns `router`, `escalate`, `redact`, `shield`, `guard` and
`check` on or off for every plugin pointed at it. An app you name under `apps` in
this plugin's options, or turn on or off in the decision view, stays yours; the listener sets the others.
"Follow the Router listener" in the app dialog (F6) hands the apps you changed
there back to the listener; the ones in the options stay yours. A plain decision
runtime, whose listing answers HTTP 404, sets nothing.

Rules the listener binds to guard and redact (`vsr.rules`) come with the same
listing, by name and description only, and the listener evaluates them at
`POST /v1/systemone/rules`, beside the decision model's questions and within the
same wait. Guard sends the whole command: a rule that matches holds it for
native approval with the rule's description. Redact sends the output with its
secrets hidden, the same text the PII question reads, and hides what either
finds at once, a rule's find as `[vsr:<rule name>]`. Where a rule and a PII span
cover the same text, the rule's name is the tag. A `required` rule the listener
could not check holds the command or withholds the output, whatever the wait; an
advisory one is recorded as skipped. An output over 256 KiB of UTF-8 is past what
one rules call reads, so its rules count as not checked. An answer for another
revision of the rules reads the listing again and asks once more. An outage keeps
the rules read before, so it never drops a required rule; they are kept in
OpenCode's plugin storage, so a restart during an outage keeps them too. Rule-name tags get the same write-back
protection as vSR's own.

When Redact rewrites an output it tells the model that the tags are vSR's, not
part of the source, and that the values are hidden on purpose and not to be read
another way.

The listing may use the `vllm-sr/4.0` form (`data`) or the `vllm-sr/5.0` form
(`models`, each with its `name`, `features` and `card`).

The HTTP client serializes decision requests and includes queue time in
deadlines; the listing and rules calls do not wait in that queue. A
network failure opens a 60-second breaker; HTTP errors, malformed replies and
caller cancellation are reported separately. Discovery tries
`/v1/systemone/models`, falling back to `/v1/models` only on HTTP 404. The selected
model and its advertised question types are checked before use. Responses over
2 MiB or missing requested answers are rejected. Cancelling a request stops the
local wait; it cannot guarantee a remote model stops computing.

Check is off by default; enable it through the app dialog or `apps.check`.
It checks a root session's final answer against successful native read/shell text
actually delivered during that human turn, after Redact/Shield processing. Queued
prompts, other sessions, earlier turns, failed/withheld results and synthetic
background turns cannot provide evidence. Answers shorter than 40 UTF-16 units
or turns without delivered evidence are not checked. Check does not request
another generation or silently modify the answer.

`check.deadlineMs` defaults to 1500 ms, including queue time (maximum 10000).
`check.unsupportedThreshold` defaults to 0.5. Scores at or above it mark the
answer unverified; lower scores mark it verified against supplied evidence,
not against outside knowledge. Prompt, combined evidence and answer each have
an 8000-unit ceiling; at most 64 read/shell sources per turn are tracked.
Truncated inputs or missing evidence make a low-score result partial, never
fully verified. Offline, timed-out or malformed replies produce skipped checks.
Evidence is transient bounded memory and is absent from RPC/decision records.

The native terminal loads the exported `./tui` entry automatically with the
server plugin. Open a session, then use these keys or slash/palette commands:

| Key     | Command          | Action                                                     |
| ------- | ---------------- | ---------------------------------------------------------- |
| F5      | `/vsr`           | Open decisions for the selected session                    |
| F6      | `/vsr-apps`      | Enable or disable an app, or follow the Router listener    |
| F7      | `/vsr-waits`     | Choose Redact, Shield or Guard bounded/strict waits        |
| F8      | `/vsr-feedback`  | Label a retained decision; Router also offers model labels |
| F9      | `/vsr-check`     | Check the decision runtime: one case per enabled app       |
| F10     | `/vsr-sessions`  | Select from the latest 100 accessible sessions             |
| Palette | `/vsr-goal`      | Choose balanced, quality or save                           |
| Palette | `/vsr-refresh`   | Refresh the authoritative snapshot                         |
| Palette | `/vsr-calibrate` | Calibrate an app's thresholds from your answers            |

## Calibrate from your own answers

vSR keeps the newest 5,000 decisions it can learn from, across sessions and
restarts, in OpenCode's plugin storage: each with its subject, the scores it held
against its thresholds and the thresholds in use then. A rule's decision, a
skipped or withheld one, and a route outside the configured tiers are not kept.
Run `/vsr-calibrate` and pick an app:

1. It asks about the unanswered decisions nearest the app's line, on both sides:
   "Should vSR have asked you before this command ran?" with what vSR did and
   how near the line it was; for router, whether the routed model was too weak,
   right or stronger than needed. Skip any you are unsure of, or stop.
2. For apps other than router, pick what matters more: balanced, fewer false
   alarms or catch more.
3. It shows the new lines, and how many of your answers the current and the new
   ones agree with. Apply, or keep the current ones.

Some answers come without asking: F8 feedback on a decision, a held command you
declined (it should have asked), and pushback within 15 minutes of a route to
the lighter two models (it needed a stronger one). An app needs 8 answers before
its lines move, and each line moves at most 0.15 per calibration, the same fit
as the Claude Code plugin's `/vsr:calibrate`. Escalate only logs when it
escalates, so its lines only rise.

Calibrated thresholds are kept in plugin storage and apply over the plugin
options at once and after a restart. Changing a threshold in `opencode.json`
drops the calibration in favor of what you set. The panel's Calibration line
says when you last calibrated and how many decisions came since, and suggests
`/vsr-calibrate` after 300.

Use arrows and Enter in native dialogs, Escape to dismiss. Composer counts and
the panel show metadata, reasons, coverage, partial/withheld/unverified states,
planned/observed models and confirmed feedback. The 2.2.0 SDK has no per-tool
activity annotation slot: the panel associates decisions by tool-call, turn or
message identity instead. Rendering never classifies or generates; only explicit
Diagnose queries capabilities. The panel replaces host fallback content only
while this plugin's own panel is selected, releasing its slot for other panels.

App flags, waits and goal apply to all sessions served by the same plugin
instance/location. Controls persist in native plugin storage before taking
effect and survive reload/restart. Concurrent or obsolete clients must refresh
and retry; a failed save does not optimistically change the display. Changing
configured app/wait/goal defaults supersedes previously saved preferences.
Decision records and feedback remain bounded memory until Stage 6. Snapshots
refresh on events and every three seconds, discard late responses from another
session and reconcile with a new server instance after reconnect.

Child usage appears after actual completion, including background children.
Totals deduplicate child sessions; failed spawns and pending/missing observations
are not zero-cost completed work. Cost figures use the host's estimate only
when the observed model has an explicit finite price table. Unknown prices are
shown separately, including when the host reports zero. Explicit zero prices
are valid. These estimates are not billing receipts or savings measurements.
