# Developing vSR for OpenCode

Use Node 22.18 or later and the pinned OpenCode 2.2.0 SDK. From this directory:

```bash
npm ci
npm run check
npm test
npm run test:integration
```

CI runs `make agent-opencode-test` (install, check and unit tests) as part of the
Agent Plugins contract, `make agent-plugins-test`.

`src/index.ts` owns host registration and cleanup. `settings.ts` validates options,
`runtime.ts` owns bounded HTTP calls and capability discovery, `decisions.ts`
retains session-scoped metadata, and `rpc.ts` defines server/UI messages.
`routing-policy.ts` owns tier/effort selection, `routing.ts` owns per-turn
escalation and cancellation, and `routing-hooks.ts` adapts native prompt,
context and tool lifecycle events. Classifier question framing is deliberately
copied into `questions.ts`; sibling plugin imports remain forbidden.
`guard-policy.ts` owns shared-budget command classification and fixed-rule
precedence, `shell.ts` retains the baseline shell scanner and its adversarial
fixtures, and `guard-hooks.ts` tightens native permissions and observes terminal
outcomes. Pending guards reject saturation instead of dropping active checks.
`listener.ts` reads a Router listener's `vsr` profile and rules and asks those
rules, ported from the Claude Code plugin's `lib/profile.ts` and `lib/rules.ts`.
`connection.ts` defines the `/connect` integration and reads its credential;
`index.ts` switches the runtime client to it on load, at each prompt and when a
credential changes, and drops the previous runtime's listener rules.
`verify.ts` asks the runtime check's cases (the panel's runtime line, `/vsr-config` and the setup skill's `vsr_check`) through each app's own policy path; its cases and
expected verdicts are the Claude Code plugin's smoke probes. `skills/vsr-setup`
is registered as an OpenCode skill and carries no scripts: its commands mirror
the Claude Code plugin's `local.py` and share its pin, venv and port.
`log.ts` keeps the learnable decisions in plugin storage, one key per line;
`calibration.ts` is a port of the Claude Code plugin's calibrate scripts
(`fitting.py`, the sampling in `questions.py` with its two per side, implied
labels in `decisions.py`, and the effect and recommendation in `plan.py`), and `test/fixtures/calibration-golden.json` holds fits those
scripts computed, so the two stay equal. `calibrated.ts` applies and keeps the
fitted thresholds; the questions and the suggestion text, in `plain.py`'s
words with the agent and OpenCode's tiers, are in `view.ts`. One difference is
deliberate: the Claude Code plugin logs redact and shield only when they act,
so their lines only rise there, while this log keeps what they let through too,
so their lines can move either way. `probes.ts` is the probe in `runtime.py`
(`run_probes`): the cases are `probes.jsonl` (test/probes.test.ts fails when
they differ), judged at the Claude Code plugin's default thresholds they were
measured at, since some of this plugin's defaults differ.
`masking.ts` owns fixed secret patterns, validated Unicode spans and interval
masking. `output-policy.ts` owns Redact/Shield scanning, coverage and attachment
policy. `output-hooks.ts` adapts tool results/outgoing context and prevents
placeholder writes. Check's context hook must run after this adapter so
evidence reflects content actually delivered to the model.

`test:integration` requires the installed OpenCode 2.2.0 CLI and installs no
models. It packs/extracts the package and starts an isolated private server.
The routing fixtures capture real outgoing model/variant requests, retries,
explicit/resumed/agent-specific choices, parent preservation, observed usage,
pushback, concurrent children and session isolation. They use a synthetic
generation provider and decision runtime, so they qualify host behavior rather
than model accuracy. The CLI tests explicitly pass the parent model/variant;
OpenCode 2.2.0 `run` does not apply the configured default variant consistently.
Guard integration uses the host's session/permission HTTP APIs to leave native
approvals pending, then explicitly approves, rejects or interrupts them. It
checks filesystem markers before/after permission replies, including concurrent
sessions, failed shell execution, offline fixed rules, bounded/strict failures
and interruption of a stalled decision HTTP request. No product code responds
automatically to permission requests.
Output fixtures capture actual generation requests for native read/shell/web,
failed/large outputs, background notifications and images. They assert fixed
secrets do not reach decision requests either, validate actual write/edit
refusals and session isolation, and interrupt a stalled strict check before
resuming the session. Fixture model replies validate host integration; they do
not measure PII/injection model accuracy. Pure tests retain baseline secret/IP
fixtures and add invalid/overlapping/Unicode spans, chunk boundaries, scan
ceilings, partial coverage, attachments and cross-file/hunk patch protection.

Tests use Node's built-in test runner, synthetic inputs and loopback HTTP
servers. Real runtime tests require an explicit separately started model. Do
not download weights, call paid APIs or consume user credentials in unit tests.
The lifecycle contract includes cancellation during response streaming and
queueing; neither should open the offline breaker. Test the installed packed
artifact through the real OpenCode CLI when changing exports or registration.

The package is self-contained and cannot import files beside its installed
root. The CLI resolves its plugin SDK imports in the host. A package changes
with its README and ownership/check registration.

`check.ts` owns human-turn evidence, limits and answer checking; `check-hooks.ts`
registers after output processing and joins final answers to their exact human
boundary. `controls.ts` serializes persistent compare-and-set updates;
`view.ts` owns snapshot recovery and usage summaries independently of the
renderer. `tui.tsx` uses the native Solid owner, keymap, dialogs and panel slots.
Do not create a separate Solid root around setup: the host's keymap provider
belongs to the supplied owner. Panel replacement is active only while our own
panel is selected; dimensions follow the renderer's resize events.

The Check fixture drives real overlapping/queued human turns and captures the
masked grounding sent to its synthetic runtime. Native background-child tests
verify terminal usage and known versus missing prices. TUI tests use Python's
standard-library PTY driver against the installed CLI, send actual function keys
and dialog selections, then assert server state and absence of generation or
classification calls. They need a Unix PTY and `python3`, not an external GUI.
The control fixture restarts the private server and validates persisted settings
and rejection of the previous instance's client revision. Pure controller tests
cover stale events, late snapshots, failed saves and session changes.

OpenTUI/Solid development dependencies provide TSX types; the native host supplies
the shared runtime modules. `seroval` is overridden to 1.6.10 because the Solid
version required by OpenTUI otherwise resolves its older vulnerable minor line.
Do not let an audit fix replace the pinned host SDK or peer versions silently.

## Behavior in detail

The README describes what a person sees. This is the contract each app keeps,
for changing the code and reviewing it.

### Router and Escalate

Router applies to new subagent calls whose agent names no model, `general` and
`explore` alike; an agent with its own model is recorded as `kept`
(`agent-model`). Configure three distinct models
in ascending capability order from one provider, and use that ladder for the
main session's model too. Replace the example IDs with models enabled in your
OpenCode provider configuration. Optional `efforts` lists name supported native
variants in ascending order; omit them to use each model's default. Variants
must appear in OpenCode's model catalog. Explicit child models, resumed children,
agents with their own model and models outside this ladder retain their choices.
The main session is never changed. A child receives its selection at creation;
provider retries retain that selection.

Difficulty and reasoning-step scores use a 0–2 scale. Defaults are the Claude
Code plugin's (its plugin.json): `easyBelow: 0.5`, `hardFrom: 1.5`,
`minConfidence: 0.3`, `downgradeConfidence: 0.6`, and the steps answer's own
lines `stepsLow: 0.5`, `stepsHigh: 1.2`, `stepsMinConfidence: 0.3`, which set the
effort as its pickEffort does. `routing.goal` is `"balanced"`, `"save"` (uses the
ordinary confidence threshold for downsizing) or `"quality"` (no downsizing).
Low confidence, invalid/disabled models or unavailable classification preserve
the host's original choice. `routing.deadlineMs` defaults to 1500 and includes
queue time. Tasks over 32,000 UTF-16 code units skip classification.

Escalate classifies admitted human prompts when their turn actually executes.
Wrong-answer/different-answer probability at least
`routing.feedbackThreshold: 0.8`, or repeat probability at least
`routing.repeatThreshold: 0.75`, adds a re-check reminder.
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

### Guard

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

### Redact and Shield

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

When Redact rewrites an output it tells the model that the tags are vSR's, not
part of the source, and that the values are hidden on purpose and not to be read
another way.

Shield covers text returned by `read` and `webfetch`. It frames classifier input
as file/page contents and uses `shield.injectionThreshold` (default 0.9). Flagged
content is delivered as a quoted JSON string with untrusted-data guidance.
Shell output is not shielded, and web output is not redacted, matching the
baseline tool coverage. Other custom tools and ordinary human prompts are
outside these output adapters. Checks rewrite model-visible content; they do
not erase original files or the host's output artifacts.

Both apps inspect each output once, when its tool completes, as the Claude Code
plugin does; OpenCode keeps the checked text in the session, so later requests
and a restarted server send it as it was checked, and record no new decisions.
The outgoing context boundary inspects only what tool completion never sees:
native error results (named by their call's source) and synthetic background
notifications. It also recovers the session's placeholders from the context, so
writes stay guarded after a restart. Classification runs in overlapping windows (initially 2000 UTF-16
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

### Check

Check is off by default; enable it through the app dialog or `apps.check`.
It checks a root session's final answer against successful native read/shell text
actually delivered during that human turn, after Redact/Shield processing. Queued
prompts, other sessions, earlier turns, failed/withheld results and synthetic
background turns cannot provide evidence. Answers shorter than 40 UTF-16 units
or turns without delivered evidence are not checked. Check does not request
another generation or silently modify the answer.

`check.deadlineMs` defaults to 3000 ms, as the Claude Code plugin's check waits, including queue time (maximum 10000).
`check.unsupportedThreshold` defaults to 0.5. Scores at or above it mark the
answer unverified; lower scores mark it verified against supplied evidence,
not against outside knowledge. Prompt, combined evidence and answer each have
an 8000-unit ceiling; at most 64 read/shell sources per turn are tracked.
Truncated inputs or missing evidence make a low-score result partial, never
fully verified. Offline, timed-out or malformed replies produce skipped checks.
Evidence is transient bounded memory and is absent from RPC/decision records.

### Router listener and runtime client

Pointed at a Router listener that publishes a System One model, vSR reads the
listener's listing (`GET /v1/systemone/models`) at the first prompt, which the
first check waits for, and again at a prompt a minute or more later; loading the
plugin calls no runtime. Its `vsr` profile (`apis.systemone.vsr` in the Router's
config, or Default, On or Off in the Dashboard under Config › Global Config › vSR
agent plugins) turns `router`, `escalate`, `redact`, `shield`, `guard` and
`check` on or off for every plugin pointed at it. An app you name under `apps` in
this plugin's options, or turn on or off in the decision view, stays yours; the listener sets the others.
"follow the Router listener" in `/vsr-config` hands what you changed back to the
listener; the ones in the options stay yours. A plain decision
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

### Decision view and RPC

The native terminal loads the exported `./tui` entry automatically with the
server plugin. Retention defaults to 500 decisions per session and 64 sessions,
with least recently used session eviction.

Use arrows and Enter in native dialogs, Escape to dismiss. Composer counts and
the panel show metadata, reasons, coverage, partial/withheld/unverified states,
planned/observed models and confirmed feedback. While the panel has focus,
`↑`/`↓` pick a decision, `Enter` opens its card (a dialog drawn from the
snapshot, so an answer fills in at once) and `1`-`3` answer it; a mouse press
on an answer answers, and on a decision, in the panel or the sidebar, opens its
card. Answers are offered only for the decisions the snapshot lists in
`answerable` (those in the calibration log). The 2.2.0 SDK has no per-tool
activity annotation slot: the panel associates decisions by tool-call, turn or
message identity instead. Rendering never classifies or generates; only explicit
Diagnose queries capabilities. The panel replaces host fallback content only
while this plugin's own panel is selected, releasing its slot for other panels.

App flags, waits and goal apply to all sessions served by the same plugin
instance/location. Controls persist in native plugin storage before taking
effect and survive reload/restart. Concurrent or obsolete clients must refresh
and retry; a failed save does not optimistically change the display. Changing
configured app/wait/goal defaults supersedes previously saved preferences.
The session view's decisions are bounded memory; feedback on a decision vSR can
learn from is also kept in the calibration log. Snapshots
refresh on events and every three seconds, discard late responses from another
session and reconcile with a new server instance after reconnect.

Child usage appears after actual completion, including background children.
Totals deduplicate child sessions; failed spawns and pending/missing observations
are not zero-cost completed work. Cost figures use the host's estimate only
when the observed model has an explicit finite price table. Unknown prices are
shown separately, including when the host reports zero. Explicit zero prices
are valid. These estimates are not billing receipts or savings measurements.

The server registers RPC `vsr.opencode.v1` with `snapshot`, `diagnose`, `verify`,
`controls`, `feedback` and the `calibration*` methods. `diagnose` queries runtime capabilities and
`verify` asks the runtime check's cases; loading the package alone sends no content to a model. Import the exported `./rpc`
definition to call these through an OpenCode client. A snapshot holds the
requested session's decisions and those of the subagent sessions under it, in
time order, each with the `sessionID` it was made in; feedback is sent with that
session. The server learns a session's parent from OpenCode when it first
records a decision there, keeps it with the session in plugin storage, and
notifies the views of the session and of every session above it. Of a decision's text only its subject is kept: the command,
child task, prompt or output source (file, address or command, never the output
itself), on one line, at most 200 characters, with fixed secret patterns hidden.
Each decision also keeps a copy of its app's thresholds when it was made
(`thresholds`, in the calibration log's order), so a later calibration does not
redraw the bars of decisions already made.
The session view is bounded by retention settings and kept in plugin storage,
one `sessions/<id>` key per session written shortly after it changes, as the
Claude Code plugin keeps its decisions in session state. A restored decision
passes the same checks as a new one, keeps its outcome, observed model, usage
and feedback, and is not logged again; the newest sessions within
`maxSessions` are restored and older keys removed. The calibration log below is
kept in plugin storage too. A new server has a new snapshot instance ID,
allowing clients to discard obsolete revisions.
