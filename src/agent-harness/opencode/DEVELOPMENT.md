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
`verify.ts` asks F9's cases through each app's own policy path; its cases and
expected verdicts are the Claude Code plugin's smoke probes. `skills/vsr-setup`
is registered as an OpenCode skill and carries no scripts: its commands mirror
the Claude Code plugin's `local.py` and share its pin, venv and port.
`masking.ts` owns fixed secret patterns, validated Unicode spans and interval
masking. `output-policy.ts` owns Redact/Shield scanning, coverage and attachment
policy. `output-hooks.ts` adapts tool results/outgoing context and prevents
placeholder writes. A context hook added later for Check must run after this
adapter so evidence reflects content actually delivered to the model.

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
with its README and ownership/check registration. Development proceeds in
stages toward one integration Draft PR; intermediate foundations are not a
claim that the six-app product is complete.

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
