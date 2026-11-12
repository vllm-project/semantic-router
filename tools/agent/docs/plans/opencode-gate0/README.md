# OpenCode v2.2.0 capability spike

This is a disposable harness experiment, not the vSR product plugin. It uses
Python's standard library, the installed `opencode` binary, and a loopback fixture
model endpoint. No cloud provider credential is passed to the child process.

Run one case from the worktree root:

```bash
VSR_CASE=redact python3 tools/agent/docs/plans/opencode-gate0/probe.py
```

Cases: `redact`, `error`, `deny`, `explicit`, `resume`, `concurrent`,
`ask-allow`, `ask-reject`, `ask-cancel`, `read`, `web`, `long`, `strict`,
`partial`, `background`, `rpc`, `deny-override-allow`, `deny-override-ask`,
`evidence`, `lifecycle`, `live-routing`. Unknown case names are rejected.

The `ask-*`, `background`, and `rpc` cases launch the real terminal UI inside a pseudoterminal for
about 18 seconds. In the `ask-*` cases, the test plugin waits for a native pending permission request,
checks that the marker file is absent, holds it for 750 ms, then responds through
OpenCode's permission API or interrupts the child session. Automatic replies
exist only in this fixture; they must never be copied into the product guard.
The fixture TUI registers a composer contribution, panel and command, dispatches
the command programmatically and sends Ctrl+G through the pseudoterminal. The
runner checks render callbacks and literal marker text in terminal output. This
is not a visual layout review or a test of a person clicking approval buttons.

`concurrent` starts one password-protected loopback OpenCode server and two
simultaneous root sessions in the same project. One child is routed, the other
explicitly retains `auto`; request headers associate model/effort with each
child session. This tests harness request isolation, not future product state
such as escalation flags, logs, or calibration.

Each run prints a temporary artifact directory and an assertion result. The
runner uses isolated XDG paths and standalone/private processes. The host may
still discover global skills. Raw request traces can contain those instructions;
inspect and sanitize before sharing them. Nothing under the temporary artifact
directory belongs in a commit.

Output cases use native tools and capture the next model request. `read` and
`web` fetch synthetic content; `long` emits 200 KB and exercises the host's
truncation/artifact mechanism. `strict` and `partial` use predetermined results
to prove withholding/unchecked-marker delivery, not real classifier timeouts.

`background` uncovered an additional path: finished shell output is admitted as
a synthetic user message, outside the completed tool result. The fixture's
context hook replaces the whole shell notification before model consumption.
The check inspects all non-assistant messages in outbound requests. It does not
prove stored history or on-disk shell artifacts are sanitized.

`rpc` tests a native RPC snapshot and toggle from TUI to server, server event
updates back to TUI, and visible updated panel text. Handlers are asynchronous
on v2.2.0; a synchronous snapshot handler returned `rpc.internal` during discovery.
The case also checks execution-completion events and context retrieval contain
the final answer and masked tool evidence. The product must track the evidence
actually delivered per turn, rather than treat the entire stored history as
current-turn evidence. RPC subscriptions are live; reconnect needs a snapshot.

Sanitized test outcomes and source hashes are recorded in `results.json`.
Remaining qualification includes real mouse/keyboard approval choices,
finer-grained streaming, additional non-text/error tool paths, queued/steered
turn boundaries, reload/restart, and real decision-model calls.

## Deny precedence and turn evidence

`deny-override-allow` and `deny-override-ask` configure a scoped shell deny
(`printf executed*`) after a general allow. The plugin registers the requested
override, but OpenCode does not invoke permission-evaluation hooks for that deny.
Assertions verify the native shell tool is still available, its attempted call
fails, the override handler is not reached, and the marker file is absent. An
initial broad `shell/* deny` experiment removed the shell tool entirely; scoped
rules avoid that shortcut and exercise actual native permission rejection.

`evidence` launches two sessions on one server and then continues each session.
A1, B1 and B2 read different files with synthetic secrets. A2 reads nothing.
The fixture records admitted prompt IDs, current-turn tool IDs, and matching
results visible in the outgoing context; Map keys deduplicate repeated results.
Completion observations combine those results with the final answer. Assertions
check that only the matching masked result is retained, A2 evidence is empty,
raw secrets and foreign-session evidence are absent, and old A1 evidence still
exists in A2's outgoing history. Thus the test exercises actual exclusion from
retained history, rather than an empty-history shortcut.

This is a candidate evidence-collection strategy, not a shipped Check app. It
qualifies sequential prompts within each session and parallel independent
sessions. Queued/steering prompts, cancellation, rejected prompt admission,
compaction, retries and reconnect races need further qualification; a production
collector must bind to the executing turn rather than assume prompt admission
and execution always coincide. Only fixture text tools and a fixture model are
used, and no hallucination classifier runs.

## Plugin reload and server restart

`lifecycle` runs a root/child tool sequence, changes the test flag through HTTP
RPC, then edits the temporary plugin source to trigger native reload. It checks
cleanup occurred, the stored flag survives, and a new task produces exactly two
decisions. It then terminates and restarts the private server using the same
isolated data directories and repeats the checks through a fresh client.
Across the three tasks each subagent and shell hook runs exactly three times:
no duplicate registrations. Durable flags use `ctx.storage`; in-memory decision
counts intentionally restart at zero. The fixture records observations rather
than claiming every product log/calibration field has a persistence contract.

This validates source reload and fresh-client reconnect after restart. A live
TUI remaining open through server outage, lost RPC-event replay, and stale-control
errors still need explicit tests. RPC event subscriptions are live only: clients
must recover with a fresh snapshot and re-subscribe after connection loss.

## Live decision-model smoke probe

After starting a pinned local runtime, run:

```bash
python3 tools/agent/docs/plans/opencode-gate0/live_probe.py --output /tmp/vsr-live-smoke.json
```

This helper permits localhost only and reuses the Claude plugin's synthetic
smoke inputs, request framing, parsers, and default thresholds. It reads no user
runtime settings or tokens. It requires all six apps to return parseable answers
and records classifier verdict flips, drift, and latency separately. A successful
exit means transport/schema compatibility, not accuracy or interactive deadlines.
It is not yet an OpenCode-to-Vela integration test. Startup instructions and
runtime/model pins are in the parent plan.

`live-routing` additionally requires the local decision runtime on port 8790.
It sends the native child task to a real complexity score question and applies
its decision to the child model; model generation still uses the loopback fixture
provider, so no paid completion API is involved. It checks one confident routing
case, not every routing threshold or classifier outage. The simpler fixture
router rejects an unqualified score instead of implementing product fallback.

On 2026-11-13, the full 30 synthetic Claude plugin probes were also executed
against the pinned vela3-lite runtime: all were `ok`, with no flip, drift beyond
0.15, or parsing/HTTP error. On ARM64 CPU/float32 with four OMP threads, observed
latency ranged from 770 to 1923 ms, median 1281 ms; four exceeded 1500 ms and none
exceeded 3000 ms. These are short serial probes, not a throughput benchmark or
proof that every app stays within its deadline. The six-case smoke also passed.
See `live-results.json` for pins, dependency versions and per-case observations.

A Chinese/emoji-prefix PII example returned valid Python code-point spans whose
slices matched the returned names/email text. TypeScript uses UTF-16 offsets:
convert units before masking after astral characters. This checks one example,
not general PII recall or all normalization/offset cases.
