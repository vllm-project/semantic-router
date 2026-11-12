# OpenCode vSR plugin: full-feature implementation plan

## Status and intent

- Recorded: 2026-11-12. Planning and public overlap investigation completed;
  live API probes completed on OpenCode v2.2.0. Stage 1 product implementation
  is in progress; remaining edge qualification stays explicit below.
- User requirement: deliver all six Claude Code vSR applications in OpenCode.
  Develop in stages and deliver one integrated Draft PR for community testing.
  An intermediate subset is not completion.
- Proposed approach: a native OpenCode plugin with server and terminal UI parts,
  using the existing decision-runtime HTTP API. This is a reviewable proposal,
  not a claim that every required extension point has been verified.
- Local reference revision: `d7ab7f587b0477f32a184c51a35b79363a7ab82e`.
  Remote main observed during discovery: `a34126c5580842a26401be9fdc6e4a00eb43d790`.
  Recheck main and overlapping work before implementation.

## Approaches and boundaries

1. Native plugin (recommended): integrate at tool, permission, child-session,
   prompt, and completion boundaries. Supports automatic behavior and a native
   decision view, subject to a pinned-version capability experiment.
2. MCP tools and skills: useful for explicitly invoked classification, but do
   not establish interception of built-in tools or automatic child routing.
   They cannot meet full parity alone.
3. Gateway/provider configuration: existing repository work already covers
   model traffic routing. It does not supply local command approval or the
   plugin's tool-result handling and interactive decision view.

The user selected OpenCode and full functionality. Native integration is the
proposed implementation choice. Do not silently replace missing functionality
with a gateway or advisory-only feature and call the migration complete.
If a required API is unavailable, record the blocker and a concrete upstream
extension proposal; revise the implementation design before building around it.

## Existing and potentially overlapping work

Checked public GitHub issues and PRs (including closed/merged items and comments)
for `OpenCode` and `open code`, related harness issues, upstream branch names,
and the current remote `src/agent-harness/` directory on 2026-11-12.

| Work | Observed state and people | Relationship |
| --- | --- | --- |
| [#2424](https://github.com/vllm-project/semantic-router/issues/2424), [PR #2425](https://github.com/vllm-project/semantic-router/pull/2425) | Closed issue; merged PR by AayushSaini101, 2026-07-12 | OpenCode Auto Mode via provider/gateway configuration; blog and assets, not a native six-app plugin. |
| [#6082](https://github.com/vllm-project/semantic-router/issues/6082) | Open, authored by theohsiung; no assignees or comments at inspection | Codex plugin prototypes and eventual implementation. Explicitly excludes OpenCode. Coordinate runtime contracts and parity lessons, without assuming shared implementation ownership. |
| [#6124](https://github.com/vllm-project/semantic-router/issues/6124) | Open, authored by subin9; no assignees or comments at inspection | Per-entrypoint gateway behavior for several harnesses; OpenCode is future scope. Distinct from local plugin hooks. |
| [#5378](https://github.com/vllm-project/semantic-router/issues/5378) | Open epic; API currently lists subin9 as assignee | Coding-agent gateway compatibility, stable model placement, and evaluation. Historical comment says work is distributed across sub-issues; do not interpret the epic assignment as ownership of an OpenCode plugin. |
| [PR #1899](https://github.com/vllm-project/semantic-router/pull/1899) | Merged | Coding-agent manifests/documentation; inspect when choosing configuration locations, not evidence of a native six-app plugin. |

No matching open OpenCode plugin PR was found. Remote `src/agent-harness/`
contained only `AGENTS.md` and `claude-code/`; no upstream branch name identified
an OpenCode plugin. These observations do not exclude private work, unpushed
branches, unnamed forks, or unindexed discussions. No one was contacted and no
GitHub issue, comment, assignment, or PR was created by this task.

Before publishing implementation work, recheck these records and prepare a
focused issue describing native six-app scope and its distinction from #6124.
Publishing or contacting maintainers needs the user's instruction; use local
authenticated `gh` for any subsequently authorized GitHub writes.

## Full acceptance scope

| Capability | Required behavior and evidence |
| --- | --- |
| Router | Classify child task difficulty and steps; select configured model tier and supported effort. Verify actual outbound model/effort, not only a UI label. Preserve explicit choices, resumed children, and main-session model. Keep a selected child model stable across its run and retries. Low confidence or unavailable classification retains the original choice. |
| Escalate | Detect pushback, provide re-check context, and raise the next eligible child according to a defined tier/effort policy. Consume escalation once at the intended boundary; isolate sessions and concurrent child launches. |
| Redact | Mask secrets and PII before the model receives covered read/shell output, including failures and large/chunked output. Preserve correct span offsets and Unicode. Prevent writes that insert masking placeholders as real data. Verify outgoing model content and error paths. |
| Shield | Detect injection in covered file/web content and add appropriate untrusted-content guidance. Preserve source provenance and visible partial/unchecked outcomes. Strict mode withholds content that cannot be checked. |
| Guard | Split shell commands correctly, apply fixed rules and model classification, and obtain native confirmation before executing held commands. Reject without side effects; never relax an existing deny or approval requirement. |
| Check | Compare the final answer with evidence actually delivered to the model that turn, after masking/shielding. Display unsupported findings without recursive continuation loops. Keep off by default, matching the current CPU-oriented baseline; still implement and test it fully. |
| Setup | Configure an existing local/remote runtime and install/start a pinned local runtime where supported; discover capabilities, protect credentials, verify each app, and document restart/uninstall. Follow the existing canonical local-runtime workflow where applicable. |
| Calibration | Label decisions as right/wrong and routed models as appropriate/too weak/too strong; fit bounded threshold updates with reproducible fixtures. Store schema-versioned feedback linked to decisions. |
| UI and observability | Native decision view, app toggles, wait controls, runtime status, per-decision reasons, feedback actions, and aggregate counts. Show decisions beside relevant activity where the verified UI API permits; document any placement difference. Include usage-based routing cost estimates only for known prices and observed usage. |

Full parity means functional behavior, not identical pixels. Native TUI controls
and decision visibility are required; desktop GUI is outside the initial parity
target. A missing TUI API is an explicit design blocker, not permission to omit UI.

## Architecture and durable constraints

- Add a self-contained `src/agent-harness/opencode/` package with its own `OWNER`,
  `AGENTS.md`, `DEVELOPMENT.md`, README, package manifest, and tests.
- Separate cohesive policy/runtime-client logic from OpenCode adapters and TUI.
  Use a typed server-to-UI contract for decisions, state, controls, and feedback.
- Consume `/v1/decisions` and capability discovery over HTTP. Do not import
  router/runtime code or sibling `claude-code/` modules at runtime.
- Reuse classifier framing, pure algorithms, and test cases deliberately, with
  their source revision recorded. Start self-contained; shared packaging is a
  separate architectural change, not an incidental dependency on `../`.
- Make model tiers configurable. Start qualification with one provider/model
  family; document the tested matrix. Cross-provider routing is not assumed.
- Scope all mutable state to the relevant session, turn, child and tool call.
  Define persistence, retention, concurrent writes, reload and restart behavior.
- Define one explicit failure table for timeout, offline, cancellation, malformed
  reply, unsupported question type, and exhausted coverage. Fixed guard rules
  remain active offline. Strict waits hold/withhold unchecked actions/content.
  Bounded waits visibly report partial/skipped checks. Reconcile README/code
  disagreements in the Claude baseline before copying any timeout semantics.
- Default logs contain decision metadata, not credentials or raw sensitive tool
  output. Remote-runtime setup explains what content leaves the machine.
- No new Router configuration contract is required by the proposal. If that
  changes, apply `tools/agent/docs/change-surfaces.md` before implementation.

## Delivery sequence

All steps below are pending. Each implementation stage includes its own tests and
documentation; later stages do not defer basic correctness of earlier stages.

### Gate 0: pinned-version feasibility and design freeze

- [ ] Select and record an installable OpenCode release and SDK version; resolve
  V1/V2 differences. Do not use floating `latest` for qualification.
- [ ] Prove access to child task text and a reliable child-start boundary; show
  effective model/effort, explicit/resume exceptions and concurrent isolation.
- [ ] Prove permission escalation before side effects, including cancellation.
- [ ] Prove result replacement on successful, failed, streamed/large tool outputs
  before model consumption; list all tool paths and coverage exclusions.
- [ ] Prove turn evidence collection, prompt escalation, completion hooks,
  native TUI controls, activity association, and server/UI state transport.
- [ ] Record commands, sanitized observations, versions, and gaps. Freeze the
  adapter API, failure table, UI placement and supported installation matrix.

The model-resolution hook documented today is experimental and same-provider.
Its per-request behavior alone does not establish child-run-stable routing.
If Gate 0 fails, keep the full acceptance scope pending and record dependencies.

### Stage 1: package, runtime client, settings and decision contracts

- [x] Implement capability discovery, authenticated requests, parsing, deadlines,
  cancellation, offline circuit breaker and bounded recovery.
- [x] Implement validated settings, session-scoped metadata records, event/feedback
  schemas, credential handling and runtime diagnostics. Durable logs and feedback
  remain in Stage 6.
- [x] Register ownership/minimum checks in `tools/agent/domains.yaml`; provide
  package installation smoke coverage and basic developer documentation.

### Stage 2: router and escalate

- [ ] Implement task scoring, tier/effort mapping, confidence policy, stable child
  selection, explicit/resume exclusions, and atomic escalation consumption.
- [ ] Capture actual model/usage facts for later cost reporting.
- [ ] Test request-level observations, concurrent children, retries, cancellation,
  unsupported models/question types, offline fallback, and prompt pushback.

### Stage 3: guard

- [ ] Port shell decomposition and fixed rules with adversarial fixtures for
  quoting, substitutions, heredocs, pipelines, wrappers and nested shells.
- [ ] Integrate native approval; track held/approved/declined outcomes.
- [ ] Verify no side effects before approval, rejection/cancellation behavior,
  existing permissions, time budgets, offline rules and strict wait policy.

### Stage 4: redact and shield

- [ ] Implement secret rules, span masking, source framing, bounded scanning,
  injection guidance and withholding, plus placeholder write-back prevention.
- [ ] Cover read/shell/web adapters and error/large-output paths explicitly.
- [ ] Assert synthetic secrets never reach captured model requests on checked
  paths. Test partial/unchecked visibility, Unicode offsets and benign versions.

### Stage 5: check and complete native decision UI

- [ ] Implement turn-scoped evidence and answer checking with bounded latency.
- [ ] Complete native panel, activity annotations where supported, status/counts,
  toggles, waits, feedback, offline/partial/withheld states and keyboard operation.
- [ ] Derive displays from server decisions, never run classification in render.
- [ ] Test multi-session selection, reload/reconnect, failed control changes,
  stale events and usage/cost accounting. Label price-based figures estimates.

### Stage 6: setup, calibration and complete user lifecycle

- [ ] Add setup and calibration workflows using the selected OpenCode mechanisms.
- [ ] Qualify local CPU, optional GPU and remote runtime paths with explicit
  prerequisites; reuse supported runtime installation conventions and pins.
- [ ] Implement feedback persistence, bounded threshold fitting and reminders.
- [ ] Document install/update/trust where applicable, credentials, uninstall,
  limits, tool coverage, failure policy and differences from Claude Code.

### Stage 7: full parity qualification and reproducible measurements

- [ ] Install a packed artifact in a clean environment; run all six apps together.
- [ ] Exercise offline/recovery, timeout, strict mode, cancellation, restart,
  concurrent sessions/children, malformed replies and unsupported capabilities.
- [ ] Record decision p50/p95 latency, timeout/partial rates, false holds and
  missed fixtures, routing choices, task success, observed tokens and costs.
- [ ] Compare plugin disabled/enabled on the same pinned task/model set. Report
  failures and classifier overhead; do not infer savings from model labels alone.
- [ ] Publish only sanitized reproducibility artifacts when authorized; complete
  parity/coverage matrix and minimum-version documentation.

Dependencies: Gate 0 -> PR 1 -> PRs 2/3/4; PR 5 integrates those applications;
PR 6 uses their settings/events/feedback; PR 7 qualifies the combined result.
Branches may be developed independently after contracts stabilize, but merge
only self-contained, tested steps. Do not push to merge-queued branches.

## Validation and completion

Unit/contract tests use small fixtures, temporary directories and a fake decision
runtime. Native harness integration tests verify actual side effects, outgoing
model requests and UI behavior. Model-backed tests take explicit runtime/model
inputs and record pins; downloaded models/caches stay ignored.

For each PR run `make impact ENV=cpu CHANGED_FILES="..."`, then
`make check CHANGED_FILES="..."`, plus the registered package tests and appropriate
integration assertions. Run `make harness-check` for registry, harness code,
workflow or instruction changes. Verify the installed artifact as well as source.

Completion requires every full-scope row above demonstrated on the documented
supported release, all seven deliveries reconciled, and no unresolved blocker
disguised as a successful skip. The exact PR count may change while keeping scope.

## Resume here

Planning validation on 2026-11-12: independent builder review found no material
unanswered planning questions. Markdown formatting, spelling, structure and
dependency checks passed. `make impact` selects the harness domain. The harness
suite ran 796 tests with five failures in Kubernetes generated-contract tests
because `go` and `controller-gen` are unavailable in the current environment;
the full repository check is therefore not green. No application code changed.

1. Recheck overlapping GitHub work and current main.
2. Review this proposed design and resolve Gate 0 against an actual OpenCode build.
3. Record a capability matrix and final technical choices before product code.
4. Update checkboxes with links and evidence as work lands; leave untested items
   pending. No implementation is complete at the time this document was written.

## References

- [OpenCode plugin packaging](https://opencode.ai/docs/build/plugins)
- [OpenCode server hooks and API](https://opencode.ai/docs/build/plugins/server)
- [Claude Code reference](../../../../src/agent-harness/claude-code/README.md)
- [Plugin isolation rules](../../../../src/agent-harness/AGENTS.md)
- [Existing OpenCode gateway guide](../../../../website/blog/2026-07-09-opencode-auto-mode.md)

Official API documentation was inspected during the preceding planning discussion;
its capabilities remain documentary evidence until Gate 0 measures them.

## Gate 0 observations: local OpenCode v2.2.0 (2026-11-12)

Executed the installed CLI against a loopback OpenAI-compatible fixture endpoint
using `run --standalone`, temporary project/config/data/cache/state directories,
and an environment allowlist without provider credentials. No paid model or
Vela runtime was used. Existing global skills may still be discovered by the
host; user configuration and credentials were not modified. Raw traces remain
in temporary directories and must not be published without inspection.

Reproduction spike: [probe.py](opencode-gate0/probe.py), Python standard library
only. Run from this worktree with the installed `opencode` on PATH:

```bash
VSR_CASE=redact python3 tools/agent/docs/plans/opencode-gate0/probe.py
VSR_CASE=error python3 tools/agent/docs/plans/opencode-gate0/probe.py
VSR_CASE=deny python3 tools/agent/docs/plans/opencode-gate0/probe.py
VSR_CASE=explicit python3 tools/agent/docs/plans/opencode-gate0/probe.py
VSR_CASE=resume python3 tools/agent/docs/plans/opencode-gate0/probe.py
```

| Probe | Observed result |
| --- | --- |
| Plugin loading and prompt admission | Native object-form server plugin loads; prompt hook receives text. |
| Child task access and routing | `tool.hook("execute.before")` sees native `subagent` input including plaintext prompt; setting `input.model="probe/routed"` changes actual outbound model. |
| Stable child execution | Main requests stay on `auto`; two child requests separated by a real shell call stay on `routed`. |
| Effort | Child-scoped context hook sets `options.reasoningEffort="high"`; endpoint receives `reasoning_effort: high`. Main requests have no effort override. |
| Explicit model | Input naming `probe/auto` is preserved by the spike's exclusion rule; all requests remain `auto`. |
| Resume | Continuing the child with its returned session ID, without another model override, sends the subsequent request to `routed`. |
| Successful output replacement | A shell prints a synthetic secret. Replacing both result content and structured output causes the next request to contain `MASKED`, without the original value in tool results. |
| Nonzero command output | Exit code 7 still reaches the completed-result hook; the same output replacement works. Tool infrastructure errors are a separate path. |
| Permission denial | Permission evaluation changed shell effect to deny. The requested marker file is absent, and the next model request receives a permission rejection. |
| Experimental model resolution | Registration returned without error, but the callback did not fire in these runs. Do not depend on it for v2.2.0; cause and wider coverage are unqualified. |

These are deterministic harness-boundary tests, not classifier quality, real
backend support, savings, or end-to-end product parity. Remaining Gate 0 work:
native ask/approve/reject/cancel dialogs, explicit deny precedence, concurrent
sessions/children, background and large/chunked output, read/web paths, completion
and evidence hooks, native TUI controls and state transport, and restart behavior.
Only the fixture provider was exercised; cross-provider support is not established.

### Follow-up: native permission lifecycle, TUI and concurrency

The fixture now includes readable server/TUI modules and a
[reproduction guide](opencode-gate0/README.md).

- Non-interactive `opencode run` automatically rejects an `ask` request with
  feedback explaining that it cannot ask the user. Do not use that mode to
  qualify the interactive guard workflow.
- Real TUI execution in a pseudoterminal produces a pending native permission
  request. The marker file is absent both on admission and after a 750 ms hold.
  A test-only API reply of `once` permits the command and creates the marker;
  `reject` leaves it absent. Interrupting the waiting child also leaves it absent.
  A rejection stops the child's current step rather than producing another
  child model request, which differs from a corrected-error continuation.
- The TUI module loads on v2.2.0. The composer contribution and session panel
  both render their marker text in captured terminal output. A registered
  command opens the panel; the harness dispatches it and sends Ctrl+G. This
  establishes native UI extension feasibility, not final layout/interaction
  quality or a human clicking the permission controls.
- Two simultaneous root sessions on one private server retain different child
  selections (`routed/high` versus explicit `auto/default`) across two requests
  each, with session IDs captured at the outgoing HTTP boundary. Product state
  isolation remains to be tested once such state exists.

No Vela endpoint has been called. Mock decisions are sufficient for these
control-boundary probes. Before release, connect the pinned `vela3-lite` runtime
(or a verified compatible runtime advertising the required question types) and
validate framing/parsing for all six apps, score/span semantics, latency,
timeouts, offline recovery, and representative positive/negative cases. Real
runtime acceptance is required; a live runtime is not needed to finish the
remaining host API probes. Gate 0 remains partially complete.

### Follow-up: output paths, completion evidence and native RPC

The fixture suite now covers 16 named cases; see
[sanitized outcomes and source hashes](opencode-gate0/results.json). The additions
exercise the installed v2.2.0 binary rather than simulated hook callbacks.

| Boundary | Measurement and design consequence |
| --- | --- |
| Native read and webfetch | The after-tool hook can replace synthetic file/page content before the next outbound request. A loopback HTTP server supplies the page. This establishes text-result replacement, not production injection detection. |
| Long foreground shell output | A 200 KB result enters the after-tool hook with `truncated: true` and an on-disk output artifact reference. Replacement prevents original tool text reaching the next model request. Raw artifacts remain on disk; future reads and other access paths still require protection. |
| Strict and bounded policy presentation | Fixed fixture decisions demonstrate `WITHHELD` without original content and `UNCHECKED` alongside deliberately unfiltered content. Actual classifier deadlines, chunk accounting and cancellation remain runtime-client tests. |
| Background shell completion | The original tool-result-only approach leaked synthetic output through a later `<shell ...>` user message. A context hook replacing this message before transmission prevents the leak in the qualified fixture. Both result and outgoing-context interception are required. Stored history and local artifacts are not thereby sanitized. |
| Completion evidence | `session.execution.succeeded` identifies root and child completion. `ctx.session.context` includes final text and masked tool content; context hooks also expose the outgoing messages. The product must retain per-turn delivered evidence, exclude raw/unread content, handle deduplication and isolate sessions. |
| Native RPC and TUI | TUI reads server decision counts, changes a test setting, receives the server update event and renders the changed state. Async RPC handlers work; a synchronous snapshot handler failed on this release. Use async handlers and snapshot recovery after missed events. |

The initially observed background leak is reproduced and addressed by the spike;
no production guardrail has been implemented. The fixture masks whole output or
notifications, not semantic PII spans. All fixture state and automatic permission
replies remain disposable and must not be copied unchanged into product code.

Core extension feasibility now has evidence for routing, permissions, text
output replacement, completion and UI transport. Do not mark full Gate 0 or
six-app parity complete: remaining work includes deny precedence, wider stream
and tool-error coverage, per-turn evidence boundaries, reload/restart behavior,
and interactive UX acceptance. Then freeze the supported API matrix and validate
real vela3-lite framing, spans and failure behavior before shipping product PRs.

### Follow-up: explicit deny precedence and per-turn evidence

Added `deny-override-allow`, `deny-override-ask`, and `evidence` to the
[reproduction suite](opencode-gate0/README.md).

- With a scoped configured deny for the tested shell command, the shell tool
  remains available and receives a real attempted call. Native permission
  rejection happens without invoking the plugin's permission evaluator; neither
  an allow nor an ask override can run, and no marker file is created.
- A broad deny of every shell command instead removes the shell tool from the
  available tool list. This is a separate behavior, not used as the final proof
  of permission-evaluation precedence.
- Two sessions execute four prompts on one server. A1 and B1 run concurrently;
  A2 continues A without reading; B2 continues B and reads a different file.
  The candidate collector keys records by session and admitted prompt ID,
  matches current tool IDs against model-visible tool results, and deduplicates
  by tool ID. Completion records contain the correct final answer and only the
  matching masked evidence. A2 has no evidence despite A1 remaining in its model
  history. Outbound traces contain no raw file secrets or foreign-session
  evidence. This establishes a feasible collection path for the measured cases.

Production Check is still pending. Prompt admission is not an executing-turn
boundary when queuing or steering is involved; qualify those cases, rejection,
interruptions and compaction before adopting this collector. Snapshotting full
history at completion alone is insufficient. These tests do not evaluate whether
an answer is grounded; they test which evidence would be supplied to that check.

Validation for this follow-up: all 19 fixture cases passed on OpenCode v2.2.0,
including the previous 16 cases. Source hashes match the recorded results.
Python/TypeScript formatting, spelling and whitespace checks passed. The earlier
repository-wide harness prerequisite limitation (missing Go/controller-gen)
remains; this result does not claim the full repository check is green.

### 2026-11-13: reload/restart and live-runtime preparation

`lifecycle` verifies a private OpenCode server with native plugin source reload
and a complete server restart. The candidate flag is written through plugin
storage and remains false after both operations. Cleanup runs on reload; each
fresh task records exactly two decisions. Across three tasks, each native
subagent/shell hook runs three times, demonstrating no duplicate registration.
Reconnection is through fresh clients; this does not qualify a still-open TUI
client's reconnection or event-loss handling. Session decision counters remain
volatile by design in this fixture.

Local environment inspection found no running decision service on loopback port
8790 and no existing vela3-lite snapshot. The machine is ARM64. A temporary,
non-service runtime installation uses the Claude plugin's exact runtime commit
`579e0ca2a0b1b03580eada4213da84fe68db57c2`. The proposed live model revision is
`vllm-sr/vela3-lite@c789afd6fbeaceb3c734b95b9ca3ad9772a7f434`.
Actual runtime installation/startup and all-six-app smoke outcomes are pending
until recorded below. No project content is used as model input.

The fixture regression suite passed 20/20 cases after the lifecycle addition.
For the ARM64 live experiment, use official CPU PyTorch `2.9.1+cpu` in the
temporary venv; the default PyPI CUDA wheel is unnecessary for this API test.
Model weights and the pinned base snapshot were downloaded successfully.

### Live results: 2026-11-13

The pinned runtime started successfully on localhost using ARM64 CPU, float32,
four OMP threads and PyTorch 2.9.1+cpu. All six smoke cases parsed and matched
their reference verdicts. All 30 existing synthetic probes returned `ok`:
zero protocol errors, zero verdict flips, and zero score drift beyond 0.15.
Observed latency: median 1281 ms, range 770–1923 ms; four calls exceeded 1500 ms.
Do not equate these short-case timings with production hook deadline coverage.

`live-routing` proves an OpenCode child-task hook can call the actual Vela
complexity classifier, apply the confident decision, and produce an observed
`auto -> routed -> routed -> auto` generation-request sequence. Generation
uses the local fixture provider; no paid model or private project input is used.
Other five apps have live runtime contract probes, not yet integrated product
implementations in OpenCode.

A synthetic Chinese/emoji-prefix PII request produced valid code-point spans:
Python slices match each returned span text. The future TypeScript masker must
convert code points to UTF-16 units for astral characters. Accuracy, recall,
normalization variants and strict runtime-failure behavior remain unqualified.

Pins, dependency versions, individual results and Unicode evidence are in
[live-results.json](opencode-gate0/live-results.json). Model snapshots remain
in the normal ignored Hugging Face cache; the venv is temporary and no systemd
or login service was created. The temporary runtime is stopped after verification.

This advances Gate 0; it does not complete six-app product parity. Remaining
qualification includes queued/steering turn boundaries, finer stream/tool-error
paths, still-open TUI outage recovery and manual interaction acceptance. The
full repo harness prerequisite limitation remains recorded above.

Final follow-up verification: 21/21 OpenCode cases passed, including live routing.
The recorded source hashes match the current probes. Formatting, spelling and
whitespace checks passed; no GitHub write or product implementation was made.

### Accepted delivery update

The user approved staged implementation followed by one integration Draft PR
for community testing. The seven headings above now denote implementation
stages, not seven publishing events. Keep cohesive commits and per-stage checks;
prepare one complete feature matrix and install/test guide before publication.
No GitHub publication is performed by this implementation step.

Stage 1 now adds the self-contained `src/agent-harness/opencode` package: strict
settings, bounded/cancellable HTTP discovery and decisions, credential indirection,
session metadata retention, a versioned RPC definition and native package tests.
Stage-specific behavior and validation are recorded when checks finish. The
remaining queue/steering and recovery cases are acceptance requirements of their
owning implementation stages; no unsupported behavior is declared complete.

### Stage 1 implementation checkpoint

The user-approved native package foundation is implemented. The public exports
are the OpenCode server module and versioned RPC definition. There is no app/UI
behavior claim at this checkpoint: routing/escalation, guards, output handling,
Check and the terminal UI remain assigned to later stages.

Validation: 12 fixture-based Node tests passed, strict TypeScript and formatting
passed, and a packed artifact loaded in the installed OpenCode 2.2.0. The native
installation test verifies model-list discovery, token delivery without token
exposure, rejected malformed RPC inputs, and no runtime call on plugin load.
The CI contract runs the same Node framework with native JUnit evidence. CI
selection/batching tests passed; full repository results are recorded next.

Implementation choices: environment-variable credential indirection; rejected
redirects; 2 MiB response bound; queue time included in request deadlines;
separate cancellation/timeout/network failures; 60-second network breaker;
per-session bounded metadata and feedback with snapshot instance/revision IDs.
State is volatile until the durable calibration/logging stage. No raw text is a
field in decision records. Remote computation may outlive HTTP cancellation.

The spike's dynamic base64 import was replaced by its ordinary static import to
satisfy the AST security gate. Historical probe hashes in results.json refer to
the original tested snapshot; this syntax-only cleanup is not a new live-model
measurement and does not rewrite historical outcomes.

Stage 1 final validation: `make check` passed for every changed path. This
included `make harness-check` (797 CI tests, 74 agent-script tests and tiny-model
checks), all source/security/structure checks, TypeScript/format checks and 12
Node unit tests. The packed-artifact native OpenCode integration test passed
separately. A SHA-256-verified temporary Go 1.25.0 toolchain resolved the earlier
host prerequisite failures; no global Go installation was changed. The new CI
contract's Node/JUnit commands also ran successfully on the local ARM64 host;
that local result is not represented as a hosted AMD64 CI receipt.

The next implementation stage is router/escalate with configurable model tiers,
confidence/fallback rules and actual child-model/effort assertions. The integration
Draft PR remains pending until the complete six-app scope is reviewable.
