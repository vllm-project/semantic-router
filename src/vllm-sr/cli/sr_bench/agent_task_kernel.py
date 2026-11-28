from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

SYSTEM_INSTRUCTIONS: str = (
    "You are a concise coding-agent benchmark assistant. Follow exact "
    "answer-token instructions when they are present."
)


@dataclass(frozen=True)
class TaskTurn:
    phase: str
    prompt: str
    tool_name: str = ""
    tool_result: str = ""
    expected_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    name: str
    turns: tuple[TaskTurn, ...]
    suite: str = "smoke"


def task_specs(suite: str = "smoke") -> tuple[TaskSpec, ...]:
    specs = smoke_task_specs() + long_horizon_task_specs()
    if suite == "all":
        return specs
    return tuple(task for task in specs if task.suite == suite)


def smoke_task_specs() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(
            name="python-debug",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We are debugging a Python dataclass test failure. "
                        "First say what evidence you need."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt=("Use the provided tool result to identify the root cause."),
                    tool_name="read_test_log",
                    tool_result=(
                        "pytest: ValueError: mutable default <class 'list'> for "
                        "field tags is not allowed; use default_factory"
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the final fix. Include exact tokens "
                        "ROOT_CAUSE=mutable-default and FIX=default-factory."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=mutable-default",
                        "FIX=default-factory",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="ops-triage",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A router failure appeared after inserting a fault proxy. "
                        "Plan the triage briefly."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the network inspection result to decide the fix.",
                    tool_name="inspect_network",
                    tool_result=(
                        "Envoy and vLLM are attached to pr1989-ga-vllm-sr-network; "
                        "the proxy was started on host networking. Envoy reports "
                        "no healthy upstream for fault-proxy."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the final triage. Include exact tokens "
                        "ROOT_CAUSE=wrong-docker-network and FIX=same-network."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=wrong-docker-network",
                        "FIX=same-network",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="routing-policy",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We need a session-aware routing decision. Explain what "
                        "state matters before switching models."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the provided tool result before deciding.",
                    tool_name="observe_agent_phase",
                    tool_result=(
                        "The agent is inside an active tool loop and has a stable "
                        "session id. A later turn may pass an idle timeout."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the policy. Include exact tokens "
                        "DO_NOT_SWITCH=tool-loop and MAY_SWITCH=idle-timeout."
                    ),
                    expected_terms=(
                        "DO_NOT_SWITCH=tool-loop",
                        "MAY_SWITCH=idle-timeout",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="cache-reporting",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We are interpreting cache-token benchmark output. State "
                        "what must be checked."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the probe output to state the limitation.",
                    tool_name="read_cache_probe",
                    tool_result=(
                        "router: cached_token_reporting=missing; direct-backend: "
                        "cached_token_reporting=missing; both paths returned 8/8 HTTP 200."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the limitation. Include exact tokens "
                        "LIMITATION=missing-cached-token-field and "
                        "NEXT=backend-reporting."
                    ),
                    expected_terms=(
                        "LIMITATION=missing-cached-token-field",
                        "NEXT=backend-reporting",
                    ),
                ),
            ),
        ),
    )


def long_horizon_task_specs() -> tuple[TaskSpec, ...]:
    return (
        TaskSpec(
            name="multi-file-regression",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A release-blocking regression appeared after a router config "
                        "change. Build a short investigation plan."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the failed smoke output to narrow the failure.",
                    tool_name="run_agent_smoke",
                    tool_result=(
                        "make agent-smoke-local: router health OK, dashboard OK, "
                        "Envoy chat returns HTTP 503 no healthy upstream."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the generated Envoy config diff before choosing a fix.",
                    tool_name="diff_envoy_config",
                    tool_result=(
                        "runtime config changed backend base_url to fault-proxy, but "
                        "the generated Envoy cluster still points at the old upstream."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue from the same response state and state whether the "
                        "router should switch models during this diagnosis."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now convert the diagnosis into a maintainer-facing release "
                        "note and validation command."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the release-blocking fix. Include exact tokens "
                        "ROOT_CAUSE=stale-envoy-cluster, FIX=regenerate-envoy, "
                        "VALIDATE=agent-smoke-local."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=stale-envoy-cluster",
                        "FIX=regenerate-envoy",
                        "VALIDATE=agent-smoke-local",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="code-review-followup",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Review a draft PR for a session-aware routing fix. Start by "
                        "identifying what evidence you need before approving it."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the PR diff summary before naming the risk.",
                    tool_name="read_pr_diff",
                    tool_result=(
                        "The PR adds selected-model and replay-id headers to normal "
                        "responses, but looper fast responses still return before "
                        "selected-confidence and context-token-count are attached."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the CI failure before deciding the requested change.",
                    tool_name="read_ci_log",
                    tool_result=(
                        "go test ./pkg/extproc fails: expected looper response to "
                        "include x-vsr-selected-confidence and x-vsr-context-token-count."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same PR review with provider-managed state. "
                        "State whether the router should switch models during this "
                        "review loop."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now turn the review into a concrete fix request and a "
                        "validation command."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the review decision. Include exact tokens "
                        "FINDING=missing-looper-diagnostics, "
                        "FIX=emit-tracked-headers, "
                        "VALIDATE=extproc-header-tests."
                    ),
                    expected_terms=(
                        "FINDING=missing-looper-diagnostics",
                        "FIX=emit-tracked-headers",
                        "VALIDATE=extproc-header-tests",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="research-synthesis",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Synthesize the research framing for session-aware agentic "
                        "routing. Start from the problem, not from implementation."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the related-work note before choosing the claim.",
                    tool_name="read_related_work_note",
                    tool_result=(
                        "Single-turn LLM routing papers optimize per-request "
                        "cost-quality. Agent serving papers emphasize long sessions, "
                        "tool pauses, and KV-cache lifetime."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the experiment summary before deciding the evidence.",
                    tool_name="read_experiment_summary",
                    tool_result=(
                        "Synthetic policy matrix removes tool-loop and provider-state "
                        "switch violations. AMD overlay runs show long-session and "
                        "failure-recovery evidence, but branch-image and cached-token "
                        "evidence are still blocked."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the synthesis under the same provider-managed "
                        "continuation state and preserve the framing."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now translate the paper framing into a blog-safe claim that "
                        "does not overstate GA readiness."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the synthesis. Include exact tokens "
                        "CLAIM=session-continuity, EVIDENCE=policy-and-amd, "
                        "LIMITATION=not-ga-ready."
                    ),
                    expected_terms=(
                        "CLAIM=session-continuity",
                        "EVIDENCE=policy-and-amd",
                        "LIMITATION=not-ga-ready",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="maintainer-handoff",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Act as the maintainer orchestrating multiple coding agents. "
                        "Decide what should be reviewed today."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the issue board snapshot before assigning work.",
                    tool_name="read_issue_board",
                    tool_result=(
                        "v0.3 tracker is open. Session-aware routing needs fresh AMD "
                        "branch-image evidence, positive cached-token telemetry, and "
                        "expanded real-agent tasks."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the PR board snapshot before deciding next actions.",
                    tool_name="read_pr_board",
                    tool_result=(
                        "Unified semantic-router PR is draft and mergeable. Blog PR "
                        "is draft. Older scattered PRs should remain closed."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the maintainer handoff with the same provider state "
                        "and preserve the release-blocking priorities."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the maintainer handoff. Include exact tokens "
                        "ACTION=refresh-amd-evidence, ACTION=review-stale-prs, "
                        "ROUTE=maintainer-ops."
                    ),
                    expected_terms=(
                        "ACTION=refresh-amd-evidence",
                        "ACTION=review-stale-prs",
                        "ROUTE=maintainer-ops",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="cluster-boundary",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We are reviewing how ExtProc should route a selected model. "
                        "Identify which layer owns cluster choice and which layer "
                        "owns endpoint load balancing."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the Envoy routing fact.",
                    tool_name="inspect_envoy_route_config",
                    tool_result=(
                        "ExtProc can mutate request headers and body before routing; "
                        "Envoy clusters own endpoint load balancing among upstreams."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue from the same provider-managed state and decide "
                        "whether the router should own endpoint load balancing."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the maintainer boundary rule.",
                    tool_name="check_extproc_boundary",
                    tool_result=(
                        "The router may emit selected-model, selected-decision, and "
                        "cluster-routing signals, but endpoint load balancing remains "
                        "an Envoy cluster/LB concern."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the routing boundary. Include exact tokens "
                        "ROUTER=cluster-signal, ENVOY=endpoint-lb, "
                        "NO=router-owned-endpoints."
                    ),
                    expected_terms=(
                        "ROUTER=cluster-signal",
                        "ENVOY=endpoint-lb",
                        "NO=router-owned-endpoints",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="session-switch-policy",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Design a model-switch policy for a long coding-agent session. "
                        "Start by separating hard locks from soft costs."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the agent phase observation.",
                    tool_name="observe_phase",
                    tool_result=(
                        "The current turn is inside a tool-call/tool-result loop, has "
                        "a stable session id, and has previous_response_id state."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue from provider-managed state. State what the router "
                        "must not do."
                    ),
                ),
                TaskTurn(
                    phase="idle_boundary",
                    prompt=(
                        "Assume the same session later idles past the configured "
                        "timeout and starts a new subtask."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "The matched decision changed from coding-debug to research. "
                        "Decide whether reselection is allowed."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the switch policy. Include exact tokens "
                        "LOCK=tool-loop, LOCK=provider-state, RESET=idle-or-drift."
                    ),
                    expected_terms=(
                        "LOCK=tool-loop",
                        "LOCK=provider-state",
                        "RESET=idle-or-drift",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="cache-economics",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "We need to decide whether switching away from a long frontier "
                        "model session is worth it. Identify the cache accounting input."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the pricing table.",
                    tool_name="read_pricing",
                    tool_result=(
                        "frontier prompt_per_1m=15.0, cached_input_per_1m=1.5; "
                        "small prompt_per_1m=0.2, cached_input_per_1m=0.04."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the cache probe result.",
                    tool_name="read_cache_probe",
                    tool_result=(
                        "Current backend responses omit usage.prompt_tokens_details."
                        "cached_tokens, so observed cached-token ratio is unavailable."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt="Continue the cost analysis in the same session state.",
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the cache economics conclusion. Include exact tokens "
                        "COST=input-checkout-delta, FRONTIER=stricter-cache, "
                        "LIMITATION=missing-cached-token-field."
                    ),
                    expected_terms=(
                        "COST=input-checkout-delta",
                        "FRONTIER=stricter-cache",
                        "LIMITATION=missing-cached-token-field",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="release-triage",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Act as maintainer for a release candidate. Decide what evidence "
                        "is still blocking GA."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the benchmark status.",
                    tool_name="read_benchmark_status",
                    tool_result=(
                        "Local matrix complete; AMD overlay non-disruption complete; "
                        "repeat-failure recovery complete; full branch image blocked by "
                        "host disk pressure; cached-token field missing."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now classify what belongs in release notes versus remaining "
                        "GA blockers."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the PR status.",
                    tool_name="read_pr_status",
                    tool_result=(
                        "Unified PR is draft and mergeable; blog PR is open; branch-image "
                        "benchmark and positive cached-token backend evidence are missing."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the release triage. Include exact tokens "
                        "EVIDENCE=overlay-only, BLOCKER=branch-image, "
                        "BLOCKER=positive-cached-tokens."
                    ),
                    expected_terms=(
                        "EVIDENCE=overlay-only",
                        "BLOCKER=branch-image",
                        "BLOCKER=positive-cached-tokens",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="observability-debug",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A maintainer asks why the logical model is auto but the physical "
                        "model changed. Plan the observability answer."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the route trace.",
                    tool_name="read_route_trace",
                    tool_result=(
                        "trace includes current_model=qwen-small, selected_model="
                        "frontier-reasoner, decision_drift=true, idle_expired=false, "
                        "hard_locked=false, replay_id=abc123."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same explanation and decide which trace fields "
                        "should be exposed."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the dashboard fields.",
                    tool_name="read_dashboard_headers",
                    tool_result=(
                        "response headers expose x-vsr-selected-model, x-vsr-selected-"
                        "decision, and x-vsr-replay-id."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the debug answer. Include exact tokens "
                        "EXPLAIN=decision-drift, TRACE=replay-id, "
                        "HEADER=x-vsr-selected-model."
                    ),
                    expected_terms=(
                        "EXPLAIN=decision-drift",
                        "TRACE=replay-id",
                        "HEADER=x-vsr-selected-model",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="test-fix-iteration",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A coding agent has a failing session-aware routing test. "
                        "Start with the smallest evidence you need before editing."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the failed test output before naming the bug.",
                    tool_name="run_targeted_tests",
                    tool_result=(
                        "go test ./pkg/extproc fails: expected x-vsr-session-phase "
                        "on looper immediate responses, got headers without it."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the relevant code search result before proposing a patch.",
                    tool_name="rg_session_phase_header",
                    tool_result=(
                        "processor_res_header_mutation.go emits tracked headers for "
                        "normal upstream responses. req_filter_looper_response.go "
                        "builds the immediate looper response separately."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the rerun result before finalizing.",
                    tool_name="rerun_targeted_tests",
                    tool_result=(
                        "go test ./pkg/extproc -run TestCreateLooperResponseIncludes"
                        "TrackedHeaders -count=1 passed after adding the session-phase "
                        "header to the looper path."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same fix thread with provider-managed state. "
                        "State whether the router should switch models mid-debug."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the coding-agent fix summary. Include exact tokens "
                        "BUG=looper-missing-session-phase, PATCH=tracked-header-path, "
                        "VALIDATE=targeted-extproc-test."
                    ),
                    expected_terms=(
                        "BUG=looper-missing-session-phase",
                        "PATCH=tracked-header-path",
                        "VALIDATE=targeted-extproc-test",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="codebase-refactor-planning",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A coding agent is about to touch a known router hotspot. "
                        "Plan a scoped refactor before changing code."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the code search result before naming the boundary.",
                    tool_name="rg_hotspot",
                    tool_result=(
                        "processor_req_body_routing.go currently owns request parsing, "
                        "model selection, backend metadata resolution, and response "
                        "header mutation."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the module rule before choosing the patch shape.",
                    tool_name="read_module_boundary",
                    tool_result=(
                        "Hotspot files must not grow in responsibility. Prefer a small "
                        "orchestrator plus helpers, and validate affected extproc tests."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same refactor thread with provider-managed state "
                        "and preserve the current model while reviewing tool output."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now convert the refactor plan into a PR review checklist "
                        "without adding unrelated cleanup."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the refactor plan. Include exact tokens "
                        "PLAN=extract-helper, BOUNDARY=small-orchestrator, "
                        "VALIDATE=check."
                    ),
                    expected_terms=(
                        "PLAN=extract-helper",
                        "BOUNDARY=small-orchestrator",
                        "VALIDATE=check",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="research-artifact-review",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Review the paper evidence package for a session-aware routing "
                        "submission. Start by deciding what belongs near the main claim."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the figure inventory before choosing what to keep.",
                    tool_name="read_figure_inventory",
                    tool_result=(
                        "Figures 7, 9, 10, 11, 12, 13, 16, and 17 mix multiple "
                        "results per image and drift away from the sections that "
                        "explain them."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the result table before deciding the evidence boundary.",
                    tool_name="read_result_table",
                    tool_result=(
                        "Synthetic matrix and ablation pass as policy evidence. "
                        "Branch-image AMD and positive cached-token backend evidence "
                        "are still missing."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same paper review with provider state and keep "
                        "the claim boundary unchanged."
                    ),
                ),
                TaskTurn(
                    phase="idle_boundary",
                    prompt=(
                        "Assume the author returns after an idle gap and asks whether "
                        "the blog should include every experiment detail."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the paper review. Include exact tokens "
                        "FIGURE=single-claim, APPENDIX=process-detail, "
                        "LIMITATION=missing-amd-evidence."
                    ),
                    expected_terms=(
                        "FIGURE=single-claim",
                        "APPENDIX=process-detail",
                        "LIMITATION=missing-amd-evidence",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="tool-error-recovery-loop",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A long-running coding agent hit a tool failure while "
                        "validating session-aware routing. Decide how to recover."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the failed command before naming the likely cause.",
                    tool_name="run_branch_image_probe",
                    tool_result=(
                        "docker run ghcr.io/example/vllm-sr:pr1989 failed with "
                        "image not found; local mounted-binary diagnostic still passes."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the build log before deciding whether this is GA evidence.",
                    tool_name="read_image_build_log",
                    tool_result=(
                        "ROCm branch image build was skipped after disk pressure. "
                        "No immutable image tag or full branch-image summary exists."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the rerun plan before finalizing recovery.",
                    tool_name="plan_recovery_run",
                    tool_result=(
                        "Free disk, rebuild the branch image, rerun diagnostic, "
                        "long-session, failure-recovery, and agent-task summaries, "
                        "then assemble full-branch-image-benchmark evidence."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same recovery thread under provider-managed "
                        "state and preserve the active session."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the recovery plan. Include exact tokens "
                        "ROOT_CAUSE=missing-branch-image, "
                        "RECOVERY=rebuild-and-rerun, PRESERVE=session-state."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=missing-branch-image",
                        "RECOVERY=rebuild-and-rerun",
                        "PRESERVE=session-state",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="paper-evidence-audit",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Audit a paper section before submission. Separate method "
                        "claims from GA release claims."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the paper excerpt before identifying the overclaim.",
                    tool_name="read_paper_excerpt",
                    tool_result=(
                        "The draft says the system is ready for release after "
                        "synthetic and overlay AMD runs, but the branch-image "
                        "benchmark and positive cached-token backend evidence are "
                        "still missing."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the machine-readable GA report before rewriting.",
                    tool_name="read_ga_report",
                    tool_result=(
                        "ga_ready=false; blockers are branch_image_amd_validation, "
                        "amd_agent_task_quality_1, and cache_token_reporting."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now translate the correction into a blog-safe limitation "
                        "sentence without experiment-log detail."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same evidence audit under provider-managed "
                        "state and preserve the limitation boundary."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the paper audit. Include exact tokens "
                        "CLAIM=method-evidence, BLOCKER=ga-evidence, "
                        "FIX=separate-readiness."
                    ),
                    expected_terms=(
                        "CLAIM=method-evidence",
                        "BLOCKER=ga-evidence",
                        "FIX=separate-readiness",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="multi-agent-delegation",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Act as a maintainer coordinating coding agents for the next "
                        "release. Decide which agents to spawn and what each must prove."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the release board before delegating.",
                    tool_name="read_release_board",
                    tool_result=(
                        "Open work: branch-image AMD benchmark, expanded agent-task AMD "
                        "rerun, cached-token backend probe, blog draft, and PR review."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the delegation with provider-managed state. Keep "
                        "the same priorities and do not switch away during handoff."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the sub-agent status update before final scheduling.",
                    tool_name="read_subagent_status",
                    tool_result=(
                        "AMD agent is waiting on image build capacity; docs agent has "
                        "blog draft green; reviewer agent says semantic-router PR still "
                        "needs evidence blockers documented."
                    ),
                ),
                TaskTurn(
                    phase="idle_boundary",
                    prompt=(
                        "Assume the maintainer returns after an idle interval. Decide "
                        "what can be safely reselected and what state must be preserved."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the multi-agent plan. Include exact tokens "
                        "DELEGATE=amd-evidence, DELEGATE=docs-review, "
                        "PRESERVE=release-blockers."
                    ),
                    expected_terms=(
                        "DELEGATE=amd-evidence",
                        "DELEGATE=docs-review",
                        "PRESERVE=release-blockers",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="issue-pr-maintenance-loop",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Act as maintainer after a release planning pass. Reconcile "
                        "the issue and PR board before assigning agents."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the GitHub board snapshot before deciding actions.",
                    tool_name="read_github_board",
                    tool_result=(
                        "v0.3 tracker is open. Missing release-blocker issues: AMD "
                        "branch-image benchmark and positive cached-token backend "
                        "evidence. Unified PR #1989 is draft. Older split PRs are "
                        "conflicted or closed. Blog PR #224 is draft."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the maintainer policy before updating the board.",
                    tool_name="read_maintainer_policy",
                    tool_result=(
                        "Milestone-bound release blockers stay attached to the release "
                        "plan. New user requests and bugs are triaged separately. Daily "
                        "issue and PR state belongs under .agent-harness/maintainer."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same maintainer thread with provider-managed "
                        "state and preserve the release scope."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now convert the board reconciliation into a short next-action "
                        "summary without creating GitHub objects."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the maintainer board action. Include exact tokens "
                        "CREATE=release-blocker-issues, CLOSE=stale-split-prs, "
                        "TRACK=maintainer-board."
                    ),
                    expected_terms=(
                        "CREATE=release-blocker-issues",
                        "CLOSE=stale-split-prs",
                        "TRACK=maintainer-board",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="configuration-contract-review",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Review a configuration contract change for session-aware "
                        "routing. Start by identifying the schema and ownership risks."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the schema diff before naming the contract.",
                    tool_name="read_config_schema_diff",
                    tool_result=(
                        "Model pricing now carries cached_input_per_1m alongside "
                        "prompt_per_1m and completion_per_1m. Unknown pricing fields "
                        "must be rejected by config loaders."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the boundary audit before deciding what must not change.",
                    tool_name="read_extproc_boundary_audit",
                    tool_result=(
                        "Semantic Router is an Envoy ExtProc component. It may emit "
                        "cluster-level routing signals and diagnostics, but endpoint "
                        "load balancing remains owned by Envoy clusters."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same review with provider-managed state and "
                        "preserve the selected model while inspecting tool output."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now translate the review into a validation plan that avoids "
                        "endpoint-selection overclaims."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the configuration review. Include exact tokens "
                        "SCHEMA=cached-input-rate, BOUNDARY=no-endpoint-selection, "
                        "VALIDATE=config-loader-tests."
                    ),
                    expected_terms=(
                        "SCHEMA=cached-input-rate",
                        "BOUNDARY=no-endpoint-selection",
                        "VALIDATE=config-loader-tests",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="repo-bisect-debug",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A coding agent is chasing a regression that only appears in "
                        "the session-aware routing branch. Start with a bisect plan."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the failing targeted test before selecting the next step.",
                    tool_name="run_targeted_test",
                    tool_result=(
                        "go test ./pkg/extproc -run TestSessionPhaseHeaders "
                        "fails: selected model is present but session phase is empty."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the git bisect result before naming the fix.",
                    tool_name="run_git_bisect",
                    tool_result=(
                        "First bad commit changed looper immediate responses to bypass "
                        "the shared tracked-header writer."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same debug session with provider-managed state. "
                        "State whether the router may switch models mid-bisect."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now turn the root cause into a short PR note and a validation "
                        "command."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the bisect result. Include exact tokens "
                        "ROOT_CAUSE=looper-header-bypass, FIX=shared-header-writer, "
                        "VALIDATE=session-phase-test."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=looper-header-bypass",
                        "FIX=shared-header-writer",
                        "VALIDATE=session-phase-test",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="dependency-upgrade-regression",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A CI-only operator integration failure appeared after a "
                        "dependency install path changed. Plan the investigation."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the failing CI log before naming the dependency risk.",
                    tool_name="read_ci_log",
                    tool_result=(
                        "make install failed while fetching raw.githubusercontent.com/"
                        "kubernetes-sigs/kustomize/master/hack/install_kustomize.sh: "
                        "Github rate-limiter failed the request."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the Makefile snippet before proposing the repair.",
                    tool_name="read_operator_makefile",
                    tool_result=(
                        "controller-gen already installs through go install. kustomize "
                        "is the only operator tool that still depends on a raw GitHub "
                        "install script."
                    ),
                ),
                TaskTurn(
                    phase="idle_boundary",
                    prompt=(
                        "Assume the same agent resumes after an idle gap and sees "
                        "other operator backends pass. Decide the smallest fix."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same CI thread with provider-managed state and "
                        "preserve the failure context."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the CI repair. Include exact tokens "
                        "ROOT_CAUSE=raw-script-rate-limit, FIX=go-install-kustomize, "
                        "VALIDATE=operator-ci-gate."
                    ),
                    expected_terms=(
                        "ROOT_CAUSE=raw-script-rate-limit",
                        "FIX=go-install-kustomize",
                        "VALIDATE=operator-ci-gate",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="literature-data-extraction",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Prepare a related-work and evidence note for a paper on "
                        "session-aware agentic routing."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the literature extraction before framing the gap.",
                    tool_name="extract_related_work",
                    tool_result=(
                        "Routing papers optimize per-request cost-quality. Agent "
                        "serving papers discuss tool pauses, long sessions, and "
                        "KV-cache lifetime."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the experiment table before choosing the evidence claim.",
                    tool_name="extract_result_table",
                    tool_result=(
                        "Synthetic matrix removes unsafe switches; ablation identifies "
                        "locks and reset boundaries; AMD overlay validates serving-path "
                        "feasibility but not full branch-image release evidence."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same research note with provider-managed state "
                        "and keep the claim boundary unchanged."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the paper review note before finalizing the output.",
                    tool_name="read_paper_review_note",
                    tool_result=(
                        "Do not present experimental process in the blog. Keep process "
                        "detail in the paper experiment section or appendix."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the research extraction. Include exact tokens "
                        "GAP=single-turn-routing, EVIDENCE=session-policy, "
                        "DESTINATION=paper-not-blog."
                    ),
                    expected_terms=(
                        "GAP=single-turn-routing",
                        "EVIDENCE=session-policy",
                        "DESTINATION=paper-not-blog",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="stale-pr-rebase-triage",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Act as maintainer reviewing a draft PR that has been idle "
                        "while main moved forward. Decide what evidence is needed "
                        "before keeping it open."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the PR diff and conflict summary before deciding.",
                    tool_name="read_pr_conflicts",
                    tool_result=(
                        "The PR edits session-aware routing and benchmark thresholds. "
                        "It now conflicts with main in the GA report defaults and its "
                        "CI run is older than the latest branch-image evidence changes."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the maintainer PR policy before choosing an action.",
                    tool_name="read_pr_policy",
                    tool_result=(
                        "Stale split PRs that diverge from the unified branch should "
                        "be closed. Useful work should be rebased into the unified PR "
                        "and rerun with fresh CI."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same PR review under provider-managed state and "
                        "preserve the stale-branch context."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now turn the review into a concise maintainer decision and "
                        "validation expectation."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the stale PR triage. Include exact tokens "
                        "DECISION=close-or-rebase, BLOCKER=conflicts-and-drift, "
                        "VALIDATE=fresh-ci."
                    ),
                    expected_terms=(
                        "DECISION=close-or-rebase",
                        "BLOCKER=conflicts-and-drift",
                        "VALIDATE=fresh-ci",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="benchmark-regression-root-cause",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A new benchmark summary looks slower than the previous GA "
                        "report. Start by separating measurement drift from a real "
                        "routing regression."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the long-session comparison before naming the metric.",
                    tool_name="read_long_session_comparison",
                    tool_result=(
                        "Balanced and stateful sessions have low p95 router overhead. "
                        "The idle workload includes real wall-clock sleeps, so its p95 "
                        "overhead is not a hot-path regression signal."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the evidence identity check before deciding release risk.",
                    tool_name="read_evidence_identity",
                    tool_result=(
                        "The summary came from an overlay run, not the required "
                        "branch-image run. It is useful serving-path evidence but "
                        "cannot satisfy the branch-image GA blocker."
                    ),
                ),
                TaskTurn(
                    phase="idle_boundary",
                    prompt=(
                        "Assume the benchmark owner resumes after an idle interval "
                        "with the same summary and asks whether the branch can release."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same regression analysis under provider-managed "
                        "state and keep the evidence boundary intact."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the benchmark diagnosis. Include exact tokens "
                        "METRIC=p95-overhead, ROOT_CAUSE=stale-baseline, "
                        "ACTION=rerun-branch-image."
                    ),
                    expected_terms=(
                        "METRIC=p95-overhead",
                        "ROOT_CAUSE=stale-baseline",
                        "ACTION=rerun-branch-image",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="paper-figure-quality-review",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Review paper and blog figures for the session-aware routing "
                        "draft. Start by identifying what makes a figure too dense."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the figure audit before deciding the fix.",
                    tool_name="inspect_figures",
                    tool_result=(
                        "Several charts combine readiness thresholds, result values, "
                        "and long explanatory notes. The paper needs low-density, "
                        "single-conclusion figures placed near the relevant text."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the blog standard before choosing what belongs there.",
                    tool_name="read_blog_standard",
                    tool_result=(
                        "The blog should explain why single-turn routing is not "
                        "enough, how ACR works, key results, and current limitations. "
                        "Detailed experiment process belongs in the paper."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same figure review under provider-managed state "
                        "and preserve the paper-versus-blog boundary."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now convert the review into an action for the next paper/blog "
                        "revision."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the figure review. Include exact tokens "
                        "FIGURE=single-conclusion, FIX=split-dense-chart, "
                        "DESTINATION=paper-near-text."
                    ),
                    expected_terms=(
                        "FIGURE=single-conclusion",
                        "FIX=split-dense-chart",
                        "DESTINATION=paper-near-text",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="feature-implementation-loop",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A coding agent must implement a small session-aware routing "
                        "fix without expanding a hotspot file. Start by identifying "
                        "the minimal code path to inspect."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the code search result before naming the change.",
                    tool_name="rg_session_memory_guard",
                    tool_result=(
                        "session_aware.go owns stay-vs-switch scoring; "
                        "req_filter_classification.go builds AgenticSessionContext; "
                        "processor_res_header_mutation.go emits response evidence."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the targeted test output before deciding the patch.",
                    tool_name="run_targeted_selection_tests",
                    tool_result=(
                        "go test ./pkg/selection -run TestSessionAwareSelector "
                        "passes except the new case for provider-state continuity, "
                        "which needs HasNonPortableContext in the constructed session."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same implementation session under provider-managed "
                        "state and keep the current model pinned while reviewing tools."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now convert the code fix into reviewer-facing validation and "
                        "docs notes."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the implementation summary. Include exact tokens "
                        "IMPLEMENT=session-memory-guard, TEST=targeted-selection, "
                        "DOC=scope-boundary."
                    ),
                    expected_terms=(
                        "IMPLEMENT=session-memory-guard",
                        "TEST=targeted-selection",
                        "DOC=scope-boundary",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="research-claim-grounding-loop",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "Ground a paper claim for session-aware agentic routing. Start "
                        "by separating the method contribution from release readiness."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the related-work extraction before framing the gap.",
                    tool_name="read_related_work_extract",
                    tool_result=(
                        "Model routing literature focuses on single-request model "
                        "choice. Agent-serving notes emphasize multi-turn state, "
                        "tool pauses, and reusable prefixes."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the evidence audit before writing the claim.",
                    tool_name="read_evidence_audit",
                    tool_result=(
                        "Synthetic matrix and ablation are policy evidence. AMD "
                        "overlay evidence is serving-path evidence. Branch-image and "
                        "backend-positive cache accounting are still release blockers."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same research note with provider-managed state "
                        "and keep the claim boundary stable."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now turn the grounded claim into one blog-safe sentence that "
                        "does not read like an experiment log."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the grounded claim. Include exact tokens "
                        "CLAIM=long-horizon-routing, "
                        "EVIDENCE=separate-policy-and-ga, "
                        "LIMITATION=backend-cache-proof."
                    ),
                    expected_terms=(
                        "CLAIM=long-horizon-routing",
                        "EVIDENCE=separate-policy-and-ga",
                        "LIMITATION=backend-cache-proof",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="tool-timeout-retry-loop",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A long-running coding agent is validating GA evidence when a "
                        "tool call times out. Decide how to continue without losing "
                        "the session state."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the timeout result before deciding recovery.",
                    tool_name="run_live_agent_task_suite",
                    tool_result=(
                        "agent_task_live_benchmark.py timed out after 120s during "
                        "task repetition 2. Prior turns emitted replay ids and "
                        "session-phase headers."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the partial artifact before deciding what to report.",
                    tool_name="read_partial_summary",
                    tool_result=(
                        "summary.json records validation_failures with timeout=true, "
                        "successful_requests=281, and no tool-loop switch violations "
                        "before the timeout."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the retry plan before finalizing.",
                    tool_name="plan_retry",
                    tool_result=(
                        "Keep the same session prefix, rerun the same suite with a "
                        "higher timeout, and keep the partial validation failure "
                        "attached to the GA evidence trail."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same retry thread under provider-managed state "
                        "and preserve the timeout evidence."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the retry decision. Include exact tokens "
                        "RECOVER=retry-same-session, PRESERVE=tool-loop-lock, "
                        "REPORT=validation-failure."
                    ),
                    expected_terms=(
                        "RECOVER=retry-same-session",
                        "PRESERVE=tool-loop-lock",
                        "REPORT=validation-failure",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="ci-patch-review-loop",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A coding agent must fix a CI-only failure in the session-aware "
                        "routing PR. Start by deciding what evidence to inspect before "
                        "editing."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the failing GitHub Actions excerpt before naming the bug.",
                    tool_name="read_github_actions_log",
                    tool_result=(
                        "Pre-commit failed in agent CI lint after changed-files "
                        "classification included a generated paper artifact. Local "
                        "make check passed after removing the artifact."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the worktree audit before deciding the patch.",
                    tool_name="git_status_and_diff_check",
                    tool_result=(
                        "Only source files should be staged. The PDF build artifact "
                        "must stay untracked, and git diff --check must remain clean."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same CI patch session with provider-managed "
                        "state and keep the current model while reading tool output."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the rerun result before finalizing the review note.",
                    tool_name="rerun_pr_gate",
                    tool_result=(
                        "make check on changed files passed; GitHub build-paper "
                        "and unit checks are green; pre-commit is rerunning."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the CI patch review. Include exact tokens "
                        "BUG=generated-artifact-staged, FIX=source-only-commit, "
                        "VALIDATE=check."
                    ),
                    expected_terms=(
                        "BUG=generated-artifact-staged",
                        "FIX=source-only-commit",
                        "VALIDATE=check",
                    ),
                ),
            ),
        ),
        TaskSpec(
            name="paper-rebuttal-revision-loop",
            suite="long-horizon",
            turns=(
                TaskTurn(
                    phase="user_turn",
                    prompt=(
                        "A reviewer says the session-aware routing paper reads like an "
                        "experiment log. Start by planning a revision that improves the "
                        "scientific narrative."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the reviewer note before deciding the main edit.",
                    tool_name="read_reviewer_note",
                    tool_result=(
                        "The paper should foreground the problem formulation, hard "
                        "continuity constraints, switch-cost model, and limitations. "
                        "Dense experiment process should move to the appendix."
                    ),
                ),
                TaskTurn(
                    phase="tool_loop",
                    prompt="Use the figure audit before choosing how to revise figures.",
                    tool_name="read_figure_audit",
                    tool_result=(
                        "Result figures should each support one conclusion and appear "
                        "near the text that interprets them. Blog figures should stay "
                        "lower density than paper result plots."
                    ),
                ),
                TaskTurn(
                    phase="provider_state",
                    prompt=(
                        "Continue the same rebuttal with provider-managed state and "
                        "preserve the claim boundary."
                    ),
                ),
                TaskTurn(
                    phase="topic_drift",
                    prompt=(
                        "Now turn the rebuttal into a concrete revision plan without "
                        "claiming GA readiness."
                    ),
                ),
                TaskTurn(
                    phase="final",
                    prompt=(
                        "Return the paper revision plan. Include exact tokens "
                        "NARRATIVE=problem-to-limitations, FIGURE=one-conclusion, "
                        "CLAIM=no-ga-ready."
                    ),
                    expected_terms=(
                        "NARRATIVE=problem-to-limitations",
                        "FIGURE=one-conclusion",
                        "CLAIM=no-ga-ready",
                    ),
                ),
            ),
        ),
    )


def build_messages(task: TaskSpec, turn_index: int) -> list[dict[str, Any]]:
    turn = task.turns[turn_index]
    system = {"role": "system", "content": SYSTEM_INSTRUCTIONS}
    messages: list[dict[str, Any]] = [system]
    for prior_index, prior in enumerate(task.turns[:turn_index]):
        messages.append({"role": "user", "content": prior.prompt})
        if prior.expected_terms:
            messages.append({"role": "assistant", "content": "Acknowledged."})
        elif prior.tool_result:
            messages.extend(
                tool_messages(task_name=task.name, turn_index=prior_index, turn=prior)
            )
        else:
            messages.append(
                {"role": "assistant", "content": "I will inspect the evidence."}
            )
    if turn.tool_result:
        messages.extend(
            tool_messages(task_name=task.name, turn_index=turn_index, turn=turn)
        )
    prompt = turn.prompt
    if turn.expected_terms:
        prompt = prompt + "\n\n" + scoring_instruction(turn.expected_terms)
    messages.append({"role": "user", "content": prompt})
    return messages


def scoring_instruction(expected_terms: tuple[str, ...]) -> str:
    tokens = "\n".join(f"- {term}" for term in expected_terms)
    return (
        "Scoring uses exact substring matching. Copy every required token "
        "exactly as written, one token per line, before any explanation:\n"
        f"{tokens}"
    )


def tool_messages(
    task_name: str, turn_index: int, turn: TaskTurn
) -> list[dict[str, Any]]:
    call_id = f"call_{task_name}_{turn_index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": turn.tool_name or "read_tool_result",
                        "arguments": json.dumps({"task": task_name}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": turn.tool_result},
    ]


def score_answer(
    content: str, expected_terms: tuple[str, ...]
) -> tuple[float | None, list[str]]:
    if not expected_terms:
        return None, []
    normalized = content.lower()
    missing = [term for term in expected_terms if term.lower() not in normalized]
    found = len(expected_terms) - len(missing)
    return round(found / len(expected_terms), 4), missing
