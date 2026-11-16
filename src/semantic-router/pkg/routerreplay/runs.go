package routerreplay

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// Markers a run call carries, and counts a run's events by.
const (
	// RunMarkerSwitch is a call its agent's previous call made on another model.
	RunMarkerSwitch = "switch"
	// RunMarkerEscalation is a Looper call a later attempt than the first answered.
	RunMarkerEscalation = "escalation"
	// RunMarkerGuard is a call a guard flagged: jailbreak, PII or hallucination.
	RunMarkerGuard = "guard"
	// RunMarkerBlock is a call a guard refused.
	RunMarkerBlock = "block"
	// RunMarkerFallback is a call another model answered after its own failed.
	RunMarkerFallback = "fallback"
	// RunMarkerFailure is a call that failed or got an error status.
	RunMarkerFailure = "failure"
	// RunMarkerToolFailure is a call that carried a failed tool result.
	RunMarkerToolFailure = "tool_failure"
)

// Run outcome statuses: the main agent's last call answered, failed, was
// abandoned by the client, or is still running.
const (
	RunStatusAnswered   = "answered"
	RunStatusFailed     = "failed"
	RunStatusAborted    = "aborted"
	RunStatusInProgress = "in_progress"
	RunStatusUnknown    = "unknown"
)

// RunListResponse lists agent runs, one per session, most recent activity
// first.
type RunListResponse struct {
	Object     string       `json:"object"`
	Count      int          `json:"count"`
	Total      int          `json:"total"`
	Limit      int          `json:"limit"`
	Offset     int          `json:"offset"`
	HasMore    bool         `json:"has_more"`
	NextOffset *int         `json:"next_offset,omitempty"`
	Data       []RunSummary `json:"data"`
}

// RunSummary is one agent run at a glance: the session, the name the
// harness gave it, the harness, the task its main agent was given when Replay
// kept the prompt, its agents, the models that answered, how it ended, what
// it cost and took, and the events worth a look.
type RunSummary struct {
	SessionID  string           `json:"session_id"`
	Name       string           `json:"session_name,omitempty"`
	Harness    string           `json:"harness,omitempty"`
	Prompt     string           `json:"prompt,omitempty"`
	StartedAt  time.Time        `json:"started_at"`
	EndedAt    time.Time        `json:"ended_at"`
	DurationMS int64            `json:"duration_ms"`
	Calls      int              `json:"calls"`
	Agents     []RunAgent       `json:"agents"`
	Models     []AggregateValue `json:"models"`
	Outcome    RunOutcome       `json:"outcome"`
	Cost       RunCost          `json:"cost"`
	Events     map[string]int   `json:"events,omitempty"`
}

// RunAgent is one agent of a run. Label names it in the run, main or
// subagent N by first call; Parent is the label of the agent that spawned
// it. ID is the short hash of the agent id the harness sent.
// SpawnedAfterRecordID is the parent agent's last call before this agent's
// first one, where the diagram marks the spawn.
type RunAgent struct {
	Label                string    `json:"label"`
	ID                   string    `json:"id,omitempty"`
	Parent               string    `json:"parent,omitempty"`
	Kind                 string    `json:"kind,omitempty"`
	Calls                int       `json:"calls"`
	FirstCallAt          time.Time `json:"first_call_at"`
	LastCallAt           time.Time `json:"last_call_at"`
	SpawnedAfterRecordID string    `json:"spawned_after_record_id,omitempty"`
	// Task is what the spawning call's tool call asked the agent for, such as
	// its subagent type and description, when Replay captured its arguments.
	Task string `json:"task,omitempty"`
	// ReturnedToRecordID is the parent's first call after the agent's last
	// one, the call that took its result, and ReturnStatus how it came back:
	// failed when the agent's last call or its spawn tool's result failed.
	ReturnedToRecordID string `json:"returned_to_record_id,omitempty"`
	ReturnStatus       string `json:"return_status,omitempty"`
	// OfferedTools names the tools the agent's first turn offered the
	// model; a call's OfferedToolsAdded and OfferedToolsRemoved say where
	// that changed, as when Claude Code's ToolSearch loads deferred tools.
	OfferedTools []string `json:"offered_tools,omitempty"`
	Models       []string `json:"models,omitempty"`
	Cost         RunCost  `json:"cost"`
}

// RunOutcome is how a run ended: Status from its main agent's last call,
// the record of the final answer when that call gave one, the tool results
// its calls carried and how many failed, and the outcomes reported for its
// calls through the outcome API.
type RunOutcome struct {
	Status              string         `json:"status"`
	FinalAnswerRecordID string         `json:"final_answer_record_id,omitempty"`
	ToolResults         int            `json:"tool_results"`
	ToolFailures        int            `json:"tool_failures"`
	Reported            []RunOutcomeAt `json:"reported,omitempty"`
}

// RunOutcomeAt is an outcome reported for one call of the run.
type RunOutcomeAt struct {
	RecordID string        `json:"record_id"`
	Outcome  store.Outcome `json:"outcome"`
}

// RunCost totals a run's recorded costs per currency as the aggregate's
// total_spend does: each call's own cost and the calls the Router made for
// it. UnpricedRecords counts the calls missing all or part of their price,
// and UnpricedRouterCalls the Router's own calls without one; none is
// counted as free. Tokens add the calls that reported usage.
type RunCost struct {
	ByCurrency          []RunCurrencyCost `json:"by_currency,omitempty"`
	UnpricedRecords     int               `json:"unpriced_records"`
	UnpricedRouterCalls int               `json:"unpriced_router_calls"`
	PromptTokens        int               `json:"prompt_tokens"`
	CompletionTokens    int               `json:"completion_tokens"`
}

// RunCurrencyCost is one currency's total, the Router's own calls apart.
type RunCurrencyCost struct {
	Currency        string  `json:"currency"`
	Total           float64 `json:"total"`
	RouterCallSpend float64 `json:"router_call_spend"`
}

// RunResponse is one agent run as a tree: its summary, whose agents each
// name their parent, and every call in time order naming its agent, with
// the Looper hops a call made nested under it.
type RunResponse struct {
	Object string     `json:"object"`
	Run    RunSummary `json:"run"`
	Calls  []RunCall  `json:"calls"`
}

// RunCall is one call of a run: who made it, how the Router routed it and
// why, what answered, what it cost, and how it ended. Markers flag the
// events a diagram shows; Reason holds protection's or the Looper's reason.
type RunCall struct {
	RecordID         string                 `json:"record_id"`
	RequestID        string                 `json:"request_id,omitempty"`
	Agent            string                 `json:"agent"`
	Timestamp        time.Time              `json:"timestamp"`
	DurationMS       int64                  `json:"duration_ms"`
	Recipe           string                 `json:"recipe,omitempty"`
	Entrypoint       string                 `json:"entrypoint,omitempty"`
	Decision         string                 `json:"decision,omitempty"`
	Signals          []string               `json:"signals,omitempty"`
	Model            string                 `json:"model,omitempty"`
	PreviousModel    string                 `json:"previous_model,omitempty"`
	Effort           string                 `json:"effort,omitempty"`
	SelectionMethod  string                 `json:"selection_method,omitempty"`
	SessionAction    string                 `json:"session_action,omitempty"`
	Reason           string                 `json:"reason,omitempty"`
	RequestKind      string                 `json:"request_kind,omitempty"`
	LifecycleState   string                 `json:"lifecycle_state"`
	ResponseStatus   int                    `json:"response_status,omitempty"`
	TerminalReason   string                 `json:"terminal_reason,omitempty"`
	PromptTokens     *int                   `json:"prompt_tokens,omitempty"`
	CompletionTokens *int                   `json:"completion_tokens,omitempty"`
	Cost             *float64               `json:"cost,omitempty"`
	Currency         string                 `json:"currency,omitempty"`
	PricingSource    string                 `json:"pricing_source,omitempty"`
	RouterCalls      []store.RouterCallCost `json:"router_calls,omitempty"`
	LooperAttempts   int                    `json:"looper_attempts,omitempty"`
	ToolResults      int                    `json:"tool_results,omitempty"`
	ToolFailures     int                    `json:"tool_failures,omitempty"`
	FinalAnswer      bool                   `json:"final_answer,omitempty"`
	Outcomes         []store.Outcome        `json:"outcomes,omitempty"`
	Markers          []string               `json:"markers,omitempty"`
	ToolCalls        []RunToolCall          `json:"tool_calls,omitempty"`
	// The tools this turn offered beyond the agent's previous turn, and those
	// it no longer offered.
	OfferedToolsAdded   []string  `json:"offered_tools_added,omitempty"`
	OfferedToolsRemoved []string  `json:"offered_tools_removed,omitempty"`
	Hops                []RunCall `json:"hops,omitempty"`
}

// RunToolCall is a tool call the model made in a call's turn, as far as
// Replay kept it: its name always, its arguments only when the route
// captured them, and the status of the result the agent's next request
// carried, succeeded or failed, once there was one.
type RunToolCall struct {
	Name string `json:"name"`
	ID   string `json:"id,omitempty"`
	// Summary is the argument a person reads the call by, such as a command
	// or a file name, and Arguments the arguments on one bounded line.
	Summary   string `json:"summary,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Status    string `json:"status,omitempty"`
	// TimingSource says where the times come from: gateway, as the Router
	// sees them, or plugin, as a harness plugin reports them.
	TimingSource string `json:"timing_source,omitempty"`
	// From the gateway: RequestedAt is when the response that asked for the
	// call ended, ReturnedAt when the agent's next request that carried its
	// result began, and ClientMS the time between, the tool's execution,
	// any approval wait and the harness's own time. The tool calls of one
	// response share that window.
	RequestedAt *time.Time `json:"requested_at,omitempty"`
	ReturnedAt  *time.Time `json:"returned_at,omitempty"`
	ClientMS    *int64     `json:"client_ms,omitempty"`
	// From a harness plugin, joined by the tool call id (Claude Code's
	// tool_use id, Codex's call_id): when the tool ran and how long it waited
	// for approval. Empty from the gateway.
	StartedAt  *time.Time `json:"started_at,omitempty"`
	EndedAt    *time.Time `json:"ended_at,omitempty"`
	ApprovalMS *int64     `json:"approval_ms,omitempty"`
}

// ModelListResponse lists the models the Router's configuration holds, in
// name order, each with the label Insights shows for it.
type ModelListResponse struct {
	Object string       `json:"object"`
	Data   []ModelLabel `json:"data"`
}

// ModelLabel is a configured model's name and how a person reads it: its
// card's display name when the catalog or the operator gives one, else the
// display name of the catalog card its provider model id maps to, else that
// provider model id. A label two models would share reads as each one's
// configured name instead.
type ModelLabel struct {
	Name  string `json:"name"`
	Label string `json:"label"`
}
