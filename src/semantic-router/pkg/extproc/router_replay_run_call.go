package extproc

import (
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// replayRunEntry is one record of a run reduced to what the run view
// shows, so a scan over many sessions keeps no record bodies.
type replayRunEntry struct {
	call            routerreplay.RunCall
	agent           store.AgentIdentity
	parentRequestID string
	// Tool calls whose results the call carried, and those that failed.
	toolResults, toolFailures []string
}

func newReplayRunEntry(record routerreplay.RoutingRecord) replayRunEntry {
	entry := replayRunEntry{call: buildReplayRunCall(record), parentRequestID: record.ParentRequestID}
	if record.Agent != nil {
		entry.agent = *record.Agent
	}
	entry.agent.ID = agentOf(record)
	entry.toolResults, entry.toolFailures = replayNewToolResults(record.ToolTrace)
	entry.call.ToolResults, entry.call.ToolFailures = len(entry.toolResults), len(entry.toolFailures)
	if entry.call.ToolResults > 0 {
		entry.call.Markers = append(entry.call.Markers, routerreplay.RunMarkerToolLoop)
	}
	if entry.call.ToolFailures > 0 {
		entry.call.Markers = append(entry.call.Markers, routerreplay.RunMarkerToolFailure)
	}
	return entry
}

func buildReplayRunCall(record routerreplay.RoutingRecord) routerreplay.RunCall {
	call := routerreplay.RunCall{
		RecordID: record.ID, RequestID: record.RequestID, Timestamp: record.Timestamp,
		DurationMS: record.DurationMS, Recipe: record.Recipe, Entrypoint: record.OriginalModel,
		Decision: record.Decision, Signals: replayMatchedSignals(record.Signals), Model: record.SelectedModel,
		Effort: firstNonEmptyString(record.ReasoningEffort, record.ReasoningMode), SelectionMethod: record.SelectionMethod, RequestKind: record.RequestKind,
		LifecycleState: record.LifecycleState, ResponseStatus: record.ResponseStatus, TerminalReason: record.TerminalReason,
		PromptTokens: record.PromptTokens, CompletionTokens: record.CompletionTokens,
		RouterCalls: replayRouterCalls(record), FinalAnswer: replayEndsWithFinalAnswer(record.ToolTrace),
		Outcomes: replayReportedOutcomes(record.Outcomes), ImageCount: record.ImageCount,
	}
	if record.Agent != nil {
		call.RequestClass, call.AgentType = record.Agent.RequestClass, record.Agent.Type
	}
	if currency := replayRecordCurrency(record); currency != "" && finiteReplayCost(record.ActualCost) {
		call.Cost, call.Currency = record.ActualCost, currency
	}
	reasons := []string{}
	if diagnostics := record.RouteDiagnostics; diagnostics != nil {
		call.SessionAction = diagnostics.SessionAction
		reasons = append(reasons, firstNonEmptyString(diagnostics.SessionReason, diagnostics.HardLockReason, diagnostics.DecisionReason))
		if looper := diagnostics.Looper; looper != nil {
			call.LooperAttempts = len(looper.Attempts) + looper.DroppedAttemptCount
			reasons = append(reasons, replayLooperReason(looper))
		}
	}
	call.Reason = strings.Join(nonEmptyStrings(reasons), "; ")
	call.Markers = replayRunCallMarkers(record)
	return call
}

// replayRunCallMarkers flags what a diagram marks on a call by itself; a
// model switch needs the agent's previous call and is marked by the run.
func replayRunCallMarkers(record routerreplay.RoutingRecord) []string {
	var markers []string
	guard := record.JailbreakDetected || record.PIIDetected || record.HallucinationDetected
	// A guard decision that answers with its fast response refuses with a
	// 200 and calls no model.
	refused := (record.JailbreakDetected || record.PIIDetected) && record.SelectionMethod == "fast_response"
	block := record.PIIBlocked || refused || (guard && record.ResponseStatus >= 400 && record.ResponseStatus < 500)
	failed := !block && (record.LifecycleState == routerreplay.LifecycleFailed || record.ResponseStatus >= 400)
	requestClass := ""
	if record.Agent != nil {
		requestClass = record.Agent.RequestClass
	}
	for _, marker := range []struct {
		name string
		set  bool
	}{
		{routerreplay.RunMarkerCompaction, requestClass == config.AgentRequestClassCompaction},
		{routerreplay.RunMarkerHousekeeping, requestClass == config.AgentRequestClassAuxiliary || record.RequestKind == store.RequestKindSessionTitle},
		{routerreplay.RunMarkerImage, record.ImageCount > 0},
		{routerreplay.RunMarkerEscalation, replayLooperEscalated(record)},
		{routerreplay.RunMarkerGuard, guard},
		{routerreplay.RunMarkerBlock, block},
		{routerreplay.RunMarkerFallback, replayFellBack(record)},
		{routerreplay.RunMarkerFailure, failed},
	} {
		if marker.set {
			markers = append(markers, marker.name)
		}
	}
	return markers
}

// replayLooperEscalated reports an escalation answered by a later model than
// the one it tried first.
func replayLooperEscalated(record routerreplay.RoutingRecord) bool {
	if record.RouteDiagnostics == nil || record.RouteDiagnostics.Looper == nil {
		return false
	}
	looper := record.RouteDiagnostics.Looper
	if looper.Algorithm != config.DecisionAlgorithmEscalate || len(looper.Attempts) == 0 {
		return false
	}
	for _, attempt := range looper.Attempts {
		if attempt.Selected {
			return attempt.Model != looper.Attempts[0].Model
		}
	}
	return false
}

// replayFellBack reports a call that tried another candidate after its
// first: the fallback audit records every request it runs for.
func replayFellBack(record routerreplay.RoutingRecord) bool {
	for _, outcome := range record.Outcomes {
		if attempts, _ := strconv.Atoi(outcome.Metadata["attempts"]); outcome.Source == fallbackOutcomeSource && attempts > 1 {
			return true
		}
	}
	return false
}

func replayLooperReason(looper *routerreplay.LooperDiagnostics) string {
	for _, attempt := range looper.Attempts {
		if attempt.Ordinal == looper.FinalAttemptOrdinal && attempt.Reason != "" {
			return looper.Algorithm + ": " + attempt.Reason
		}
	}
	return looper.Algorithm
}

// replayNewToolResults returns the tool calls whose results a request
// carried after the model's last turn, the results it sends now rather than
// the history every later request repeats, and those that failed.
func replayNewToolResults(trace *routerreplay.ToolTrace) ([]string, []string) {
	if trace == nil {
		return nil, nil
	}
	start := 0
	for index, step := range trace.Steps {
		if step.Source == replayToolSourceRequest &&
			(step.Type == replayToolStepAssistantToolCall || step.Type == replayToolStepAssistantFinalResponse) {
			start = index + 1
		}
	}
	var results, failures []string
	for _, step := range trace.Steps[start:] {
		if step.Source != replayToolSourceRequest || step.Type != replayToolStepClientToolResult {
			continue
		}
		id := step.ToolCallID
		if id == "" {
			id = step.ToolName
		}
		results = append(results, id)
		if step.Status == "failed" {
			failures = append(failures, id)
		}
	}
	return results, failures
}

// replayEndsWithFinalAnswer reports a response that answered without asking
// for a tool.
func replayEndsWithFinalAnswer(trace *routerreplay.ToolTrace) bool {
	if trace == nil {
		return false
	}
	answered := false
	for _, step := range trace.Steps {
		if step.Source == replayToolSourceRequest {
			continue
		}
		switch step.Type {
		case replayToolStepAssistantToolCall:
			return false
		case replayToolStepAssistantFinalResponse:
			answered = true
		}
	}
	return answered
}

// replayReportedOutcomes keeps the outcomes reported through the outcome
// API, whose verdicts are its own, and leaves out what the Router appends.
func replayReportedOutcomes(outcomes []store.Outcome) []store.Outcome {
	var reported []store.Outcome
	for _, outcome := range outcomes {
		switch routerruntime.RouterOutcomeVerdict(outcome.Verdict) {
		case routerruntime.RouterOutcomeVerdictGoodFit, routerruntime.RouterOutcomeVerdictUnderpowered,
			routerruntime.RouterOutcomeVerdictOverprovisioned, routerruntime.RouterOutcomeVerdictFailed:
			reported = append(reported, outcome)
		}
	}
	return reported
}

// replayMatchedSignals names each matched signal as type:name, sorted.
func replayMatchedSignals(signals routerreplay.Signal) []string {
	var matched []string
	for kind, names := range replaySignalGroups(signals) {
		for _, name := range names {
			matched = append(matched, kind+":"+name)
		}
	}
	sort.Strings(matched)
	return matched
}

// replaySignalFamily is one signal type and the signals of it a record matched.
type replaySignalFamily struct {
	name    string
	matched []string
}

// replaySignalFamilies is a record's matched signals by signal type, as an
// array so that a scan over the whole history allocates nothing for them.
func replaySignalFamilies(s routerreplay.Signal) [17]replaySignalFamily {
	return [...]replaySignalFamily{
		{"keyword", s.Keyword},
		{"domain", s.Domain},
		{"fact_check", s.FactCheck},
		{"user_feedback", s.UserFeedback},
		{"turn_relation", s.TurnRelation},
		{"language", s.Language},
		{"context", s.Context},
		{"complexity", s.Complexity},
		{"modality", s.Modality},
		{"authz", s.Authz},
		{"jailbreak", s.Jailbreak},
		{"safety", s.Safety},
		{"pii", s.PII},
		{"conversation", s.Conversation},
		{"metadata", s.Metadata},
		{"input_modality", s.InputModality},
		{"decision", s.Decision},
	}
}

// replaySignalGroups is a record's matched signals by signal type.
func replaySignalGroups(s routerreplay.Signal) map[string][]string {
	families := replaySignalFamilies(s)
	groups := make(map[string][]string, len(families))
	for _, family := range families {
		groups[family.name] = family.matched
	}
	return groups
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func nonEmptyStrings(values []string) []string {
	kept := values[:0]
	for _, value := range values {
		if value != "" {
			kept = append(kept, value)
		}
	}
	return kept
}
