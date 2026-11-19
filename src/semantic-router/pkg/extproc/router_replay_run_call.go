package extproc

import (
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	// The tool calls its response asked for, and those of the agent's
	// previous turn its request repeats with their results' status.
	turnTools, carriedTools []routerreplay.RunToolCall
	// turnIndex counts the user's prompts before the request, so a request
	// with the same count as the agent's previous one continues its turn.
	turnIndex int
	// offered names the tools the request offered the model.
	offered []string
}

func newReplayRunEntry(record routerreplay.RoutingRecord) replayRunEntry {
	entry := replayRunEntry{call: buildReplayRunCall(record), parentRequestID: record.ParentRequestID, turnIndex: record.TurnIndex}
	if record.Agent != nil {
		entry.agent = *record.Agent
	}
	entry.agent.ID = agentOf(record)
	entry.toolResults, entry.toolFailures = replayNewToolResults(record.ToolTrace)
	entry.turnTools, entry.carriedTools = replayTurnTools(record.ToolTrace), replayCarriedTools(record.ToolTrace)
	if record.ToolTrace != nil {
		entry.offered = record.ToolTrace.OfferedTools
	}
	return entry
}

// countReplayToolResults counts each tool result on the call that first
// carried it, in time order. A later call carries the results again in its
// history, which shows no step for a text answer or a new prompt, and a side
// request or Looper hop copies them: none of these sent a result.
func countReplayToolResults(entries []replayRunEntry, reports map[string]store.ToolEvent) {
	seen := make(map[string]bool)
	for index := range entries {
		entry := &entries[index]
		var results, failures []string
		if entry.parentRequestID == "" && entry.call.RequestKind == "" {
			results, failures = firstReplayToolResults(*entry, seen, reports)
		}
		entry.toolResults, entry.toolFailures = results, failures
		entry.call.ToolResults, entry.call.ToolFailures = len(results), len(failures)
		if len(results) > 0 {
			entry.call.Markers = append(entry.call.Markers, routerreplay.RunMarkerToolLoop)
		}
		if len(failures) > 0 {
			entry.call.Markers = append(entry.call.Markers, routerreplay.RunMarkerToolFailure)
		}
	}
}

// firstReplayToolResults are the results a call carried that no earlier call
// did, and those that failed: as the harness plugin reported the tool call,
// where one did, since a Responses result carries no error flag, else as the
// result says.
func firstReplayToolResults(entry replayRunEntry, seen map[string]bool, reports map[string]store.ToolEvent) (results, failures []string) {
	for _, id := range entry.toolResults {
		if seen[id] {
			continue
		}
		seen[id] = true
		results = append(results, id)
		failed := slices.Contains(entry.toolFailures, id)
		if report, ok := reports[id]; ok && report.Status != "" {
			failed = report.Status == store.ToolEventFailed
		}
		if failed {
			failures = append(failures, id)
		}
	}
	return results, failures
}

func buildReplayRunCall(record routerreplay.RoutingRecord) routerreplay.RunCall {
	call := routerreplay.RunCall{
		RecordID: record.ID, RequestID: record.RequestID, Timestamp: record.Timestamp,
		DurationMS: record.DurationMS, Recipe: record.Recipe, Entrypoint: record.OriginalModel,
		Decision: record.Decision, Signals: replayMatchedSignals(record.Signals), Model: replayAnsweringModel(record),
		Effort: firstNonEmptyString(record.ReasoningEffort, record.ReasoningMode), SelectionMethod: record.SelectionMethod, RequestKind: record.RequestKind,
		LifecycleState: record.LifecycleState, ResponseStatus: record.ResponseStatus, TerminalReason: record.TerminalReason,
		PromptTokens: record.PromptTokens, CompletionTokens: record.CompletionTokens,
		RouterCalls: replayRouterCalls(record), FinalAnswer: replayEndsWithFinalAnswer(record),
		Outcomes: replayReportedOutcomes(record.Outcomes), ImageCount: record.ImageCount,
	}
	if record.Agent != nil {
		call.RequestClass, call.AgentType = record.Agent.RequestClass, record.Agent.Type
	}
	if currency := replayRecordCurrency(record); currency != "" && finiteReplayCost(record.ActualCost) {
		call.Cost, call.Currency = record.ActualCost, currency
		if record.CostAccounting != nil {
			call.PricingSource = record.CostAccounting.PricingSource
		}
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
	refused := (record.JailbreakDetected || record.PIIDetected) &&
		(record.SelectionMethod == "fast_response" || record.SelectedModel == "")
	block := record.PIIBlocked || refused || guard && replayClientError(record) || replayRouterRefused(record)
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

// replayRouterRefused reports a call the Router refused itself: a guard, a
// fast_response block, a response guard, a rate limit or a spend budget,
// with a 4xx or, for a stream already under way, its error event.
func replayRouterRefused(record routerreplay.RoutingRecord) bool {
	switch record.TerminalReason {
	case replayTerminalRejected, budgetExceededCode, budgetUnpricedCode:
		return true
	}
	return false
}

func replayClientError(record routerreplay.RoutingRecord) bool {
	return record.ResponseStatus >= 400 && record.ResponseStatus < 500
}

// replayAnsweringModel is the model that answered a call: after a fallback
// the candidate its audit says succeeded, which an older record does not
// name as its selected model.
func replayAnsweringModel(record routerreplay.RoutingRecord) string {
	for _, outcome := range record.Outcomes {
		if outcome.Source == fallbackOutcomeSource && outcome.Reason == "succeeded" && outcome.TargetRef != "" {
			return outcome.TargetRef
		}
	}
	return record.SelectedModel
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
// for a tool: from its tool trace, or, where Replay kept none, from the stop
// reason the digest of its answer records.
func replayEndsWithFinalAnswer(record routerreplay.RoutingRecord) bool {
	trace := record.ToolTrace
	if trace == nil {
		return replayAnswerStopped(record.Outcomes)
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

// replayAnswerStopped reports an answer whose model ended the turn itself or
// reached its output limit, rather than asking for a tool.
func replayAnswerStopped(outcomes []store.Outcome) bool {
	for _, outcome := range outcomes {
		if outcome.Source != primaryResponseOutcomeSource {
			continue
		}
		switch llmprotocol.StopReason(outcome.Metadata[primaryResponseStopReason]) {
		case llmprotocol.StopEndTurn, llmprotocol.StopSequence, llmprotocol.StopMaxTokens:
			return true
		}
	}
	return false
}

// replayReportedOutcomes keeps the outcomes reported through the outcome
// API and leaves out what the Router appends itself. The API records its own
// verdicts under a reporter's source; the Router writes under router or the
// name of the feature, such as a fallback's failed outcome, whose verdict
// alone would read as reported.
func replayReportedOutcomes(outcomes []store.Outcome) []store.Outcome {
	var reported []store.Outcome
	for _, outcome := range outcomes {
		if replayReportedSource(outcome.Source) && replayReportedVerdict(outcome.Verdict) {
			reported = append(reported, outcome)
		}
	}
	return reported
}

func replayReportedSource(source string) bool {
	switch routerruntime.RouterOutcomeSource(source) {
	case routerruntime.RouterOutcomeSourceUser, routerruntime.RouterOutcomeSourceAgent, routerruntime.RouterOutcomeSourceEval,
		routerruntime.RouterOutcomeSourceOperator, routerruntime.RouterOutcomeSourceProvider:
		return true
	}
	return false
}

func replayReportedVerdict(verdict string) bool {
	switch routerruntime.RouterOutcomeVerdict(verdict) {
	case routerruntime.RouterOutcomeVerdictGoodFit, routerruntime.RouterOutcomeVerdictUnderpowered,
		routerruntime.RouterOutcomeVerdictOverprovisioned, routerruntime.RouterOutcomeVerdictFailed:
		return true
	}
	return false
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
func replaySignalFamilies(s routerreplay.Signal) [18]replaySignalFamily {
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
		{"agent", s.Agent},
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
