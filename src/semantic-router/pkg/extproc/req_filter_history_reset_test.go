package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func historyResetDecision(t *testing.T, configuration map[string]interface{}) *config.Decision {
	t.Helper()
	return &config.Decision{Plugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginHistoryReset,
		Configuration: config.MustStructuredPayload(configuration),
	}}}
}

func resetConversation() *llmprotocol.Request {
	message := func(role llmprotocol.Role, body string) llmprotocol.Message {
		return llmprotocol.Message{
			Role:    role,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: body}},
		}
	}
	return &llmprotocol.Request{Messages: []llmprotocol.Message{
		message(llmprotocol.RoleUser, "old question"),
		message(llmprotocol.RoleAssistant, "old answer"),
		message(llmprotocol.RoleUser, "live question"),
	}}
}

func enabledResetContext(t *testing.T, request *llmprotocol.Request) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		SemanticRequest: request,
		VSRSelectedDecision: historyResetDecision(t, map[string]interface{}{
			"enabled": true,
			"trigger": map[string]interface{}{
				"signal":         "topic_boundary",
				"min_confidence": 0.9,
			},
		}),
	}
	bindHistoryResetPolicy(ctx)
	return ctx
}

func TestHistoryResetBindingIsInertWithoutAnEnabledPolicy(t *testing.T) {
	for name, decision := range map[string]*config.Decision{
		"absent":   {},
		"disabled": historyResetDecision(t, map[string]interface{}{"enabled": false}),
	} {
		t.Run(name, func(t *testing.T) {
			request := resetConversation()
			ctx := &RequestContext{SemanticRequest: request, VSRSelectedDecision: decision}
			bindHistoryResetPolicy(ctx)
			if ctx.HistoryResetPolicy != nil {
				t.Fatal("no policy should be bound")
			}
			if ctx.CacheReadBypass || ctx.CacheWriteBypass {
				t.Fatal("an inactive policy must not disturb caching")
			}

			prepareContextHistorySteps(ctx, request)
			if len(ctx.ContextHistorySteps) != 0 || ctx.HistoryResetAction != nil {
				t.Fatal("no reset step should be registered")
			}
			if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
				t.Fatalf("the shared stage must still run: %v", err)
			}
			if len(request.Messages) != 3 {
				t.Fatal("an inactive policy must not change the request")
			}
			if ctx.HistoryResetDiagnostics != nil {
				t.Fatal("an inactive policy must not publish diagnostics")
			}
		})
	}
}

func TestHistoryResetEnabledPolicyBypassesTheResponseCache(t *testing.T) {
	ctx := enabledResetContext(t, resetConversation())
	if ctx.HistoryResetPolicy == nil {
		t.Fatal("expected the enabled policy to bind")
	}
	if !ctx.CacheReadBypass || !ctx.CacheWriteBypass {
		t.Fatal("an enabled policy must bypass cache reads and writes")
	}
}

func TestHistoryResetRegistersOneStepBeforeTheSharedStage(t *testing.T) {
	request := resetConversation()
	ctx := enabledResetContext(t, request)

	prepareContextHistorySteps(ctx, request)
	prepareContextHistorySteps(ctx, request)
	if len(ctx.ContextHistorySteps) != 1 {
		t.Fatalf("expected exactly one registration, got %d", len(ctx.ContextHistorySteps))
	}
	if ctx.ContextRequestIR != nil {
		t.Fatal("registration must not construct the request IR")
	}
}

// Without a topic-continuity result the action must preserve history and say
// why, instead of assuming either continuation or change.
func TestHistoryResetPreservesHistoryWhenNoTriggerResultExists(t *testing.T) {
	request := resetConversation()
	ctx := enabledResetContext(t, request)
	captureOriginalContextHistory(ctx)
	prepareContextHistorySteps(ctx, request)

	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open evaluation must not reject the request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("history must be preserved, got %d messages", len(request.Messages))
	}
	if ctx.HistoryResetDiagnostics == nil {
		t.Fatal("a configured evaluation must publish a diagnostic")
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonEvidenceMissing {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
	if ctx.HistoryResetDiagnostics.RemovedMessages != 0 {
		t.Fatalf("a skipped evaluation cannot report removals, got %+v", ctx.HistoryResetDiagnostics)
	}
}

func TestHistoryResetRemovesEligibleTurnsOnAcceptedChange(t *testing.T) {
	request := resetConversation()
	ctx := enabledResetContext(t, request)
	ctx.HistoryResetTrigger = &historyreset.TriggerResult{
		Class:      historyreset.TriggerChange,
		Confidence: 0.95,
		Signal:     "topic_boundary",
	}
	captureOriginalContextHistory(ctx)
	prepareContextHistorySteps(ctx, request)

	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 || request.Messages[0].Content[0].Text != "live question" {
		t.Fatalf("expected only the live turn to survive, got %+v", request.Messages)
	}
	if ctx.HistoryResetDiagnostics.Outcome != historyreset.OutcomeApplied ||
		ctx.HistoryResetDiagnostics.RemovedMessages != 2 {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
	if len(ctx.ContextRequestIR.OriginalHistory().Messages) != 3 {
		t.Fatal("the captured original history must be unchanged by removal")
	}
}

// A policy that requires recovery cannot remove history while recoverable
// storage is unavailable, and it must say so rather than silently proceeding.
func TestHistoryResetRefusesUnrecoverableRemoval(t *testing.T) {
	request := resetConversation()
	ctx := &RequestContext{
		SemanticRequest: request,
		VSRSelectedDecision: historyResetDecision(t, map[string]interface{}{
			"enabled": true,
			"trigger": map[string]interface{}{
				"signal":         "topic_boundary",
				"min_confidence": 0.9,
			},
			"recovery": map[string]interface{}{"enabled": true, "store": "redis"},
		}),
	}
	bindHistoryResetPolicy(ctx)
	ctx.HistoryResetTrigger = &historyreset.TriggerResult{
		Class:      historyreset.TriggerChange,
		Confidence: 1,
		Signal:     "topic_boundary",
	}
	captureOriginalContextHistory(ctx)
	prepareContextHistorySteps(ctx, request)

	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open must preserve the request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatal("history must survive when required recovery is unavailable")
	}
	if ctx.HistoryResetDiagnostics.Outcome != historyreset.OutcomeFailed ||
		ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonRecoveryUnavailable {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

// fail_closed stops the plan before dispatch and leaves the request untouched.
func TestHistoryResetFailClosedRejectsBeforeDispatch(t *testing.T) {
	request := resetConversation()
	ctx := &RequestContext{
		SemanticRequest: request,
		VSRSelectedDecision: historyResetDecision(t, map[string]interface{}{
			"enabled":      true,
			"failure_mode": "fail_closed",
			"trigger": map[string]interface{}{
				"signal":         "topic_boundary",
				"min_confidence": 0.9,
			},
			"recovery": map[string]interface{}{"enabled": true, "store": "redis"},
		}),
	}
	bindHistoryResetPolicy(ctx)
	captureOriginalContextHistory(ctx)
	prepareContextHistorySteps(ctx, request)

	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err == nil {
		t.Fatal("expected the plan to be rejected")
	}
	if len(request.Messages) != 3 {
		t.Fatal("a rejected request must keep its pre-reset state")
	}
	if ctx.HistoryResetDiagnostics == nil ||
		ctx.HistoryResetDiagnostics.Outcome != historyreset.OutcomeFailed {
		t.Fatalf("a rejected plan must still publish a diagnostic, got %+v", ctx.HistoryResetDiagnostics)
	}
}

// Replay must receive counts and bounded reasons only. Conversation text,
// tool arguments, and recovery keys never reach the record.
func TestHistoryResetReplayDiagnosticsCarryNoContent(t *testing.T) {
	request := resetConversation()
	ctx := enabledResetContext(t, request)
	ctx.HistoryResetTrigger = &historyreset.TriggerResult{
		Class:      historyreset.TriggerChange,
		Confidence: 0.95,
		Signal:     "topic_boundary",
		Version:    "v1",
	}
	captureOriginalContextHistory(ctx)
	prepareContextHistorySteps(ctx, request)
	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}

	record := historyResetReplayDiagnostics(ctx)
	if record == nil {
		t.Fatal("expected a replay record for a configured evaluation")
	}
	if record.Outcome != string(historyreset.OutcomeApplied) ||
		record.Reason != historyreset.ReasonApplied {
		t.Fatalf("unexpected replay record %+v", record)
	}
	if record.RemovedMessages != 2 || record.RemovedTurns != 1 || record.ExaminedMessages != 3 {
		t.Fatalf("unexpected replay counts %+v", record)
	}
	if record.Signal != "topic_boundary" || record.TriggerClass != "change" || record.Version != "v1" {
		t.Fatalf("unexpected trigger identity %+v", record)
	}

	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatalf("failed to encode the replay record: %v", err)
	}
	for _, forbidden := range []string{"old question", "old answer", "live question"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("replay record leaked conversation content: %s", encoded)
		}
	}
}

func TestHistoryResetReplayDiagnosticsAbsentWithoutAnEvaluation(t *testing.T) {
	if historyResetReplayDiagnostics(&RequestContext{}) != nil {
		t.Fatal("an unevaluated request must not publish a replay record")
	}
}
