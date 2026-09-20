package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextdedup"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func contextDedupDecision(t *testing.T, configuration map[string]interface{}) *config.Decision {
	t.Helper()
	return &config.Decision{Plugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginContextDedup,
		Configuration: config.MustStructuredPayload(configuration),
	}}}
}

// duplicatedConversation carries one prior turn twice ahead of the live turn.
func duplicatedConversation() *llmprotocol.Request {
	message := func(role llmprotocol.Role, body string) llmprotocol.Message {
		return llmprotocol.Message{
			Role:    role,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: body}},
		}
	}
	return &llmprotocol.Request{Messages: []llmprotocol.Message{
		message(llmprotocol.RoleUser, "old question"),
		message(llmprotocol.RoleAssistant, "old answer"),
		message(llmprotocol.RoleUser, "old question"),
		message(llmprotocol.RoleAssistant, "old answer"),
		message(llmprotocol.RoleUser, "live question"),
	}}
}

func enabledDedupContext(t *testing.T, request *llmprotocol.Request, configuration map[string]interface{}) *RequestContext {
	t.Helper()
	if configuration == nil {
		configuration = map[string]interface{}{"enabled": true}
	}
	ctx := &RequestContext{
		SemanticRequest:         request,
		VSRSelectedDecisionName: "dedup",
		VSRSelectedDecision:     contextDedupDecision(t, configuration),
	}
	bindContextDedupPolicy(ctx)
	return ctx
}

func TestContextDedupBindingIsInertWithoutAnEnabledPolicy(t *testing.T) {
	for name, decision := range map[string]*config.Decision{
		"absent":   {},
		"disabled": contextDedupDecision(t, map[string]interface{}{"enabled": false}),
	} {
		t.Run(name, func(t *testing.T) {
			request := duplicatedConversation()
			ctx := &RequestContext{SemanticRequest: request, VSRSelectedDecision: decision}
			bindContextDedupPolicy(ctx)
			if ctx.ContextDedupPolicy != nil {
				t.Fatal("no policy should be bound")
			}
			(&OpenAIRouter{}).prepareContextDedupStep(ctx, request)
			if len(ctx.ContextHistorySteps) != 0 || ctx.ContextDedupAction != nil {
				t.Fatal("no dedup step should be registered")
			}
			if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
				t.Fatalf("the shared stage must still run: %v", err)
			}
			if len(request.Messages) != 5 {
				t.Fatal("an inactive policy must not change the request")
			}
			if ctx.ContextDedupDiagnostics != nil {
				t.Fatal("an inactive policy must not publish diagnostics")
			}
		})
	}
}

func TestContextDedupEnabledPolicyKeepsTheResponseCache(t *testing.T) {
	ctx := enabledDedupContext(t, duplicatedConversation(), nil)
	if ctx.ContextDedupPolicy == nil {
		t.Fatal("expected the enabled policy to bind")
	}
	if ctx.CacheReadBypass || ctx.CacheWriteBypass {
		t.Fatal("a deterministic policy must not bypass the response cache")
	}
}

func TestContextDedupRegistersOneStepBeforeTheSharedStage(t *testing.T) {
	request := duplicatedConversation()
	ctx := enabledDedupContext(t, request, nil)
	(&OpenAIRouter{}).prepareContextDedupStep(ctx, request)
	(&OpenAIRouter{}).prepareContextDedupStep(ctx, request)
	if len(ctx.ContextHistorySteps) != 1 || ctx.ContextHistorySteps[0].Kind != contextcompression.TransformDeduplicate {
		t.Fatalf("expected exactly one dedup registration, got %+v", ctx.ContextHistorySteps)
	}
	if ctx.ContextRequestIR != nil {
		t.Fatal("registration must not construct the request IR")
	}
}

func TestContextDedupRemovesTheLaterCopyThroughTheSharedStage(t *testing.T) {
	request := duplicatedConversation()
	ctx := enabledDedupContext(t, request, nil)
	captureOriginalContextHistory(ctx)
	(&OpenAIRouter{}).prepareContextDedupStep(ctx, request)
	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open evaluation must not reject the request: %v", err)
	}
	if len(request.Messages) != 3 || request.Messages[2].Content[0].Text != "live question" {
		t.Fatalf("expected the later copy removed, got %+v", request.Messages)
	}
	if len(ctx.ContextRequestIR.OriginalHistory().Messages) != 5 {
		t.Fatal("original history must keep the request as received")
	}
	diagnostics := ctx.ContextDedupDiagnostics
	if diagnostics == nil || diagnostics.Outcome != contextdedup.OutcomeApplied || diagnostics.RemovedMessages != 2 || diagnostics.RemovedTurns != 1 {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	receipts := ctx.ContextRequestIR.Transformations.Receipts()
	if len(receipts) != 2 || receipts[0].Kind != contextcompression.TransformDeduplicate || receipts[1].Kind != contextcompression.TransformCompress {
		t.Fatalf("pipeline order lost: %+v", receipts)
	}
	record := contextDedupReplayDiagnostics(ctx)
	if record == nil || record.RemovedMessages != 2 || record.Retained["ineligible"] != 1 || len(record.Segments) != 1 {
		t.Fatalf("unexpected replay diagnostics %+v", record)
	}
	if record.Segments[0].RetainedFirstMessageID != 0 || record.Segments[0].RemovedFirstMessageID != 2 {
		t.Fatalf("unexpected segment %+v", record.Segments[0])
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{"old question", "old answer", "live question"} {
		if strings.Contains(string(encoded), leak) {
			t.Fatalf("replay diagnostics leaked content %q", leak)
		}
	}
	if contextDedupReplayDiagnostics(&RequestContext{}) != nil {
		t.Fatal("a request without an evaluation must not publish replay diagnostics")
	}
}

func TestContextDedupStepOrderHoldsWhateverRegistrationOrder(t *testing.T) {
	request := &llmprotocol.Request{}
	for _, text := range []string{"stale", "stale answer", "old", "old answer", "old", "old answer", "live"} {
		role := llmprotocol.RoleUser
		if strings.HasSuffix(text, "answer") {
			role = llmprotocol.RoleAssistant
		}
		request.Messages = append(request.Messages, llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}})
	}
	ctx := enabledDedupContext(t, request, nil)
	captureOriginalContextHistory(ctx)
	(&OpenAIRouter{}).prepareContextDedupStep(ctx, request)
	// A reset policy registering after dedup must still run first.
	ctx.ContextHistorySteps = appendContextHistoryStep(ctx.ContextHistorySteps, contextcompression.TransformationStep{
		Kind: contextcompression.TransformReset, Enabled: true, FailureMode: contextcompression.FailureClosed,
		Propose: func(context.Context, contextcompression.TransformationView) (contextcompression.TransformationEdits, error) {
			return contextcompression.TransformationEdits{RemoveMessages: []int{0, 1}}, nil
		},
	})
	if err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 || request.Messages[0].Content[0].Text != "old" || request.Messages[2].Content[0].Text != "live" {
		t.Fatalf("expected reset then dedup, got %+v", request.Messages)
	}
	receipts := ctx.ContextRequestIR.Transformations.Receipts()
	if len(receipts) != 3 || receipts[0].Kind != contextcompression.TransformReset ||
		receipts[1].Kind != contextcompression.TransformDeduplicate || receipts[2].Kind != contextcompression.TransformCompress {
		t.Fatalf("pipeline order lost: %+v", receipts)
	}
	if ctx.ContextDedupDiagnostics.RemovedMessages != 2 || ctx.ContextDedupDiagnostics.ExaminedMessages != 5 {
		t.Fatalf("unexpected diagnostics %+v", ctx.ContextDedupDiagnostics)
	}
}

func TestContextDedupFailureModes(t *testing.T) {
	for _, tc := range []struct {
		name          string
		configuration map[string]interface{}
		wantErr       bool
		wantStatus    int
	}{
		{"limit_fail_open", map[string]interface{}{"enabled": true, "limits": map[string]interface{}{"max_history_turns": 2, "max_segment_turns": 2}}, false, 0},
		{"limit_fail_closed", map[string]interface{}{"enabled": true, "failure_mode": "fail_closed", "limits": map[string]interface{}{"max_history_turns": 2, "max_segment_turns": 2}}, true, 503},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := duplicatedConversation()
			ctx := enabledDedupContext(t, request, tc.configuration)
			captureOriginalContextHistory(ctx)
			(&OpenAIRouter{}).prepareContextDedupStep(ctx, request)
			err := (&OpenAIRouter{}).applyContextTransformationPlan(ctx, request)
			if (err != nil) != tc.wantErr {
				t.Fatalf("error mismatch: %v", err)
			}
			if len(request.Messages) != 5 {
				t.Fatal("a failed evaluation must leave the request unchanged")
			}
			if ctx.ContextDedupDiagnostics == nil || ctx.ContextDedupDiagnostics.Outcome != contextdedup.OutcomeFailed ||
				ctx.ContextDedupDiagnostics.Reason != contextdedup.ReasonHistoryLimitExceeded {
				t.Fatalf("unexpected diagnostics %+v", ctx.ContextDedupDiagnostics)
			}
			if tc.wantErr {
				status, message := contextTransformationFailure(ctx)
				if status != tc.wantStatus || !strings.Contains(message, "deduplication") {
					t.Fatalf("unexpected rejection %d %q", status, message)
				}
			}
		})
	}
	status, message := contextTransformationFailure(&RequestContext{})
	if status != 500 || !strings.Contains(message, "compression") {
		t.Fatalf("a compression-only failure must keep its mapping: %d %q", status, message)
	}
}

func TestContextDedupWithoutNeutralRequestIsBlocked(t *testing.T) {
	ctx := &RequestContext{
		VSRSelectedDecisionName: "dedup",
		VSRSelectedDecision:     contextDedupDecision(t, map[string]interface{}{"enabled": true}),
	}
	bindContextDedupPolicy(ctx)
	(&OpenAIRouter{}).prepareContextDedupStep(ctx, nil)
	if ctx.ContextDedupAction == nil || len(ctx.ContextHistorySteps) != 1 {
		t.Fatal("a blocked action must still register so the receipt explains the outcome")
	}
	ir := contextcompression.ParseSemanticRequest(nil, contextcompression.Provenance{})
	if err := ir.ApplySteps(context.Background(), ctx.ContextHistorySteps); err != nil {
		t.Fatalf("fail-open must not reject: %v", err)
	}
	finalizeContextDedupDiagnostics(ctx, ir)
	if ctx.ContextDedupDiagnostics == nil || ctx.ContextDedupDiagnostics.Reason != contextdedup.ReasonUnsupportedRepresentation {
		t.Fatalf("unexpected diagnostics %+v", ctx.ContextDedupDiagnostics)
	}
}

func TestContextDedupPolicyMirrorsConfiguration(t *testing.T) {
	policy := contextDedupPolicy(&config.ContextDedupPluginConfig{
		Normalization: "whitespace",
		FailureMode:   "fail_closed",
		Limits:        &config.ContextDedupLimitsConfig{MaxHistoryTurns: 16, MaxSegmentTurns: 4, TimeoutMs: 10},
	})
	if policy.Normalization != contextdedup.NormalizationWhitespace || !policy.FailClosed {
		t.Fatalf("unexpected policy %+v", policy)
	}
	if policy.MaxHistoryTurns != 16 || policy.MaxSegmentTurns != 4 || policy.MaxHistoryBytes != config.DefaultContextDedupMaxHistoryBytes {
		t.Fatalf("unexpected limits %+v", policy)
	}
	if policy.Timeout.Milliseconds() != 10 {
		t.Fatalf("unexpected timeout %v", policy.Timeout)
	}
}
