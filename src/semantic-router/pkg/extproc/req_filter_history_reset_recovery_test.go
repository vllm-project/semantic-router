package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func recoverableResetDecision(t *testing.T, overrides map[string]interface{}) *config.Decision {
	t.Helper()
	configuration := map[string]interface{}{
		"enabled": true,
		"trigger": map[string]interface{}{
			"signal":         "topic_boundary",
			"min_confidence": 0.9,
		},
		"recovery": map[string]interface{}{
			"enabled":     true,
			"store":       "redis",
			"ttl_seconds": 900,
		},
	}
	for key, value := range overrides {
		configuration[key] = value
	}
	return historyResetDecision(t, configuration)
}

// recoverableResetRouter wires the shared recovery store and the trusted scope
// inputs a real deployment supplies, without requiring a live backend.
func recoverableResetRouter() *OpenAIRouter {
	return &OpenAIRouter{
		Config:              &config.RouterConfig{},
		CompressionRecovery: &recoveryStoreStub{},
	}
}

func recoverableResetContext(t *testing.T, decision *config.Decision, request *llmprotocol.Request) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		RequestID:               "request-1",
		VSRSelectedDecision:     decision,
		VSRSelectedDecisionName: "reset",
		Headers:                 map[string]string{"x-authz-user-id": "user-1"},
		SemanticRequest:         request,
	}
	bindHistoryResetPolicy(ctx)
	captureOriginalContextHistory(ctx)
	ctx.HistoryResetTrigger = &historyreset.TriggerResult{
		Class:      historyreset.TriggerChange,
		Confidence: 0.95,
		Signal:     "topic_boundary",
		Binding:    historyResetEvidenceBinding(ctx),
	}
	return ctx
}

func TestRecoverableResetIssuesOneRetrievableKey(t *testing.T) {
	router := recoverableResetRouter()
	request := resetConversation()
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 {
		t.Fatalf("expected the prior turn to be removed, got %d messages", len(request.Messages))
	}
	if len(ctx.ContextCompressionRecoveryKeys) != 1 {
		t.Fatalf("expected one issued key, got %v", ctx.ContextCompressionRecoveryKeys)
	}
	if ctx.HistoryResetDiagnostics.RecoveryStatus != historyreset.RecoveryStored {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}

	// The reserved tool must advertise exactly the issued key.
	var reserved *llmprotocol.Tool
	for index := range request.Tools {
		if request.Tools[index].Name == contextcompression.RetrieveToolName {
			reserved = &request.Tools[index]
		}
	}
	if reserved == nil {
		t.Fatal("the reserved retrieval tool was not installed")
	}
	if !strings.Contains(string(reserved.InputSchema), ctx.ContextCompressionRecoveryKeys[0]) {
		t.Fatalf("the reserved tool does not accept the issued key: %s", reserved.InputSchema)
	}

	// The stored payload must reconstruct the removed turn.
	store := router.CompressionRecovery.(*recoveryStoreStub)
	scope := router.contextCompressionScope(ctx)
	entry, err := store.Get(t.Context(), scope, ctx.ContextCompressionRecoveryKeys[0])
	if err != nil {
		t.Fatalf("the issued key is not retrievable: %v", err)
	}
	envelope, err := historyreset.DecodeEnvelope(entry.Content)
	if err != nil {
		t.Fatalf("stored payload is not a valid envelope: %v", err)
	}
	if envelope.Messages != 2 || envelope.Removed[0].Message.Content[0].Text != "old question" {
		t.Fatalf("unexpected stored envelope %+v", envelope)
	}
}

// Durable conversation state is untouched: recovery stores a detached copy and
// the captured original history still describes the request as it arrived.
func TestRecoverableResetLeavesTheOriginalHistoryIntact(t *testing.T) {
	router := recoverableResetRouter()
	request := resetConversation()
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	original := ctx.ContextRequestIR.OriginalHistory()
	if len(original.Messages) != 3 {
		t.Fatalf("the original history must be unchanged, got %d messages", len(original.Messages))
	}
}

// A streaming request cannot run the retrieval follow-up, so a policy that
// requires recovery preserves history and says so.
func TestRecoverableResetRefusesStreamingRequests(t *testing.T) {
	router := recoverableResetRouter()
	request := resetConversation()
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)
	ctx.ExpectStreamingResponse = true

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open must preserve the request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatal("a streaming request must keep its history")
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonStreamingUnsupported {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
	if len(ctx.ContextCompressionRecoveryKeys) != 0 {
		t.Fatal("no key may be issued for an unsupported request")
	}
}

// A client that already defines the reserved tool name is detected before any
// removal, because a collision discovered afterwards would strand content.
func TestRecoverableResetRefusesAReservedToolCollision(t *testing.T) {
	router := recoverableResetRouter()
	request := resetConversation()
	request.Tools = []llmprotocol.Tool{{Name: contextcompression.RetrieveToolName}}
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open must preserve the request: %v", err)
	}
	if len(request.Messages) != 3 || len(request.Tools) != 1 {
		t.Fatal("a reserved-tool collision must not change the request")
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonReservedToolConflict {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

// Without a store the policy cannot keep its recoverability promise.
func TestRecoverableResetRefusesRemovalWithoutAStore(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	request := resetConversation()
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open must preserve the request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatal("history must survive when recovery is unavailable")
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonRecoveryUnavailable {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

// An untrusted scope is as disqualifying as a missing store: without it the
// stored payload could not be bound to this caller.
func TestRecoverableResetRefusesRemovalWithoutATrustedScope(t *testing.T) {
	router := recoverableResetRouter()
	request := resetConversation()
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)
	ctx.Headers = map[string]string{}

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open must preserve the request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatal("history must survive without a trusted scope")
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonRecoveryUnavailable {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

func TestRecoverableResetFailClosedRejectsBeforeDispatch(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	request := resetConversation()
	decision := recoverableResetDecision(t, map[string]interface{}{"failure_mode": "fail_closed"})
	ctx := recoverableResetContext(t, decision, request)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err == nil {
		t.Fatal("fail-closed must reject when required recovery is unavailable")
	}
	if len(request.Messages) != 3 {
		t.Fatal("a rejected request must keep its pre-reset state")
	}
	if ctx.HistoryResetDiagnostics.Outcome != historyreset.OutcomeFailed {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

// Evidence that describes a different request cannot authorize removal.
func TestResetRejectsEvidenceBoundToAnotherRequest(t *testing.T) {
	router := recoverableResetRouter()
	request := resetConversation()
	ctx := recoverableResetContext(t, recoverableResetDecision(t, nil), request)
	ctx.HistoryResetTrigger.Binding = "some-other-request"

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open must preserve the request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatal("stale evidence must not remove history")
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonEvidenceStale {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

func TestHistoryResetEvidenceBindingFollowsTheResolvedHistory(t *testing.T) {
	first := &RequestContext{SemanticRequest: resetConversation()}
	captureOriginalContextHistory(first)

	second := &RequestContext{SemanticRequest: resetConversation()}
	captureOriginalContextHistory(second)
	if historyResetEvidenceBinding(first) != historyResetEvidenceBinding(second) {
		t.Fatal("the same conversation must produce the same binding")
	}

	other := resetConversation()
	other.Messages[0].Content[0].Text = "a different question"
	third := &RequestContext{SemanticRequest: other}
	captureOriginalContextHistory(third)
	if historyResetEvidenceBinding(first) == historyResetEvidenceBinding(third) {
		t.Fatal("a different conversation must produce a different binding")
	}
	if historyResetEvidenceBinding(&RequestContext{}) != "" {
		t.Fatal("an unresolved history must not produce a binding")
	}
}

// Compression and reset share one reserved tool and one key set, so a model
// sees a single retrieval surface however many actions removed content.
func TestContextActionsShareOneReservedRetrievalTool(t *testing.T) {
	ctx := &RequestContext{}
	request := resetConversation()

	if err := registerContextRecoveryKeys(ctx, request, "reset-key"); err != nil {
		t.Fatalf("reset key registration failed: %v", err)
	}
	if err := registerContextRecoveryKeys(ctx, request, "compression-key", "reset-key"); err != nil {
		t.Fatalf("compression key registration failed: %v", err)
	}

	reserved := 0
	var schema string
	for _, tool := range request.Tools {
		if tool.Name == contextcompression.RetrieveToolName {
			reserved++
			schema = string(tool.InputSchema)
		}
	}
	if reserved != 1 {
		t.Fatalf("expected exactly one reserved tool, got %d", reserved)
	}
	if len(ctx.ContextCompressionRecoveryKeys) != 2 {
		t.Fatalf("expected the union of issued keys, got %v", ctx.ContextCompressionRecoveryKeys)
	}
	for _, key := range []string{"reset-key", "compression-key"} {
		if !strings.Contains(schema, key) {
			t.Fatalf("the reserved tool does not accept %q: %s", key, schema)
		}
	}
}

func TestContextRecoveryKeyRegistrationIgnoresEmptyInput(t *testing.T) {
	ctx := &RequestContext{}
	request := resetConversation()
	if err := registerContextRecoveryKeys(ctx, request, ""); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(request.Tools) != 0 || len(ctx.ContextCompressionRecoveryKeys) != 0 {
		t.Fatal("an empty key must not install the reserved tool")
	}
}

// A fail-closed rejection must tell the client why in its status: an
// unavailable evidence or recovery dependency is a temporary condition, while
// anything else keeps the internal-failure mapping.
func TestFailClosedResetStatusMapping(t *testing.T) {
	cases := []struct {
		name   string
		reason string
		status int
	}{
		{"recovery_unavailable", historyreset.ReasonRecoveryUnavailable, 503},
		{"recovery_write_failed", historyreset.ReasonRecoveryWriteFailed, 503},
		{"streaming", historyreset.ReasonStreamingUnsupported, 503},
		{"reserved_tool", historyreset.ReasonReservedToolConflict, 503},
		{"evidence_missing", historyreset.ReasonEvidenceMissing, 503},
		{"evidence_stale", historyreset.ReasonEvidenceStale, 503},
		{"limit", historyreset.ReasonHistoryLimitExceeded, 500},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{HistoryResetDiagnostics: &historyreset.Diagnostics{
				Outcome: historyreset.OutcomeFailed,
				Reason:  test.reason,
			}}
			status, message := contextTransformationFailure(ctx)
			if status != test.status {
				t.Fatalf("status = %d, want %d", status, test.status)
			}
			if message == "" {
				t.Fatal("a rejection must carry a message")
			}
		})
	}

	// A compression-only failure keeps its existing response.
	status, message := contextTransformationFailure(&RequestContext{})
	if status != 500 || !strings.Contains(message, "Context compression") {
		t.Fatalf("unexpected compression mapping: %d %q", status, message)
	}
}
