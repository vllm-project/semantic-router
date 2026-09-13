package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// storedResponsesContext builds a Responses request whose earlier turn lives
// behind previous_response_id rather than in the request body.
func storedResponsesContext(t *testing.T, request *llmprotocol.Request) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		RequestID:           "request-1",
		SourceFormat:        llmprotocol.OpenAIResponsesV1,
		TargetFormat:        llmprotocol.OpenAIResponsesV1,
		SemanticRequest:     request,
		VSRSelectedDecision: historyResetDecision(t, enabledResetConfiguration()),
		ResponseObjectState: &ResponseObjectState{
			PreviousResponseID: "resp_previous",
			ConversationHistory: []*responseapi.StoredResponse{{
				Input: []responseapi.InputItem{{
					Type:    responseapi.ItemTypeMessage,
					Role:    responseapi.RoleUser,
					Content: json.RawMessage(`"stored question"`),
				}},
				OutputText: "stored answer",
			}},
		},
	}
	request.PreviousResponseID = "resp_previous"
	return ctx
}

func enabledResetConfiguration() map[string]interface{} {
	return map[string]interface{}{
		"enabled": true,
		"trigger": map[string]interface{}{
			"signal":         "topic_boundary",
			"min_confidence": 0.9,
		},
	}
}

// Stored Responses history must be resolved before the context stage. If it
// were left to dispatch, reset would remove nothing and the provider would
// still receive the turns the policy was meant to drop.
func TestResponsesStoredHistoryIsResolvedBeforeReset(t *testing.T) {
	router := &OpenAIRouter{}
	request := &llmprotocol.Request{
		Model:    "model",
		Messages: []llmprotocol.Message{neutralTextMessage(llmprotocol.RoleUser, "live question")},
	}
	ctx := storedResponsesContext(t, request)

	bindHistoryResetPolicy(ctx)
	router.resolveHistoryResetRequestHistory(ctx)
	if len(request.Messages) != 3 {
		t.Fatalf("stored history was not materialized before reset, got %d messages", len(request.Messages))
	}
	if !ctx.ResponseObjectState.ProviderContextApplied {
		t.Fatal("materialization must be marked so dispatch does not repeat it")
	}
	// The snapshot is retaken so evidence and the reset view describe the
	// history the provider would actually receive.
	if len(ctx.OriginalContextHistory.Conversation().Messages) != 3 {
		t.Fatal("the original-history snapshot still predates the stored turns")
	}

	ctx.HistoryResetTrigger = &historyreset.TriggerResult{
		Class:      historyreset.TriggerChange,
		Confidence: 0.95,
		Signal:     "topic_boundary",
		Binding:    historyResetEvidenceBinding(ctx),
	}
	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 || request.Messages[0].Content[0].Text != "live question" {
		t.Fatalf("stored history survived the reset: %+v", request.Messages)
	}

	// Dispatch must not restore what the reset removed.
	changed, err := router.materializeResponseObjectContext(request, ctx)
	if err != nil {
		t.Fatalf("dispatch materialization failed: %v", err)
	}
	if changed || len(request.Messages) != 1 {
		t.Fatalf("dispatch prepended the stored history again: %+v", request.Messages)
	}
	// The owning API keeps its lineage; reset only changed the context view.
	if ctx.ResponseObjectState.PreviousResponseID != "resp_previous" {
		t.Fatal("reset disturbed the Responses lineage")
	}
	if len(ctx.ResponseObjectState.ConversationHistory) != 1 {
		t.Fatal("reset mutated the stored conversation history")
	}
}

// A decision without an enabled reset policy keeps the existing ordering: the
// stored history is materialized at dispatch, exactly as before.
func TestResponsesStoredHistoryStaysAtDispatchWithoutReset(t *testing.T) {
	router := &OpenAIRouter{}
	request := &llmprotocol.Request{
		Model:    "model",
		Messages: []llmprotocol.Message{neutralTextMessage(llmprotocol.RoleUser, "live question")},
	}
	ctx := storedResponsesContext(t, request)
	ctx.VSRSelectedDecision = historyResetDecision(t, map[string]interface{}{"enabled": false})

	bindHistoryResetPolicy(ctx)
	router.resolveHistoryResetRequestHistory(ctx)
	if len(request.Messages) != 1 || ctx.ResponseObjectState.ProviderContextApplied {
		t.Fatal("an inactive policy must not change the materialization point")
	}
}

// An internal Looper hop continues a public turn that was already evaluated,
// so it inherits that completion instead of starting a new topic-change event.
func TestLooperHopInheritsTheCompletedResetEvent(t *testing.T) {
	router := &OpenAIRouter{}
	request := resetConversation()
	ctx := &RequestContext{
		RequestID:           "request-1",
		SemanticRequest:     request,
		VSRSelectedDecision: historyResetDecision(t, enabledResetConfiguration()),
	}
	bindHistoryResetPolicy(ctx)
	captureOriginalContextHistory(ctx)
	prepareLooperContextHistorySteps(ctx)

	if len(ctx.ContextHistorySteps) != 1 {
		t.Fatalf("expected one registered step, got %d", len(ctx.ContextHistorySteps))
	}
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("an internal hop must not remove history, got %d messages", len(request.Messages))
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonInheritedCompleted {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
	if ctx.HistoryResetDiagnostics.RemovedMessages != 0 {
		t.Fatal("an inherited event cannot report removals")
	}
}

// Even a non-removing enabled history step opts the request into the shared
// live-history compression protection, which is why the hop registers one.
func TestLooperHopKeepsTheEnabledHistoryStepRegistered(t *testing.T) {
	ctx := &RequestContext{
		SemanticRequest:     resetConversation(),
		VSRSelectedDecision: historyResetDecision(t, enabledResetConfiguration()),
	}
	bindHistoryResetPolicy(ctx)
	prepareLooperContextHistorySteps(ctx)
	prepareLooperContextHistorySteps(ctx)
	if len(ctx.ContextHistorySteps) != 1 || !ctx.ContextHistorySteps[0].Enabled {
		t.Fatalf("expected one enabled step, got %+v", ctx.ContextHistorySteps)
	}

	disabled := &RequestContext{
		SemanticRequest:     resetConversation(),
		VSRSelectedDecision: historyResetDecision(t, map[string]interface{}{"enabled": false}),
	}
	bindHistoryResetPolicy(disabled)
	prepareLooperContextHistorySteps(disabled)
	if len(disabled.ContextHistorySteps) != 0 {
		t.Fatal("an inactive policy must not register a step on the internal path")
	}
}

// The existing demand accounting must observe the request reset produced, not
// a stale estimate from the pre-reset generation.
func TestRequestDemandObservesTheResetGeneration(t *testing.T) {
	router := &OpenAIRouter{}
	request := resetConversation()
	ctx := enabledResetContext(t, request)
	captureOriginalContextHistory(ctx)
	ctx.HistoryResetTrigger = &historyreset.TriggerResult{
		Class:      historyreset.TriggerChange,
		Confidence: 0.95,
		Signal:     "topic_boundary",
		Binding:    historyResetEvidenceBinding(ctx),
	}
	captureRequestDemand(ctx, requestDemandStageOriginal, request, "model")
	before := request.Generation

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if request.Generation == before {
		t.Fatal("a committed removal must advance the request generation")
	}
	captureRequestDemand(ctx, requestDemandStagePostContext, request, "model")

	original, ok := capturedRequestDemand(ctx, requestDemandStageOriginal)
	if !ok {
		t.Fatalf("the original demand snapshot is missing: %+v", ctx.RequestDemandSnapshots)
	}
	postContext, ok := capturedRequestDemand(ctx, requestDemandStagePostContext)
	if !ok {
		t.Fatalf("the post-context demand snapshot is missing: %+v", ctx.RequestDemandSnapshots)
	}
	if postContext.RequestGeneration <= original.RequestGeneration {
		t.Fatalf(
			"the post-context snapshot must observe the new generation: original=%d post=%d",
			original.RequestGeneration,
			postContext.RequestGeneration,
		)
	}
	if postContext.PromptTokens >= original.PromptTokens {
		t.Fatalf(
			"the post-context estimate must observe the smaller request: original=%d post=%d",
			original.PromptTokens,
			postContext.PromptTokens,
		)
	}
}

func capturedRequestDemand(ctx *RequestContext, stage string) (store.RequestDemandSnapshot, bool) {
	for _, snapshot := range ctx.RequestDemandSnapshots {
		if snapshot.Stage == stage {
			return snapshot, true
		}
	}
	return store.RequestDemandSnapshot{}, false
}
