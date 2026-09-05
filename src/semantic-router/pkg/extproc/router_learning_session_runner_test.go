package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Run the production fact extraction, preflight, switch guard and session write.
// Only the upstream algorithm's proposal is scripted; expectations never feed it.
func runProtectionScenario(t *testing.T, scenario protectionScenario) []protectionRow {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	defer sessiontelemetry.ResetRouterSessionMemoryForTesting()
	cfg := routerLearningProtectionOnlyTestConfig(scenario.Scope)
	cfg.DefaultModel = "protection-cheap"
	cfg.ModelConfig = map[string]config.ModelParams{"protection-cheap": {}, "protection-frontier": {}}
	cfg.RouterLearning.Protection.Tuning = config.RouterLearningProtectionTuning{
		MinTurnsBeforeSwitch: extprocIntPtr(0),
		SwitchMargin:         extprocFloat64Ptr(0.05),
		StabilityWeight:      extprocFloat64Ptr(0),
	}
	router := &OpenAIRouter{Config: cfg}
	request := &llmprotocol.Request{}
	histories := map[string][]llmprotocol.Message{}
	rows := make([]protectionRow, 0, len(scenario.Steps))
	for turn, step := range scenario.Steps {
		request.Messages = histories[step.Conversation]
		for _, message := range step.Messages {
			request.Messages = append(request.Messages, protectionNeutralMessage(message))
		}
		histories[step.Conversation] = request.Messages
		input := protectionScenarioInput(router, scenario, step, turn, request)
		rows = append(rows, executeProtectionStep(t, router, input, scenario.ID, step, turn))
	}
	return rows
}

func protectionScenarioInput(router *OpenAIRouter, scenario protectionScenario, step protectionStep, turn int, request *llmprotocol.Request) routerLearningInput {
	ctx := routerLearningRequestContext(scenario.ID, step.Conversation)
	if step.MissingIdentity {
		delete(ctx.Headers, "x-session-id")
	}
	ctx.TurnIndex = turn
	ctx.PreviousResponseID = step.PreviousResponseID
	ctx.CacheWarmthEstimate = step.CacheWarmth
	ctx.VSRSelectedDecision = &config.Decision{
		Name:        "protection-benchmark",
		Adaptations: config.DecisionAdaptationsConfig{Protection: &config.DecisionLearningProtectionConfig{Mode: scenario.Mode}},
	}
	history := extractSignalConversationHistory(request)
	ctx.VSRConversationFacts = router.prepareSignalEvaluationInput(history).conversationFacts
	candidates := make([]config.ModelRef, 0, len(step.Candidates))
	for _, model := range step.Candidates {
		candidates = append(candidates, config.ModelRef{Model: model})
	}
	selCtx := &selection.SelectionContext{SessionID: scenario.ID, DecisionName: "protection-benchmark", CandidateModels: candidates}
	identity, _ := router.protectionIdentity(ctx, router.Config.RouterLearning.Protection)
	// Preflight sees the same accumulated session state as the subsequent guard.
	selCtx.AgenticSession = router.buildAgenticSessionContext(ctx, candidates, identity.memoryKey, "")
	proposal := &selection.SelectionResult{SelectedModel: step.Proposal, Score: step.Scores[step.Proposal], AllScores: step.Scores, Method: selection.MethodStatic}
	ref := modelRefForName(candidates, step.Proposal)
	return routerLearningInput{selCtx: selCtx, baseResult: proposal, selectedModelRef: ref, ctx: ctx}
}

func executeProtectionStep(t *testing.T, router *OpenAIRouter, input routerLearningInput, scenarioID string, step protectionStep, turn int) protectionRow {
	t.Helper()
	preflight := router.applyProtectionPreflight(input)
	decision := router.applyProtectionSwitch(input, preflight, routerLearningDecision{})
	recordRouterLearningPolicies(input.ctx, preflight, routerLearningDecision{}, decision)
	finalCtx := firstNonNilSelectionContext(decision.selectionContext, input.selCtx)
	finalResult := firstNonNilSelectionResult(decision.selectionResult, input.baseResult)
	finalRef := firstNonNilModelRef(decision.selectedModelRef, input.selectedModelRef)
	if err := selection.ValidateSelectionResult(finalCtx, finalResult); err != nil {
		t.Fatal(err)
	}
	replay := decision.policy.toReplayProtection()
	if replay == nil {
		t.Fatal("missing production Replay protection diagnostics")
	}
	row := protectionRow{
		Scenario: scenarioID, Step: step.ID, Turn: turn,
		Previous: input.selCtx.AgenticSession.PreviousModel, Proposal: step.Proposal, Selected: finalRef.Model,
		SamplingAllowed: preflight.samplingAllowed, PreflightReason: preflight.policy.Reason,
		Action: replay.Action, Reason: replay.Reason, HardLocked: replay.HardLocked,
		CacheWarmth: input.selCtx.AgenticSession.CacheWarmth, Category: step.Expected.Category,
		CandidateCount: len(input.selCtx.CandidateModels),
	}
	row.Failures = protectionFailures(row, step.Expected)
	// This writes the actual result, so the next turn cannot be preloaded with
	// its expected model. No external store or model endpoint is involved.
	recordAgenticSessionDecision(finalCtx, finalResult, finalRef, input.ctx)
	return row
}

func protectionNeutralMessage(message protectionMessage) llmprotocol.Message {
	content := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: message.Text}
	if message.ToolCallID != "" {
		if message.Role == "assistant" {
			content = llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: message.ToolCallID, Name: "calculator", Arguments: message.Text}}
		} else {
			content = llmprotocol.Content{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: message.ToolCallID, Content: []llmprotocol.Content{content}}}
		}
	}
	return llmprotocol.Message{Role: llmprotocol.Role(message.Role), Content: []llmprotocol.Content{content}}
}
