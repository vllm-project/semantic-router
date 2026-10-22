package extproc

import (
	"errors"
	"testing"
	"time"

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
	cfg.ModelConfig = map[string]config.ModelParams{
		"protection-cheap":    addTestQuality(config.ModelParams{}, 0.2),
		"protection-frontier": addTestQuality(config.ModelParams{}, 0.9),
	}
	cfg.RouterLearning.Protection.Tuning = config.RouterLearningProtectionTuning{
		MinTurnsBeforeSwitch: extprocIntPtr(0),
		SwitchMargin:         extprocFloat64Ptr(0.05),
		StabilityWeight:      extprocFloat64Ptr(0),
		ProgressGate:         protectionProgressGateTuning(scenario.ProgressGate),
	}
	router := &OpenAIRouter{Config: cfg}
	request := &llmprotocol.Request{}
	histories := map[string][]llmprotocol.Message{}
	rows := make([]protectionRow, 0, len(scenario.Steps))
	scenarioClock := time.Now().UTC().Add(-time.Minute)
	for turn, step := range scenario.Steps {
		request.Messages = histories[step.Conversation]
		for _, message := range step.Messages {
			request.Messages = append(request.Messages, protectionNeutralMessage(message))
		}
		histories[step.Conversation] = request.Messages
		input := protectionScenarioInput(router, scenario, step, turn, request)
		rows = append(rows, executeProtectionStep(t, router, input, scenario.ID, step, turn, scenarioClock.Add(time.Duration(turn)*time.Second)))
	}
	return rows
}

func protectionProgressGateTuning(gate *protectionProgressGate) *config.ProgressGateTuning {
	if gate == nil {
		return nil
	}
	enabled := true
	return &config.ProgressGateTuning{
		Enabled: &enabled, Mode: gate.Mode, CalibrationID: gate.CalibrationID,
		WindowSize: extprocIntPtr(gate.WindowSize), WindowTTLSeconds: extprocIntPtr(gate.WindowTTLSeconds),
		MinWindowOutcomes:         extprocIntPtr(gate.MinWindowOutcomes),
		MinConsecutiveRegressions: extprocIntPtr(gate.MinConsecutiveRegressions),
		MinConsecutiveRecoveries:  extprocIntPtr(gate.MinConsecutiveRecoveries),
		CooldownSeconds:           extprocFloat64Ptr(gate.CooldownSeconds),
		MaxSwitchesPerWindow:      extprocIntPtr(gate.MaxSwitchesPerWindow),
	}
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
	selCtx.AgenticSession = router.protectionSelectionContext(selCtx, ctx, identity).AgenticSession
	proposal := &selection.SelectionResult{SelectedModel: step.Proposal, Score: step.Scores[step.Proposal], AllScores: step.Scores, Method: selection.MethodStatic}
	ref := modelRefForName(candidates, step.Proposal)
	return routerLearningInput{selCtx: selCtx, baseResult: proposal, selectedModelRef: ref, ctx: ctx}
}

func executeProtectionStep(t *testing.T, router *OpenAIRouter, input routerLearningInput, scenarioID string, step protectionStep, turn int, outcomeAt time.Time) protectionRow {
	t.Helper()
	preflight := router.applyProtectionPreflight(input)
	decision, protectionErr := router.applyProtectionSwitch(input, preflight, routerLearningDecision{})
	if protectionErr != nil {
		if !errors.Is(protectionErr, selection.ErrNoEligibleCandidates) {
			t.Fatalf("%s/%s: %v", scenarioID, step.ID, protectionErr)
		}
		if decision.selectedModelRef != nil || decision.selectionResult != nil {
			t.Fatal("rejected protection returned a routable candidate")
		}
		response := router.respondSelectionRejected(input.ctx, "public", protectionErr)
		if response.GetImmediateResponse().GetStatus().GetCode() != 503 || response.GetRequestBody() != nil {
			t.Fatal("protection rejection did not terminate the request")
		}
		// A rejection has no selected-candidate policy to record. Report the
		// terminal outcome and do not publish a new session owner.
		session := input.selCtx.AgenticSession
		row := protectionRow{
			Scenario: scenarioID, Step: step.ID, Turn: turn, Category: step.Expected.Category,
			Previous: session.PreviousModel, Proposal: step.Proposal, Rejected: true,
			CandidateCount: len(input.selCtx.CandidateModels), SamplingAllowed: preflight.samplingAllowed,
			PreflightReason: preflight.policy.Reason, Action: "reject", Reason: "selection_rejected",
			HardLocked: session.ActiveToolLoop || session.HasNonPortableContext, CacheWarmth: session.CacheWarmth,
		}
		row.Failures = protectionFailures(row, step.Expected)
		return row
	}
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
	row.Gate = protectionGateReport(finalResult.SessionPolicy)
	row.Failures = protectionFailures(row, step.Expected)
	// The corpus treats each accepted step as a successful dispatch. Commit its
	// actual result; the next turn is never preloaded with an expected model.
	stageAgenticSessionDecision(finalCtx, finalResult, finalRef, input.ctx)
	if err := commitAgenticSessionDecision(input.ctx); err != nil {
		t.Fatal(err)
	}
	if step.Outcome != "" {
		sessiontelemetry.RecordTurnOutcome(routingLearningStateKey(input.ctx), sessiontelemetry.TurnOutcome{
			RequestID: scenarioID + "/" + step.ID,
			TurnIndex: turn,
			Model:     finalRef.Model,
			Category:  sessiontelemetry.TurnOutcomeCategory(step.Outcome),
			Source:    sessiontelemetry.TurnSourceOutcomeIngest,
		}, outcomeAt)
	}
	return row
}

func protectionGateReport(policy *selection.SessionPolicyTrace) *protectionGateRow {
	if policy == nil || policy.SwitchGate == nil {
		return nil
	}
	gate := policy.SwitchGate
	return &protectionGateRow{
		Decision: gate.Decision, Reason: gate.Reason, Origin: gate.Origin, Mode: gate.Mode,
		CalibrationID: gate.CalibrationID, EvidenceVersion: gate.EvidenceVersion,
		ApplicationReason: gate.ApplicationReason, Applied: gate.Applied, Enforced: gate.Enforced,
		ColdStart: gate.ColdStart, AttributableCount: gate.AttributableCount,
		MissingCount: gate.MissingCount, WindowCount: gate.WindowCount,
		RegressionStreak: gate.RegressionStreak, RecoveryStreak: gate.RecoveryStreak,
		SwitchesInWindow: gate.SwitchesInWindow,
	}
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
