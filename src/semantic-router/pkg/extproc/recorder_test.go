package extproc

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func replayRoutingRecordMetadataTestContext() *RequestContext {
	return &RequestContext{
		RequestID:                     "req-1",
		SessionID:                     "sess-replay-test",
		TurnIndex:                     2,
		VSRSelectedCategory:           "math",
		VSRReasoningMode:              "on",
		VSRSelectedDecisionConfidence: 0.91,
		VSRSelectionMethod:            "router_dc",
		VSRCacheHit:                   true,
		ExpectStreamingResponse:       true,
		VSRLearningPolicies: testLearningPolicies(
			replayTestProtectionPolicyWithTrace(&selection.SessionPolicyTrace{
				Phase:             "user_turn",
				CurrentModel:      "model-a",
				BaseSelectedModel: "model-c",
				SelectedModel:     "model-b",
				DecisionReason:    "switch_has_best_adjusted_score",
			}),
		),
		VSRSelectedDecision: &config.Decision{
			Name:     "balance",
			Tier:     3,
			Priority: 120,
		},
		VSRMatchedKeywords:     []string{"math_keyword"},
		VSRMatchedModality:     []string{"AR"},
		VSRMatchedAuthz:        []string{"premium_tier"},
		VSRMatchedJailbreak:    []string{"jailbreak_detector"},
		VSRMatchedPII:          []string{"email_block"},
		VSRMatchedKB:           []string{"privacy_kb"},
		VSRMatchedConversation: []string{"multi_turn_user"},
		VSRMatchedEvent:        []string{"critical_payment_event"},
		VSRMatchedProjection:   []string{"balance_reasoning"},
		VSRProjectionScores: map[string]float64{
			"reasoning_pressure": 0.73,
		},
		VSRSignalConfidences: map[string]float64{
			"projection:balance_reasoning": 0.73,
		},
		VSRSignalValues: map[string]float64{
			"reask:persistently_dissatisfied": 2,
		},
	}
}

func TestBuildReplayRoutingRecordCapturesSessionAndDecisionMetadata(t *testing.T) {
	ctx := replayRoutingRecordMetadataTestContext()
	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")

	assertReplayRoutingMetadata(t, record)
	assertReplayRouteDiagnostics(t, record)
	assertReplayMatchedSignals(t, record)
}

func TestBuildReplayRoutingRecordCapturesLearningAdaptations(t *testing.T) {
	ctx := replayRoutingRecordMetadataTestContext()
	ctx.VSRLearningProtectionPreflight = replayTestLearningPolicy(
		routerLearningMethodProtection,
		"apply",
		routerLearningActionAllowSampling,
		"no_tool_or_protocol_state",
		"conversation",
	).toReplayProtection()
	ctx.VSRLearningPolicies = testLearningPolicies(
		replayTestLearningPolicy(
			routerLearningMethodProtection,
			"apply",
			routerLearningActionHoldCurrent,
			"cache_cost_high",
			"conversation",
		),
		replayTestLearningPolicy(
			routerLearningMethodAdaptation,
			"observe",
			routerLearningActionObserve,
			"observe_only",
			"",
		),
	)

	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")

	if record.Learning == nil {
		t.Fatal("expected learning diagnostics")
	}
	if got := record.Learning.ProtectionPreflight.Action; got != "allow_sampling" {
		t.Fatalf("expected protection preflight action recorded, got %#v", record.Learning.ProtectionPreflight)
	}
	if got := record.Learning.Protection.Action; got != "hold_current" {
		t.Fatalf("expected protection action recorded, got %#v", record.Learning.Protection)
	}
	if got := record.Learning.Adaptation.Mode; got != "observe" {
		t.Fatalf("expected adaptation mode recorded, got %#v", record.Learning.Adaptation)
	}
}

func TestBuildReplayRoutingRecordPrefersTypedLearningSessionPolicy(t *testing.T) {
	ctx := replayRoutingRecordMetadataTestContext()
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(
		&selection.SessionPolicyTrace{
			Phase:             "typed_phase",
			CurrentModel:      "model-a",
			BaseSelectedModel: "model-c",
			SelectedModel:     "model-b",
			DecisionReason:    "typed_policy_wins",
		},
		routerLearningIdentityDiagnostics{},
	)
	ctx.VSRLearningPolicies = testLearningPolicies(policy)

	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")
	diagnostics := record.RouteDiagnostics
	if diagnostics == nil {
		t.Fatal("expected route diagnostics")
	}
	if diagnostics.SessionPhase != "typed_phase" ||
		diagnostics.PreviousModel != "model-a" ||
		diagnostics.SelectedModel != "model-b" ||
		diagnostics.SessionReason != "typed_policy_wins" {
		t.Fatalf("expected replay diagnostics from typed learning policy, got %#v", diagnostics)
	}
}

func TestBuildReplayRoutingRecordUsesObservedFinalModel(t *testing.T) {
	ctx := replayRoutingRecordMetadataTestContext()
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Mode = config.DecisionAdaptationModeObserve
	policy.Action = routerLearningActionObserve
	policy.Reason = "observe_only"
	policy.Scope = config.RouterLearningScopeConversation
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(
		&selection.SessionPolicyTrace{
			Phase:             "user_turn",
			CurrentModel:      "model-a",
			BaseSelectedModel: "model-c",
			SelectedModel:     "would-have-held-model",
			DecisionReason:    "switch_guard_hold",
		},
		routerLearningIdentityDiagnostics{},
	)
	policy.Details.Protection.finalModel = "model-b"
	ctx.VSRLearningPolicies = testLearningPolicies(policy)

	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")
	diagnostics := record.RouteDiagnostics
	if diagnostics == nil {
		t.Fatal("expected route diagnostics")
	}
	if diagnostics.SelectedModel != "model-b" ||
		diagnostics.SelectedModel == "would-have-held-model" ||
		diagnostics.SessionReason != "observe_only" {
		t.Fatalf("expected observe diagnostics to keep actual final model, got %#v", diagnostics)
	}
}

func replayTestLearningPolicy(
	method routerLearningMethod,
	mode string,
	action routerLearningAction,
	reason string,
	scope string,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(method)
	policy.Mode = mode
	policy.Action = action
	policy.Reason = reason
	policy.Scope = scope
	return policy
}

func replayTestProtectionPolicyWithTrace(trace *selection.SessionPolicyTrace) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Mode = config.DecisionAdaptationModeApply
	policy.Action = routerLearningActionAllowSwitch
	policy.Reason = "switch_allowed"
	policy.Scope = config.RouterLearningScopeConversation
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(
		trace,
		routerLearningIdentityDiagnostics{},
	)
	return policy
}

func assertReplayRoutingMetadata(t *testing.T, record routerreplay.RoutingRecord) {
	t.Helper()
	if record.SessionID != "sess-replay-test" {
		t.Fatalf("expected session_id copied, got %q", record.SessionID)
	}
	if record.TurnIndex != 2 {
		t.Fatalf("expected turn_index=2, got %d", record.TurnIndex)
	}
	if record.DecisionTier != 3 {
		t.Fatalf("expected decision tier=3, got %d", record.DecisionTier)
	}
	if record.DecisionPriority != 120 {
		t.Fatalf("expected decision priority=120, got %d", record.DecisionPriority)
	}
	if !reflect.DeepEqual(record.Projections, []string{"balance_reasoning"}) {
		t.Fatalf("unexpected projections: %#v", record.Projections)
	}
	if got := record.ProjectionScores["reasoning_pressure"]; got != 0.73 {
		t.Fatalf("expected projection score 0.73, got %v", got)
	}
	if got := record.SignalConfidences["projection:balance_reasoning"]; got != 0.73 {
		t.Fatalf("expected projection confidence 0.73, got %v", got)
	}
	if got := record.SignalValues["reask:persistently_dissatisfied"]; got != 2 {
		t.Fatalf("expected signal value 2, got %v", got)
	}
}

func assertReplayRouteDiagnostics(t *testing.T, record routerreplay.RoutingRecord) {
	t.Helper()
	diagnostics := record.RouteDiagnostics
	if diagnostics == nil {
		t.Fatal("expected route diagnostics")
	}
	assertReplayRouteDecision(t, diagnostics)
	assertReplayRouteModels(t, diagnostics)
	assertReplayRouteSession(t, diagnostics)
}

func assertReplayRouteDecision(t *testing.T, diagnostics *routerreplay.RouteDiagnostics) {
	t.Helper()
	if diagnostics.Decision != "balance" || diagnostics.SelectionMethod != "router_dc" {
		t.Fatalf("unexpected diagnostics decision/method: %#v", diagnostics)
	}
	if diagnostics.DecisionReason != "switch_has_best_adjusted_score" {
		t.Fatalf("expected raw decision reason preserved in details, got %#v", diagnostics)
	}
}

func assertReplayRouteModels(t *testing.T, diagnostics *routerreplay.RouteDiagnostics) {
	t.Helper()
	if diagnostics.OriginalModel != "model-a" ||
		diagnostics.ProposalModel != "model-c" ||
		diagnostics.PreviousModel != "model-a" ||
		diagnostics.SelectedModel != "model-b" {
		t.Fatalf("unexpected diagnostics models: %#v", diagnostics)
	}
}

func assertReplayRouteSession(t *testing.T, diagnostics *routerreplay.RouteDiagnostics) {
	t.Helper()
	if !diagnostics.SessionPolicyApplied ||
		diagnostics.SessionAction != "switch" ||
		diagnostics.SessionPhase != "user_turn" ||
		diagnostics.SessionReason != "switch_allowed" {
		t.Fatalf("unexpected diagnostics session summary: %#v", diagnostics)
	}
}

func TestBuildReplayRoutingRecordUsesBaselineSessionAction(t *testing.T) {
	ctx := replayRoutingRecordMetadataTestContext()
	ctx.VSRLearningPolicies = testLearningPolicies(
		replayTestProtectionPolicyWithTrace(&selection.SessionPolicyTrace{
			Phase:             "user_turn",
			BaseSelectedModel: "model-a",
			SelectedModel:     "model-a",
			DecisionReason:    "missing_previous_model",
		}),
	)
	policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
	if !ok {
		t.Fatal("expected protection policy fixture")
	}
	policy.Action = routerLearningActionEstablishBaseline
	policy.Reason = "new_conversation"
	ctx.VSRLearningPolicies.Set(policy)

	record := buildReplayRoutingRecord(ctx, "model-a", "model-a", "balance")
	diagnostics := record.RouteDiagnostics
	if diagnostics == nil {
		t.Fatal("expected route diagnostics")
	}
	if diagnostics.SessionAction != replaySessionActionEstablishBaseline ||
		diagnostics.SessionReason != "new_conversation" {
		t.Fatalf("expected baseline replay summary, got %#v", diagnostics)
	}
	if diagnostics.DecisionReason != "missing_previous_model" {
		t.Fatalf("expected raw trace reason preserved, got %#v", diagnostics)
	}
}

func assertReplayMatchedSignals(t *testing.T, record routerreplay.RoutingRecord) {
	t.Helper()
	if !reflect.DeepEqual(record.Signals.Modality, []string{"AR"}) {
		t.Fatalf("unexpected modality signals: %#v", record.Signals.Modality)
	}
	if !reflect.DeepEqual(record.Signals.Authz, []string{"premium_tier"}) {
		t.Fatalf("unexpected authz signals: %#v", record.Signals.Authz)
	}
	if !reflect.DeepEqual(record.Signals.Jailbreak, []string{"jailbreak_detector"}) {
		t.Fatalf("unexpected jailbreak signals: %#v", record.Signals.Jailbreak)
	}
	if !reflect.DeepEqual(record.Signals.PII, []string{"email_block"}) {
		t.Fatalf("unexpected pii signals: %#v", record.Signals.PII)
	}
	if !reflect.DeepEqual(record.Signals.KB, []string{"privacy_kb"}) {
		t.Fatalf("unexpected kb signals: %#v", record.Signals.KB)
	}
	if !reflect.DeepEqual(record.Signals.Conversation, []string{"multi_turn_user"}) {
		t.Fatalf("unexpected conversation signals: %#v", record.Signals.Conversation)
	}
	if !reflect.DeepEqual(record.Signals.Event, []string{"critical_payment_event"}) {
		t.Fatalf("unexpected event signals: %#v", record.Signals.Event)
	}
}

func TestBuildReplayRoutingRecord_CacheSimilarityAndContextTokenCount(t *testing.T) {
	ctx := &RequestContext{
		RequestID:            "req-cache-ctx",
		VSRCacheSimilarity:   0.87,
		VSRContextTokenCount: 1234,
	}

	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")

	if record.CacheSimilarity != 0.87 {
		t.Fatalf("expected cache_similarity 0.87 copied to record, got %v", record.CacheSimilarity)
	}
	if record.ContextTokenCount != 1234 {
		t.Fatalf("expected context_token_count 1234 copied to record, got %d", record.ContextTokenCount)
	}
}

func TestBuildReplayRoutingRecord_ResponseAPIChainFields(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req-resp-1",
		SessionID: "conv-chain",
		TurnIndex: 2,
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-chain",
			PreviousResponseID:   "resp_prev_1",
		},
	}
	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")
	if record.SessionID != "conv-chain" || record.TurnIndex != 2 {
		t.Fatalf("unexpected session fields: session_id=%q turn_index=%d", record.SessionID, record.TurnIndex)
	}
	if record.ConversationID != "conv-chain" || record.PreviousResponseID != "resp_prev_1" {
		t.Fatalf("unexpected response API persistence: conversation_id=%q previous_response_id=%q",
			record.ConversationID, record.PreviousResponseID)
	}
}
