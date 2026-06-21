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
		VSRSessionPolicy: map[string]interface{}{
			"phase":               "user_turn",
			"current_model":       "model-a",
			"base_selected_model": "model-c",
			"selected_model":      "model-b",
			"decision_reason":     "switch_has_best_adjusted_score",
		},
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
	ctx.VSRLearningPolicies = map[routerLearningMethod]routerLearningPolicy{
		routerLearningMethodSessionAware: replayTestLearningPolicy(
			routerLearningMethodSessionAware,
			"apply",
			routerLearningActionStay,
			"cache_hot",
			"conversation",
		),
		routerLearningMethodBandit: replayTestLearningPolicy(
			routerLearningMethodBandit,
			"observe",
			routerLearningActionSwitch,
			"quality_goal",
			"request",
		),
	}

	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")

	if record.Learning == nil {
		t.Fatal("expected learning diagnostics")
	}
	if got := record.Learning.Adaptations["session_aware"]["action"]; got != "stay" {
		t.Fatalf("expected session_aware action recorded, got %#v", record.Learning.Adaptations)
	}
	if got := record.Learning.Adaptations["bandit"]["mode"]; got != "observe" {
		t.Fatalf("expected bandit mode recorded, got %#v", record.Learning.Adaptations)
	}
}

func TestBuildReplayRoutingRecordPrefersTypedLearningSessionPolicy(t *testing.T) {
	ctx := replayRoutingRecordMetadataTestContext()
	ctx.VSRSessionPolicy = map[string]interface{}{
		"phase":          "stale_map_phase",
		"current_model":  "stale-model",
		"selected_model": "stale-selected",
	}
	policy := newRouterLearningPolicy(routerLearningMethodSessionAware)
	policy.Details.SessionAware = newSessionAwareLearningDiagnostics(
		&selection.SessionPolicyTrace{
			Phase:             "typed_phase",
			CurrentModel:      "model-a",
			BaseSelectedModel: "model-c",
			SelectedModel:     "model-b",
			DecisionReason:    "typed_policy_wins",
		},
		sessionAwareIdentityDiagnostics{},
	)
	ctx.VSRLearningPolicies = map[routerLearningMethod]routerLearningPolicy{
		routerLearningMethodSessionAware: policy,
	}

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
	if diagnostics.Decision != "balance" || diagnostics.SelectionMethod != "router_dc" {
		t.Fatalf("unexpected diagnostics decision/method: %#v", diagnostics)
	}
	if diagnostics.OriginalModel != "model-a" ||
		diagnostics.ProposalModel != "model-c" ||
		diagnostics.PreviousModel != "model-a" ||
		diagnostics.SelectedModel != "model-b" {
		t.Fatalf("unexpected diagnostics models: %#v", diagnostics)
	}
	if !diagnostics.SessionPolicyApplied ||
		diagnostics.SessionAction != "switch" ||
		diagnostics.SessionPhase != "user_turn" ||
		diagnostics.SessionReason != "switch_has_best_adjusted_score" {
		t.Fatalf("unexpected diagnostics session summary: %#v", diagnostics)
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
