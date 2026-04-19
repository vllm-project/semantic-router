package extproc

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
		VSRSelectedDecision: &config.Decision{
			Name:     "balance",
			Tier:     3,
			Priority: 120,
		},
		VSRMatchedKeywords:   []string{"math_keyword"},
		VSRMatchedModality:   []string{"AR"},
		VSRMatchedAuthz:      []string{"premium_tier"},
		VSRMatchedJailbreak:  []string{"jailbreak_detector"},
		VSRMatchedPII:        []string{"email_block"},
		VSRMatchedKB:         []string{"privacy_kb"},
		VSRMatchedProjection: []string{"balance_reasoning"},
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
	record := buildReplayRoutingRecord(replayRoutingRecordMetadataTestContext(), "model-a", "model-b", "balance")
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

func TestBuildReplayRoutingRecordCapturesSignalMetadata(t *testing.T) {
	record := buildReplayRoutingRecord(replayRoutingRecordMetadataTestContext(), "model-a", "model-b", "balance")
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
