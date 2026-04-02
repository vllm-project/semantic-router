package extproc

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildReplayRoutingRecordCapturesRoutingMetadata(t *testing.T) {
	ctx := &RequestContext{
		RequestID:                     "req-1",
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

	record := buildReplayRoutingRecord(ctx, "model-a", "model-b", "balance")
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
