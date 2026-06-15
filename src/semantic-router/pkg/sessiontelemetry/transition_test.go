package sessiontelemetry

import (
	"testing"
)

func TestRecordAndGetTransitions(t *testing.T) {
	ResetTransitionsForTesting()

	evt := ModelTransitionEvent{
		SessionID:           "sess-abc",
		TurnIndex:           2,
		FromModel:           "model-a",
		ToModel:             "model-b",
		TTFTMs:              350.0,
		CacheWarmthEstimate: 0.65,
	}
	RecordTransition(evt)

	got := GetTransitions("sess-abc")
	if len(got) != 1 {
		t.Fatalf("expected 1 transition event, got %d", len(got))
	}
	if got[0].FromModel != "model-a" || got[0].ToModel != "model-b" {
		t.Errorf("unexpected event content: %+v", got[0])
	}
	if got[0].CacheWarmthEstimate != 0.65 {
		t.Errorf("expected CacheWarmthEstimate 0.65, got %f", got[0].CacheWarmthEstimate)
	}
}

func TestGetTransitions_EmptyForUnknownSession(t *testing.T) {
	ResetTransitionsForTesting()
	if got := GetTransitions("nonexistent-session"); len(got) != 0 {
		t.Errorf("expected empty slice for unknown session, got %d events", len(got))
	}
}

func TestRecordTransition_DropsEmptySessionID(t *testing.T) {
	ResetTransitionsForTesting()
	RecordTransition(ModelTransitionEvent{SessionID: "", FromModel: "a", ToModel: "b"})
	if got := GetTransitions(""); len(got) != 0 {
		t.Errorf("expected no events for empty session ID, got %d", len(got))
	}
}

func TestRecordTransition_CapsAtMaxPerSession(t *testing.T) {
	ResetTransitionsForTesting()
	for i := 0; i < MaxTransitionEventsPerSession+50; i++ {
		RecordTransition(ModelTransitionEvent{
			SessionID: "sess-cap",
			FromModel: "a",
			ToModel:   "b",
		})
	}
	got := GetTransitions("sess-cap")
	if len(got) != MaxTransitionEventsPerSession {
		t.Errorf("expected cap of %d, got %d", MaxTransitionEventsPerSession, len(got))
	}
}

func TestResetTransitionLog(t *testing.T) {
	ResetTransitionsForTesting()
	RecordTransition(ModelTransitionEvent{SessionID: "sess-reset", FromModel: "a", ToModel: "b"})
	ResetTransitionsForTesting()
	if got := GetTransitions("sess-reset"); len(got) != 0 {
		t.Errorf("expected empty after reset, got %d events", len(got))
	}
}
