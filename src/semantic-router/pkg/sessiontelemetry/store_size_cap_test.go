package sessiontelemetry

import (
	"strconv"
	"testing"
	"time"
)

// The three in-memory session stores derive their keys from message content, so
// distinct conversations create distinct entries. Each must bound its session
// map so memory stays bounded under high session cardinality (issue #2222).

func TestRouterSessionMemoryStoreSizeCap(t *testing.T) {
	ResetForTesting()
	orig := maxRouterSessions
	maxRouterSessions = 3
	defer func() {
		maxRouterSessions = orig
		ResetForTesting()
	}()

	base := time.Now()
	// Insert more distinct sessions than the cap. Distinct timestamps make the
	// least-recently-seen victim deterministic.
	for i := 0; i < 6; i++ {
		RecordSessionDecision(SessionDecisionParams{
			SessionID:     "sess-" + strconv.Itoa(i),
			SelectedModel: "model-a",
			Timestamp:     base.Add(time.Duration(i) * time.Second),
		})
	}

	if got := routerSessionCount(); got > maxRouterSessions {
		t.Fatalf("router session count %d exceeds cap %d", got, maxRouterSessions)
	}
	if _, ok := GetRouterSessionSnapshot("sess-0", base.Add(10*time.Second)); ok {
		t.Error("expected oldest session sess-0 to be evicted by the size cap")
	}
	if _, ok := GetRouterSessionSnapshot("sess-5", base.Add(10*time.Second)); !ok {
		t.Error("expected newest session sess-5 to be retained")
	}
}

func TestTelemetryStoreSizeCap(t *testing.T) {
	ResetForTesting()
	origMax := maxTelemetrySessions
	origNow := nowFn
	maxTelemetrySessions = 3
	// Deterministic monotonic clock so each turn gets a distinct lastSeen.
	var tick int64
	nowFn = func() time.Time { tick++; return time.Unix(tick, 0) }
	defer func() {
		maxTelemetrySessions = origMax
		nowFn = origNow
		ResetForTesting()
	}()

	for i := 0; i < 6; i++ {
		RecordTurn(TurnParams{
			RequestID:    "r" + strconv.Itoa(i),
			Model:        "model-a",
			PromptTokens: 10,
			ResponseAPI:  &ResponseAPIInput{ConversationID: "conv-" + strconv.Itoa(i)},
		})
	}

	if got := telemetrySessionCount(); got > maxTelemetrySessions {
		t.Fatalf("telemetry session count %d exceeds cap %d", got, maxTelemetrySessions)
	}
}

func TestTransitionStoreSizeCap(t *testing.T) {
	ResetTransitionsForTesting()
	orig := maxTransitionSessions
	maxTransitionSessions = 3
	defer func() {
		maxTransitionSessions = orig
		ResetTransitionsForTesting()
	}()

	base := time.Now()
	for i := 0; i < 6; i++ {
		RecordTransition(ModelTransitionEvent{
			SessionID: "sess-" + strconv.Itoa(i),
			FromModel: "a",
			ToModel:   "b",
			Timestamp: base.Add(time.Duration(i) * time.Second),
		})
	}

	if got := transitionSessionCount(); got > maxTransitionSessions {
		t.Fatalf("transition session count %d exceeds cap %d", got, maxTransitionSessions)
	}
	if len(GetTransitions("sess-0")) != 0 {
		t.Error("expected oldest transition session sess-0 to be evicted by the size cap")
	}
	if len(GetTransitions("sess-5")) == 0 {
		t.Error("expected newest transition session sess-5 to be retained")
	}
}
