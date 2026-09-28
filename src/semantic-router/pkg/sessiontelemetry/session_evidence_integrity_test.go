package sessiontelemetry

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

func TestOutcomeFirstWriteAfterRestartPreservesStateAndPolicy(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := newFakeSessionStateStore()
	SetRouterSessionStateStore(store)
	t.Cleanup(func() { SetRouterSessionStateStore(nil); ResetRouterSessionMemoryForTesting() })
	now := time.Now()
	ConfigureTurnOutcomeWindow("restored", 16, time.Hour, now)
	RecordSessionDecision(SessionDecisionParams{SessionID: "restored", SelectedModel: "a", Timestamp: now.Add(-30 * time.Minute)})
	RecordSessionDecision(SessionDecisionParams{SessionID: "restored", SelectedModel: "b", Timestamp: now.Add(-25 * time.Minute)})
	for i := 0; i < 12; i++ {
		RecordTurnOutcome("restored", TurnOutcome{RequestID: fmt.Sprint(i), Model: "b", Category: TurnProgress, Source: TurnSourceRouterObserved}, now.Add(time.Duration(i-24)*time.Minute))
	}
	ResetRouterSessionMemoryForTesting()
	RecordTurnOutcome("restored", TurnOutcome{RequestID: "1", Model: "b", Category: TurnRegression, Source: TurnSourceOutcomeIngest}, now.Add(-23*time.Minute))
	got, ok := GetRouterSessionSnapshot("restored", now)
	if !ok || got.CurrentModel != "b" || got.SwitchCount != 1 || got.LastSwitchAt.IsZero() || len(got.SwitchTimestamps) != 1 {
		t.Fatalf("decision history lost on first ingest: %+v", got)
	}
	window := RecentTurnOutcomesWithPolicy("restored", now, 16, time.Hour)
	if len(window) != 12 || window[1].Category != TurnRegression || got.OutcomeWindowSize != 16 || got.OutcomeWindowTTLSeconds != 3600 {
		t.Fatalf("policy/window lost on restart: %+v, %+v", got, window)
	}
}

func TestOutcomeRecoveryGuardIsGateScoped(t *testing.T) {
	for _, tc := range []struct {
		name       string
		localSize  int
		sharedSize int
		wantModel  string
	}{
		{"gate_absent", 0, 0, "shared"},
		{"local_gate", 8, 0, "local"},
		{"restored_gate", 0, 8, "local"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ResetRouterSessionMemoryForTesting()
			t.Cleanup(ResetRouterSessionMemoryForTesting)
			now := time.Now()
			hydrateRouterSessionSnapshot(RouterSessionSnapshot{
				SessionID: tc.name, CurrentModel: "local", LastSeen: now,
				OutcomeWindowSize: tc.localSize,
			})
			hydrateRouterSessionSnapshot(RouterSessionSnapshot{
				SessionID: tc.name, CurrentModel: "shared", LastSeen: now.Add(-time.Second),
				OutcomeWindowSize: tc.sharedSize,
			})
			got, ok := GetRouterSessionSnapshot(tc.name, now)
			if !ok || got.CurrentModel != tc.wantModel {
				t.Fatalf("recovery changed outside its gate scope: %+v", got)
			}
		})
	}
}

func TestOutcomeMergeIsOrderIndependentAndModelScoped(t *testing.T) {
	base := time.Now().Add(-time.Minute)
	capture := TurnOutcome{RequestID: "req", Model: "a", Timestamp: base.UnixMilli(), Category: TurnProgress, OutputTokens: 4, Source: TurnSourceRouterObserved}
	ingest := TurnOutcome{RequestID: "req", Model: "a", Timestamp: base.Add(10 * time.Second).UnixMilli(), Category: TurnRegression, Confidence: 0.8, ConfidenceKnown: true, Source: TurnSourceOutcomeIngest}
	middle := TurnOutcome{RequestID: "other", Model: "a", Timestamp: base.Add(5 * time.Second).UnixMilli(), Category: TurnProgress, Source: TurnSourceRouterObserved}
	build := func(first, last TurnOutcome) []TurnOutcome {
		out := appendTurnOutcome(nil, first, base.Add(20*time.Second), 8, time.Minute)
		out = appendTurnOutcome(out, middle, base.Add(20*time.Second), 8, time.Minute)
		return appendTurnOutcome(out, last, base.Add(20*time.Second), 8, time.Minute)
	}
	a, b := build(capture, ingest), build(ingest, capture)
	if !reflect.DeepEqual(a, b) || len(a) != 2 || a[0].Timestamp != capture.Timestamp {
		t.Fatalf("arrival order changed the window: %+v vs %+v", a, b)
	}
	a = appendTurnOutcome(a, capture, base.Add(20*time.Second), 8, time.Minute)
	if len(a) != 2 {
		t.Fatal("duplicate capture is not idempotent")
	}
	capture.Model = "b"
	a = appendTurnOutcome(a, capture, base.Add(20*time.Second), 8, time.Minute)
	if len(a) != 3 {
		t.Fatal("same request on different models was merged")
	}
	capture.RequestID, ingest.RequestID = "", ""
	if sameTurn(capture, ingest) {
		t.Fatal("turn index alone cannot identify retries")
	}
}

func TestIngestCannotReattributeProviderFailure(t *testing.T) {
	for _, category := range []TurnOutcomeCategory{TurnProviderError, TurnToolError} {
		response := TurnOutcome{Category: category, Source: TurnSourceRouterObserved, OutputTokens: 4}
		ingest := TurnOutcome{Category: TurnRegression, Source: TurnSourceOutcomeIngest}
		for _, got := range []TurnOutcome{mergeTurnOutcome(response, ingest), mergeTurnOutcome(ingest, response)} {
			if got.Category != category || got.ModelAttributable || got.OutputTokens != 4 {
				t.Fatalf("infra failure reattributed: %+v", got)
			}
		}
	}
}

func TestFutureEvidenceAndSwitchesAreExcluded(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	now := time.Now()
	RecordTurnOutcome("future", TurnOutcome{RequestID: "future", Model: "a", Category: TurnRegression}, now.Add(time.Hour))
	if got := RecentTurnOutcomes("future", now); len(got) != 0 {
		t.Fatalf("future evidence: %+v", got)
	}
	if got := CountRecentSwitches([]int64{now.Add(-time.Second).UnixMilli(), now.Add(time.Hour).UnixMilli()}, time.Minute, now); got != 1 {
		t.Fatalf("future switch counted: %d", got)
	}
}
