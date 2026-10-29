package selection

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func sessionAwareReleaseSelector() *SessionAwareSelector {
	selector := NewSessionAwareSelector(&SessionAwareConfig{
		BaseMethod:             MethodStatic,
		MinTurnsBeforeSwitch:   10,
		SwitchMargin:           0.05,
		StayBias:               0.10,
		ToolLoopHardLock:       true,
		PrefixCacheWeight:      0.20,
		HandoffPenaltyWeight:   1,
		DefaultHandoffPenalty:  0.05,
		QualityGapMultiplier:   1,
		MaxCacheCostMultiplier: 2.5,
		SwitchHistoryWeight:    0.04,
	})
	selector.SetBaseSelector(stubSessionBase{result: &SelectionResult{
		SelectedModel: "frontier", Score: 1.0, Method: MethodStatic,
		AllScores: map[string]float64{"current": 0.999, "frontier": 1.0},
	}})
	return selector
}

func sessionAwareReleaseContext(release, toolLoop bool) *SelectionContext {
	return &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "frontier"}, {Model: "current"}},
		AgenticSession: &AgenticSessionContext{
			ID: "sess-1", TurnIndex: 3, PreviousModel: "current",
			MemoryTurnCount: 3, MemorySwitchCount: 4,
			HistoryTokens: 4096, ContextTokens: 8192,
			CacheWarmth: 1, CacheWarmthOK: true,
			ActiveToolLoop:   toolLoop,
			ReleaseRequested: release,
		},
	}
}

func TestSessionAwareReleaseResetsContinuity(t *testing.T) {
	held, err := sessionAwareReleaseSelector().Select(context.Background(), sessionAwareReleaseContext(false, false))
	if err != nil || held.SelectedModel != "current" {
		t.Fatalf("without release expected current held, got %+v, %v", held, err)
	}

	result, err := sessionAwareReleaseSelector().Select(context.Background(), sessionAwareReleaseContext(true, false))
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	if result.SelectedModel != "frontier" {
		t.Fatalf("expected release to reselect frontier, got %q", result.SelectedModel)
	}
	if result.SessionPolicy == nil || !result.SessionPolicy.ExplicitRelease {
		t.Fatalf("expected explicit release in trace, got %#v", result.SessionPolicy)
	}
	if got := result.SessionPolicy.ToMap()["explicit_release"]; got != true {
		t.Fatalf("expected explicit_release in trace map, got %#v", got)
	}
	trace := result.SessionPolicy.CandidateTraces["frontier"]
	if trace.HandoffPenalty != 0 || trace.PrefixCachePenalty != 0 || trace.SwitchHistoryPenalty != 0 {
		t.Fatalf("expected release to clear continuity penalties, got %#v", trace)
	}
}

func TestSessionAwareReleaseKeepsToolLoopHardLock(t *testing.T) {
	result, err := sessionAwareReleaseSelector().Select(context.Background(), sessionAwareReleaseContext(true, true))
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	if result.SelectedModel != "current" || result.SessionPolicy.HardLockReason != "hard_lock=tool_loop" {
		t.Fatalf("expected tool loop to keep current, got %q (%q)", result.SelectedModel, result.SessionPolicy.HardLockReason)
	}
}
