package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestRouterLearningRuntimeExperienceSnapshots(t *testing.T) {
	storage := store.NewMemoryStore(10, 0)
	recorder := routerreplay.NewRecorder(storage)
	rt := newRouterLearningRuntime(nil, recorder, nil)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "replay-1",
		Decision:      "domain_code",
		DecisionTier:  4,
		SelectedModel: "model-a",
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}

	result := rt.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
		ReplayID:  "replay-1",
		Source:    routerruntime.RouterOutcomeSourceAgent,
		Target:    routerruntime.RouterOutcomeTargetModel,
		TargetRef: "model-a",
		Verdict:   routerruntime.RouterOutcomeVerdictGoodFit,
		Score:     1,
		Metadata: map[string]string{
			"decision":      "domain_code",
			"decision_tier": "4",
		},
	})
	if result.Updated != 1 || !result.Recorded {
		t.Fatalf("expected outcome to be recorded, got %#v", result)
	}

	snapshots := rt.ExperienceSnapshots()
	if len(snapshots) != 3 {
		t.Fatalf("expected exact/tier/global fallback entries, got %d: %#v", len(snapshots), snapshots)
	}
	assertExperienceSnapshot(t, findExperienceSnapshot(snapshots, "domain_code"))
}

func findExperienceSnapshot(snapshots []routerruntime.RouterExperienceSnapshot, decision string) *routerruntime.RouterExperienceSnapshot {
	for i := range snapshots {
		if snapshots[i].Decision == decision {
			return &snapshots[i]
		}
	}
	return nil
}

func assertExperienceSnapshot(t *testing.T, exact *routerruntime.RouterExperienceSnapshot) {
	t.Helper()
	if exact == nil {
		t.Fatal("expected a matching experience snapshot")
	}
	if exact.SchemaVersion != routerruntime.RouterExperienceSnapshotSchemaVersion {
		t.Fatalf("unexpected schema version: %d", exact.SchemaVersion)
	}
	if exact.Tier != 4 || exact.Model != "model-a" {
		t.Fatalf("unexpected identity: %#v", exact)
	}
	if exact.GoodFitCount != 1 || exact.SampleCount != 1 {
		t.Fatalf("unexpected evidence: %#v", exact)
	}
	if exact.Source != routerruntime.RouterExperienceSourceRuntime {
		t.Fatalf("unexpected source: %q", exact.Source)
	}
}
