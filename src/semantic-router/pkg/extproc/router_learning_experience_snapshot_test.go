package extproc

import (
	"context"
	"reflect"
	"sort"
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

func TestRouterLearningRuntimeExperienceSnapshotsAreSortedAndRepeatable(t *testing.T) {
	rt := newRouterLearningRuntime(nil, nil, nil)
	seeds := []struct {
		decision string
		tier     int
		model    string
	}{
		{"domain_math", 2, "model-z"},
		{"domain_code", 4, "model-a"},
		{"domain_code", 4, "model-b"},
		{"domain_code", 1, "model-a"},
		{"domain_chat", 3, "model-a"},
	}
	for _, seed := range seeds {
		rt.recordModelExperience(seed.decision, seed.tier, seed.model, routerLearningOutcomeGoodFit, 1)
	}

	first := rt.ExperienceSnapshots()
	if len(first) == 0 {
		t.Fatal("expected at least one snapshot")
	}
	if !sort.SliceIsSorted(first, func(i, j int) bool {
		return experienceSnapshotLess(first[i], first[j])
	}) {
		t.Fatalf("expected snapshots sorted by (decision, tier, model), got %#v", first)
	}

	for i := 0; i < 10; i++ {
		again := rt.ExperienceSnapshots()
		if !reflect.DeepEqual(first, again) {
			t.Fatalf("expected repeatable output across calls; call %d differed:\nfirst=%#v\nagain=%#v", i, first, again)
		}
	}
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
