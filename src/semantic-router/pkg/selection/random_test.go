package selection

import (
	"context"
	"errors"
	"testing"
)

func TestRandomSelector_UniformDistribution(t *testing.T) {
	selector := NewRandomSelector()
	selCtx := &SelectionContext{CandidateModels: createCandidateModels("model-a", "model-b", "model-c")}

	const draws = 10000
	counts := make(map[string]int, 3)
	for i := 0; i < draws; i++ {
		result, err := selector.Select(context.Background(), selCtx)
		if err != nil {
			t.Fatalf("Select failed on draw %d: %v", i, err)
		}
		counts[result.SelectedModel]++
	}

	if len(counts) != 3 {
		t.Fatalf("expected all 3 candidates to be selected at least once, got %v", counts)
	}
	// Loose bounds around the 1/3 expectation: catches a biased or constant
	// index without flaking on real RNG variance.
	for model, count := range counts {
		share := float64(count) / float64(draws)
		if share < 0.20 || share > 0.45 {
			t.Errorf("model %q selected %.1f%% of the time, want roughly 33%% (counts: %v)", model, share*100, counts)
		}
	}
}

func TestRandomSelector_NeverSelectsExcludedModels(t *testing.T) {
	selector := NewRandomSelector()
	eligible := createCandidateModels("model-a", "model-c", "model-e")
	excluded := map[string]bool{"model-b": true, "model-d": true}
	selCtx := &SelectionContext{CandidateModels: eligible}

	for i := 0; i < 1000; i++ {
		result, err := selector.Select(context.Background(), selCtx)
		if err != nil {
			t.Fatalf("Select failed on draw %d: %v", i, err)
		}
		if excluded[result.SelectedModel] {
			t.Fatalf("draw %d selected filtered-out model %q", i, result.SelectedModel)
		}
	}
}

func TestRandomSelector_EmptyCandidatePool(t *testing.T) {
	selector := NewRandomSelector()

	result, err := selector.Select(context.Background(), &SelectionContext{CandidateModels: nil})
	if !errors.Is(err, ErrCandidateModelsRequired) {
		t.Fatalf("Select(nil candidates) error = %v, want %v", err, ErrCandidateModelsRequired)
	}
	if result != nil {
		t.Errorf("Select(nil candidates) returned result %+v, want nil", result)
	}
}

func TestRandomSelector_SingletonPool(t *testing.T) {
	selector := NewRandomSelector()
	selCtx := &SelectionContext{CandidateModels: createCandidateModels("only-model")}

	result, err := selector.Select(context.Background(), selCtx)
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}
	if result.SelectedModel != "only-model" {
		t.Errorf("SelectedModel = %q, want %q", result.SelectedModel, "only-model")
	}
	if result.Method != MethodRandom {
		t.Errorf("Method = %q, want %q", result.Method, MethodRandom)
	}
	if result.Score != 1.0 {
		t.Errorf("Score = %v, want 1.0", result.Score)
	}
}

func TestRandomSelector_DeterministicSeam(t *testing.T) {
	selector := NewRandomSelector()
	selector.intn = func(int) int { return 1 }
	selCtx := &SelectionContext{CandidateModels: createCandidateModels("model-a", "model-b", "model-c")}

	for i := 0; i < 5; i++ {
		result, err := selector.Select(context.Background(), selCtx)
		if err != nil {
			t.Fatalf("Select failed: %v", err)
		}
		if result.SelectedModel != "model-b" {
			t.Fatalf("draw %d selected %q, want %q", i, result.SelectedModel, "model-b")
		}
	}
}

func TestRandomSelector_PopulatesAllScores(t *testing.T) {
	selector := NewRandomSelector()
	selCtx := &SelectionContext{CandidateModels: createCandidateModels("model-a", "model-b", "model-c", "model-d")}

	result, err := selector.Select(context.Background(), selCtx)
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}
	if len(result.AllScores) != 4 {
		t.Fatalf("AllScores has %d entries, want 4: %v", len(result.AllScores), result.AllScores)
	}
	for model, score := range result.AllScores {
		if score != 0.25 {
			t.Errorf("AllScores[%q] = %v, want 0.25", model, score)
		}
	}
	if result.Confidence != 0.25 {
		t.Errorf("Confidence = %v, want 0.25", result.Confidence)
	}
}

func TestRandomSelector_MethodAndTier(t *testing.T) {
	selector := NewRandomSelector()

	if selector.Method() != MethodRandom {
		t.Errorf("Method() = %q, want %q", selector.Method(), MethodRandom)
	}
	if selector.Tier() != TierSupported {
		t.Errorf("Tier() = %q, want %q", selector.Tier(), TierSupported)
	}
	if deps := selector.ExternalDependencies(); len(deps) != 0 {
		t.Errorf("ExternalDependencies() = %v, want empty", deps)
	}
	if err := selector.UpdateFeedback(context.Background(), &Feedback{}); err != nil {
		t.Errorf("UpdateFeedback() = %v, want nil", err)
	}
}

func TestRandomSelector_FactoryAndRegistry(t *testing.T) {
	created := NewFactory(&ModelSelectionConfig{Method: "random"}).Create()
	if _, ok := created.(*RandomSelector); !ok {
		t.Fatalf("Create() returned %T, want *RandomSelector", created)
	}

	registered, ok := NewFactory(DefaultModelSelectionConfig()).CreateAll().Get(MethodRandom)
	if !ok {
		t.Fatal("CreateAll() did not register MethodRandom")
	}
	if _, ok := registered.(*RandomSelector); !ok {
		t.Fatalf("registry holds %T for MethodRandom, want *RandomSelector", registered)
	}
}
