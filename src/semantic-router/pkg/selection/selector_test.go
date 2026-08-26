/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Test helper to create candidate models
func createCandidateModels(names ...string) []config.ModelRef {
	models := make([]config.ModelRef, len(names))
	for i, name := range names {
		models[i] = config.ModelRef{Model: name}
	}
	return models
}

func TestEloSelector_Select(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name          string
		candidates    []config.ModelRef
		setupRatings  map[string]float64
		expectedModel string
		expectError   bool
	}{
		{
			name:          "select highest rated model",
			candidates:    createCandidateModels("model-a", "model-b", "model-c"),
			setupRatings:  map[string]float64{"model-a": 1400, "model-b": 1600, "model-c": 1500},
			expectedModel: "model-b",
			expectError:   false,
		},
		{
			name:          "fallback to default rating",
			candidates:    createCandidateModels("new-model-1", "new-model-2"),
			setupRatings:  map[string]float64{},
			expectedModel: "new-model-1", // First model when equal ratings
			expectError:   false,
		},
		{
			name:          "no candidates",
			candidates:    []config.ModelRef{},
			setupRatings:  map[string]float64{},
			expectedModel: "",
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selector := NewEloSelector(DefaultEloConfig())

			// Setup ratings
			for model, rating := range tt.setupRatings {
				selector.setGlobalRating(model, &ModelRating{Model: model, Rating: rating})
			}

			selCtx := &SelectionContext{
				Query:           "test query",
				CandidateModels: tt.candidates,
			}

			result, err := selector.Select(ctx, selCtx)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if result.SelectedModel != tt.expectedModel {
				t.Errorf("expected model %s, got %s", tt.expectedModel, result.SelectedModel)
			}

			if result.Method != MethodElo {
				t.Errorf("expected method %s, got %s", MethodElo, result.Method)
			}
		})
	}
}

func TestEloSelector_UpdateFeedback(t *testing.T) {
	ctx := context.Background()
	selector := NewEloSelector(DefaultEloConfig())

	// Initialize ratings
	selector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1500})
	selector.setGlobalRating("model-b", &ModelRating{Model: "model-b", Rating: 1500})

	// Submit feedback: model-a wins against model-b
	feedback := &Feedback{
		Query:       "test query",
		WinnerModel: "model-a",
		LoserModel:  "model-b",
		Tie:         false,
	}

	err := selector.UpdateFeedback(ctx, feedback)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check ratings updated
	ratingA := selector.getGlobalRating("model-a")
	ratingB := selector.getGlobalRating("model-b")

	if ratingA == nil {
		t.Fatal("rating A should not be nil")
		return // Explicit return after t.Fatal for staticcheck
	}
	if ratingB == nil {
		t.Fatal("rating B should not be nil")
		return // Explicit return after t.Fatal for staticcheck
	}

	if ratingA.Rating <= 1500 {
		t.Errorf("winner rating should increase, got %f", ratingA.Rating)
	}

	if ratingB.Rating >= 1500 {
		t.Errorf("loser rating should decrease, got %f", ratingB.Rating)
	}

	if ratingA.Wins != 1 {
		t.Errorf("winner wins should be 1, got %d", ratingA.Wins)
	}

	if ratingB.Losses != 1 {
		t.Errorf("loser losses should be 1, got %d", ratingB.Losses)
	}
}

func TestRouterDCSelector_Select(t *testing.T) {
	ctx := context.Background()

	selector := NewRouterDCSelector(DefaultRouterDCConfig())

	// Set up embedding function (mock)
	selector.SetEmbeddingFunc(func(text string) ([]float32, error) {
		// Return a simple embedding based on text length
		embedding := make([]float32, 768)
		for i := range embedding {
			embedding[i] = float32(len(text)%10) / 10.0
		}
		return embedding, nil
	})

	// Set model embeddings
	modelAEmb := make([]float32, 768)
	modelBEmb := make([]float32, 768)
	for i := range modelAEmb {
		modelAEmb[i] = 0.5
		modelBEmb[i] = 0.3
	}
	selector.SetModelEmbedding("model-a", modelAEmb)
	selector.SetModelEmbedding("model-b", modelBEmb)

	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("model-a", "model-b"),
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodRouterDC {
		t.Errorf("expected method %s, got %s", MethodRouterDC, result.Method)
	}

	if result.Score <= 0 {
		t.Errorf("expected positive score, got %f", result.Score)
	}
}

func TestAutoMixSelector_Select(t *testing.T) {
	ctx := context.Background()

	selector := NewAutoMixSelector(DefaultAutoMixConfig())

	// Initialize capabilities
	modelConfig := map[string]config.ModelParams{
		"small-model": {Pricing: config.ModelPricing{PromptPer1M: 0.5}},
		"large-model": {Pricing: config.ModelPricing{PromptPer1M: 5.0}},
	}
	selector.InitializeFromConfig(modelConfig)

	// Set verification probabilities
	selector.SetCapability("small-model", &ModelCapability{
		Model:            "small-model",
		Cost:             0.5,
		AvgQuality:       0.7,
		VerificationProb: 0.8,
		ParamSize:        7.0,
	})
	selector.SetCapability("large-model", &ModelCapability{
		Model:            "large-model",
		Cost:             5.0,
		AvgQuality:       0.95,
		VerificationProb: 0.95,
		ParamSize:        70.0,
	})

	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("small-model", "large-model"),
		CostWeight:      0.5,
		QualityWeight:   0.5,
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodAutoMix {
		t.Errorf("expected method %s, got %s", MethodAutoMix, result.Method)
	}

	// With cost awareness, cheaper model might be selected
	if result.SelectedModel == "" {
		t.Error("expected a selected model")
	}
}

func TestHybridSelector_Select(t *testing.T) {
	ctx := context.Background()

	cfg := DefaultHybridConfig()
	cfg.ExperienceWeight = 0.5
	cfg.RouterDCWeight = 0.0 // Disable RouterDC (no embeddings)
	cfg.AutoMixWeight = 0.5
	cfg.CostWeight = 0.0

	selector := NewHybridSelector(cfg)

	// Initialize Experience component
	selector.eloSelector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1600})
	selector.eloSelector.setGlobalRating("model-b", &ModelRating{Model: "model-b", Rating: 1400})

	// Initialize AutoMix component
	selector.autoMixSelector.SetCapability("model-a", &ModelCapability{
		Model:            "model-a",
		AvgQuality:       0.9,
		VerificationProb: 0.9,
		ParamSize:        70.0,
	})
	selector.autoMixSelector.SetCapability("model-b", &ModelCapability{
		Model:            "model-b",
		AvgQuality:       0.7,
		VerificationProb: 0.8,
		ParamSize:        7.0,
	})

	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("model-a", "model-b"),
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodHybrid {
		t.Errorf("expected method %s, got %s", MethodHybrid, result.Method)
	}

	// Model-a should win (higher Elo and quality)
	if result.SelectedModel != "model-a" {
		t.Errorf("expected model-a, got %s", result.SelectedModel)
	}
}

func TestStaticSelector_Select(t *testing.T) {
	ctx := context.Background()

	selector := NewStaticSelector(DefaultStaticConfig())

	// Set up category scores
	selector.SetCategoryScore("coding", "code-model", 0.9)
	selector.SetCategoryScore("coding", "general-model", 0.5)

	selCtx := &SelectionContext{
		Query:           "write python code",
		DecisionName:    "coding",
		CandidateModels: createCandidateModels("code-model", "general-model"),
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodStatic {
		t.Errorf("expected method %s, got %s", MethodStatic, result.Method)
	}

	if result.SelectedModel != "code-model" {
		t.Errorf("expected code-model, got %s", result.SelectedModel)
	}

	if result.Score != 0.9 {
		t.Errorf("expected score 0.9, got %f", result.Score)
	}
}

func TestStaticSelectorUsesWeightedRequestAffinity(t *testing.T) {
	candidates := []config.ModelRef{
		{Model: "model-a", Weight: 1},
		{Model: "model-b", Weight: 3},
	}

	for _, test := range []struct {
		name   string
		sample float64
		want   string
	}{
		{name: "first bucket start", sample: 0, want: "model-a"},
		{name: "first bucket end", sample: 0.249999, want: "model-a"},
		{name: "second bucket start", sample: 0.25, want: "model-b"},
		{name: "second bucket end", sample: 0.999999, want: "model-b"},
	} {
		t.Run(test.name, func(t *testing.T) {
			selected, scores, ok := weightedModelRefAt(candidates, 4, test.sample)
			if !ok || selected == nil || selected.Model != test.want {
				t.Fatalf("selected = %#v, ok = %t, want %s", selected, ok, test.want)
			}
			if scores["model-a"] != 0.25 || scores["model-b"] != 0.75 {
				t.Fatalf("normalized scores = %#v", scores)
			}
		})
	}

	selector := NewStaticSelector(DefaultStaticConfig())
	selectionContext := &SelectionContext{
		AffinityKey:     "request-42",
		RecipeName:      "recipe-a",
		DecisionName:    "decision-a",
		CandidateModels: candidates,
	}
	first, err := selector.Select(context.Background(), selectionContext)
	if err != nil {
		t.Fatal(err)
	}
	second, err := selector.Select(context.Background(), selectionContext)
	if err != nil {
		t.Fatal(err)
	}
	if first.SelectedModel != second.SelectedModel || first.Reasoning != "Static weighted request affinity" {
		t.Fatalf("request affinity was not stable: first=%+v second=%+v", first, second)
	}
}

func TestStaticSelectorPreservesLegacyFirstCandidateWithoutExplicitWeights(t *testing.T) {
	selector := NewStaticSelector(DefaultStaticConfig())
	result, err := selector.Select(context.Background(), &SelectionContext{
		AffinityKey: "request-42",
		CandidateModels: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != "model-a" || result.Reasoning == "Static weighted request affinity" {
		t.Fatalf("legacy static selection changed: %+v", result)
	}
}

func TestRegistry(t *testing.T) {
	registry := NewRegistry()

	// Register selectors
	registry.Register(MethodElo, NewEloSelector(nil))
	registry.Register(MethodStatic, NewStaticSelector(nil))

	// Get registered selectors
	eloSelector, ok := registry.Get(MethodElo)
	if !ok || eloSelector == nil {
		t.Error("expected Elo selector to be registered")
	}

	staticSelector, ok := registry.Get(MethodStatic)
	if !ok || staticSelector == nil {
		t.Error("expected Static selector to be registered")
	}

	// Get unregistered selector
	_, ok = registry.Get(MethodRouterDC)
	if ok {
		t.Error("expected RouterDC to not be registered")
	}
}

func TestFactory_Create(t *testing.T) {
	tests := []struct {
		name           string
		method         string
		expectedMethod SelectionMethod
	}{
		{
			name:           "create elo selector",
			method:         "elo",
			expectedMethod: MethodElo,
		},
		{
			name:           "create router_dc selector",
			method:         "router_dc",
			expectedMethod: MethodRouterDC,
		},
		{
			name:           "create automix selector",
			method:         "automix",
			expectedMethod: MethodAutoMix,
		},
		{
			name:           "create hybrid selector",
			method:         "hybrid",
			expectedMethod: MethodHybrid,
		},
		{
			name:           "create latency_aware selector",
			method:         "latency_aware",
			expectedMethod: MethodLatencyAware,
		},
		{
			name:           "create static selector (default)",
			method:         "static",
			expectedMethod: MethodStatic,
		},
		{
			name:           "unknown method defaults to static",
			method:         "unknown",
			expectedMethod: MethodStatic,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &ModelSelectionConfig{Method: tt.method}
			factory := NewFactory(cfg)
			selector := factory.Create()

			if selector.Method() != tt.expectedMethod {
				t.Errorf("expected method %s, got %s", tt.expectedMethod, selector.Method())
			}
		})
	}
}

func TestFactory_CreateAll_IncludesMLSelectors(t *testing.T) {
	cfg := &ModelSelectionConfig{
		Method: "static",
		ML:     DefaultMLSelectorConfig(),
	}
	factory := NewFactory(cfg)
	registry := factory.CreateAll()

	// Test that ML selectors are registered
	mlMethods := []SelectionMethod{MethodKNN, MethodKMeans, MethodSVM}
	for _, method := range mlMethods {
		selector, ok := registry.Get(method)
		if !ok {
			t.Errorf("expected %s selector to be registered", method)
			continue
		}
		if selector == nil {
			t.Errorf("expected %s selector to not be nil", method)
			continue
		}
		if selector.Method() != method {
			t.Errorf("expected method %s, got %s", method, selector.Method())
		}
	}
}

func TestMLSelectorAdapter_Select(t *testing.T) {
	ctx := context.Background()

	// Create KNN adapter
	knnAdapter, err := CreateKNNSelector(DefaultMLSelectorConfig(), nil)
	if err != nil {
		t.Fatalf("failed to create KNN adapter: %v", err)
	}

	// Test selection without training - ML selectors require pretrained models
	selCtx := &SelectionContext{
		Query:           "test query for model selection",
		CandidateModels: createCandidateModels("model-a", "model-b", "model-c"),
	}

	result, err := knnAdapter.Select(ctx, selCtx)
	// ML selectors return error when not trained - this is expected behavior
	if err != nil {
		if strings.Contains(err.Error(), "model not trained") {
			t.Logf("Expected behavior: ML selector requires pretrained model: %v", err)
			return // Test passes - correct behavior for untrained model
		}
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodKNN {
		t.Errorf("expected method %s, got %s", MethodKNN, result.Method)
	}

	// Result should have a valid model
	if result.SelectedModel == "" {
		t.Error("expected a selected model")
	}
}

func TestMLSelectorAdapter_Method(t *testing.T) {
	tests := []struct {
		name           string
		createFunc     func(*MLSelectorConfig, func(string) ([]float32, error)) (*MLSelectorAdapter, error)
		expectedMethod SelectionMethod
	}{
		{
			name:           "KNN adapter method",
			createFunc:     CreateKNNSelector,
			expectedMethod: MethodKNN,
		},
		{
			name:           "KMeans adapter method",
			createFunc:     CreateKMeansSelector,
			expectedMethod: MethodKMeans,
		},
		{
			name:           "SVM adapter method",
			createFunc:     CreateSVMSelector,
			expectedMethod: MethodSVM,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter, err := tt.createFunc(DefaultMLSelectorConfig(), nil)
			if err != nil {
				t.Fatalf("failed to create adapter: %v", err)
			}

			if adapter.Method() != tt.expectedMethod {
				t.Errorf("expected method %s, got %s", tt.expectedMethod, adapter.Method())
			}
		})
	}
}

func TestMLSelectorAdapter_UpdateFeedback(t *testing.T) {
	ctx := context.Background()

	knnAdapter, err := CreateKNNSelector(DefaultMLSelectorConfig(), nil)
	if err != nil {
		t.Fatalf("failed to create KNN adapter: %v", err)
	}

	// UpdateFeedback should not error (it's a no-op for now)
	feedback := &Feedback{
		Query:       "test query",
		WinnerModel: "model-a",
		LoserModel:  "model-b",
	}

	err = knnAdapter.UpdateFeedback(ctx, feedback)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestMLSelectorAdapter_WithEmbedding(t *testing.T) {
	ctx := context.Background()

	// Create mock embedding function
	mockEmbedding := func(text string) ([]float32, error) {
		embedding := make([]float32, 768)
		for i := range embedding {
			embedding[i] = float32(len(text)%10) / 10.0
		}
		return embedding, nil
	}

	knnAdapter, err := CreateKNNSelector(DefaultMLSelectorConfig(), mockEmbedding)
	if err != nil {
		t.Fatalf("failed to create KNN adapter: %v", err)
	}

	selCtx := &SelectionContext{
		Query:           "test query with embedding",
		CandidateModels: createCandidateModels("model-a", "model-b"),
	}

	result, err := knnAdapter.Select(ctx, selCtx)
	// ML selectors return error when not trained - this is expected behavior
	if err != nil {
		if strings.Contains(err.Error(), "model not trained") {
			t.Logf("Expected behavior: ML selector requires pretrained model: %v", err)
			return // Test passes - correct behavior for untrained model
		}
		t.Fatalf("unexpected error: %v", err)
	}

	if result.SelectedModel == "" {
		t.Error("expected a selected model")
	}
}

func TestMLSelectorAdapter_GetMLSelector(t *testing.T) {
	knnAdapter, err := CreateKNNSelector(DefaultMLSelectorConfig(), nil)
	if err != nil {
		t.Fatalf("failed to create KNN adapter: %v", err)
	}

	mlSelector := knnAdapter.GetMLSelector()
	if mlSelector == nil {
		t.Error("expected ML selector to not be nil")
	}

	if mlSelector.Name() != "knn" {
		t.Errorf("expected name 'knn', got '%s'", mlSelector.Name())
	}
}
