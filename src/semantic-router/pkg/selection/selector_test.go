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

func TestRegistry(t *testing.T) {
	registry := NewRegistry()

	// Register selectors
	registry.Register(MethodRandom, NewRandomSelector())
	registry.Register(MethodStatic, NewStaticSelector(nil))

	// Get registered selectors
	randomSelector, ok := registry.Get(MethodRandom)
	if !ok || randomSelector == nil {
		t.Error("expected Random selector to be registered")
	}

	staticSelector, ok := registry.Get(MethodStatic)
	if !ok || staticSelector == nil {
		t.Error("expected Static selector to be registered")
	}

	// Get unregistered selector
	_, ok = registry.Get(MethodMultiFactor)
	if ok {
		t.Error("expected MultiFactor to not be registered")
	}
}

// The registry a recipe shares holds the selectors without a per-decision
// block: static, random and the default multi_factor. systemone and a
// decision's own multi_factor block are built for their decision.
func TestFactoryCreateAllRegistersTheSharedSelectors(t *testing.T) {
	registry := NewFactory().CreateAll()
	for _, method := range []SelectionMethod{MethodStatic, MethodRandom, MethodMultiFactor} {
		selector, ok := registry.Get(method)
		if !ok || selector.Method() != method {
			t.Fatalf("registry holds %v for %s", selector, method)
		}
	}
	if _, ok := registry.Get(MethodSystemOne); ok {
		t.Fatal("the systemone selector is built per decision, not shared")
	}
}

func TestStaticSelectorScoresByMatchedDomain(t *testing.T) {
	selector := NewStaticSelector(DefaultStaticConfig())
	selector.InitializeFromConfig([]config.Category{
		{
			Name: "business",
			ModelScores: []config.ModelScore{
				{Model: "small", Score: 0.2},
				{Model: "large", Score: 0.9},
			},
		},
		{Name: "unscored"},
	})

	tests := []struct {
		name         string
		decisionName string
		categoryName string
		want         string
	}{
		{name: "decision named differently from the matched domain", decisionName: "business_route", categoryName: "business", want: "large"},
		{name: "decision named after the domain without a category", decisionName: "business", want: "large"},
		{name: "matched domain without scores falls back to the decision", decisionName: "business", categoryName: "unscored", want: "large"},
		{name: "no scores for decision or domain", decisionName: "other_route", categoryName: "unscored", want: "small"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := selector.Select(context.Background(), &SelectionContext{
				DecisionName:    tt.decisionName,
				CategoryName:    tt.categoryName,
				CandidateModels: createCandidateModels("small", "large"),
			})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.SelectedModel != tt.want {
				t.Errorf("expected %s, got %s (scores %v)", tt.want, result.SelectedModel, result.AllScores)
			}
		})
	}
}
