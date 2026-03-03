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

func TestPRISMValidator_Validate(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name            string
		modelDomains    map[string][]string
		modelDescs      map[string]string
		modelName       string
		query           string
		threshold       float64
		expectLegit     bool
		expectScoreGt0  bool
	}{
		{
			name:         "model with no domain boundaries is accepted",
			modelDomains: map[string][]string{},
			modelDescs:   map[string]string{},
			modelName:    "general-model",
			query:        "What is quantum computing?",
			threshold:    0.3,
			expectLegit:  true,
		},
		{
			name: "query matching domain keywords is accepted",
			modelDomains: map[string][]string{
				"physics-model": {"physics", "quantum", "mechanics"},
			},
			modelDescs:     map[string]string{},
			modelName:      "physics-model",
			query:          "Explain quantum mechanics",
			threshold:      0.3,
			expectLegit:    true,
			expectScoreGt0: true,
		},
		{
			name: "query not matching domain keywords is refused",
			modelDomains: map[string][]string{
				"physics-model": {"physics", "quantum", "mechanics"},
			},
			modelDescs:  map[string]string{},
			modelName:   "physics-model",
			query:       "Write a poem about love",
			threshold:   0.3,
			expectLegit: false,
		},
		{
			name: "partial keyword match with low threshold is accepted",
			modelDomains: map[string][]string{
				"code-model": {"code", "programming", "software", "debug", "algorithm"},
			},
			modelDescs:     map[string]string{},
			modelName:      "code-model",
			query:          "Help me debug this code",
			threshold:      0.3,
			expectLegit:    true,
			expectScoreGt0: true,
		},
		{
			name: "partial keyword match with high threshold is refused",
			modelDomains: map[string][]string{
				"code-model": {"code", "programming", "software", "debug", "algorithm"},
			},
			modelDescs:     map[string]string{},
			modelName:      "code-model",
			query:          "Help me debug this code",
			threshold:      0.8,
			expectLegit:    false,
			expectScoreGt0: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &PRISMConfig{
				Enabled:         true,
				Mode:            PRISMModeFineFilter,
				DomainThreshold: tt.threshold,
				RefusalPolicy:   PRISMRefusalReroute,
			}
			v := NewPRISMValidator(cfg)
			v.modelDomains = tt.modelDomains
			v.modelDescriptions = tt.modelDescs

			result := v.Validate(ctx, tt.modelName, tt.query)

			if result.Legitimate != tt.expectLegit {
				t.Errorf("expected legitimate=%v, got %v (score=%.3f, reason=%s)",
					tt.expectLegit, result.Legitimate, result.Score, result.Reason)
			}

			if tt.expectScoreGt0 && result.Score <= 0 {
				t.Errorf("expected score > 0, got %.3f", result.Score)
			}

			if result.ModelName != tt.modelName {
				t.Errorf("expected model name %s, got %s", tt.modelName, result.ModelName)
			}
		})
	}
}

func TestPRISMValidator_FilterCandidates(t *testing.T) {
	ctx := context.Background()

	cfg := &PRISMConfig{
		Enabled:         true,
		Mode:            PRISMModeCoarseFilter,
		DomainThreshold: 0.3,
		RefusalPolicy:   PRISMRefusalReroute,
	}
	v := NewPRISMValidator(cfg)
	v.modelDomains = map[string][]string{
		"physics-model": {"physics", "quantum", "mechanics"},
		"code-model":    {"code", "programming", "software"},
		"general-model": {},
	}

	candidates := []config.ModelRef{
		{Model: "physics-model"},
		{Model: "code-model"},
		{Model: "general-model"},
	}

	// Physics query should keep physics-model and general-model (no boundaries = accepted)
	filtered := v.FilterCandidates(ctx, candidates, "Explain quantum physics")

	hasPhysics := false
	hasGeneral := false
	for _, m := range filtered {
		if m.Model == "physics-model" {
			hasPhysics = true
		}
		if m.Model == "general-model" {
			hasGeneral = true
		}
	}

	if !hasPhysics {
		t.Error("expected physics-model to pass coarse filter for physics query")
	}
	if !hasGeneral {
		t.Error("expected general-model to pass coarse filter (no domain boundaries)")
	}
}

func TestPRISMValidator_FilterCandidates_AllRefused(t *testing.T) {
	ctx := context.Background()

	cfg := &PRISMConfig{
		Enabled:         true,
		Mode:            PRISMModeCoarseFilter,
		DomainThreshold: 0.99, // Very high threshold — everything gets refused
		RefusalPolicy:   PRISMRefusalReroute,
	}
	v := NewPRISMValidator(cfg)
	v.modelDomains = map[string][]string{
		"model-a": {"physics"},
		"model-b": {"code"},
	}

	candidates := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
	}

	// With impossible threshold, should fall back to original list
	filtered := v.FilterCandidates(ctx, candidates, "unrelated query about cooking")
	if len(filtered) != len(candidates) {
		t.Errorf("expected fallback to original candidates (%d), got %d", len(candidates), len(filtered))
	}
}

func TestPRISMValidator_Disabled(t *testing.T) {
	ctx := context.Background()

	cfg := &PRISMConfig{
		Enabled: false,
	}
	v := NewPRISMValidator(cfg)

	candidates := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
	}

	// FilterCandidates should pass through when disabled
	filtered := v.FilterCandidates(ctx, candidates, "any query")
	if len(filtered) != len(candidates) {
		t.Errorf("expected all candidates when disabled, got %d", len(filtered))
	}
}

func TestPRISMValidator_InitializeFromConfig(t *testing.T) {
	v := NewPRISMValidator(DefaultPRISMConfig())

	modelConfig := map[string]config.ModelParams{
		"physics-model": {
			Description:  "Specialized in physics, quantum mechanics, and thermodynamics",
			Capabilities: []string{"physics", "quantum"},
		},
		"code-model": {
			Description: "Expert at code generation and debugging",
		},
		"empty-model": {},
	}

	v.InitializeFromConfig(modelConfig)

	v.mu.RLock()
	defer v.mu.RUnlock()

	// physics-model should have description keywords + capabilities
	if domains, ok := v.modelDomains["physics-model"]; !ok || len(domains) == 0 {
		t.Error("expected physics-model to have domain keywords")
	}

	// code-model should have description keywords
	if domains, ok := v.modelDomains["code-model"]; !ok || len(domains) == 0 {
		t.Error("expected code-model to have domain keywords")
	}

	// empty-model should have no domains
	if domains, ok := v.modelDomains["empty-model"]; ok && len(domains) > 0 {
		t.Errorf("expected empty-model to have no domain keywords, got %v", domains)
	}
}

func TestExtractDomainKeywords(t *testing.T) {
	tests := []struct {
		name        string
		description string
		expectMin   int // minimum expected keywords
	}{
		{
			name:        "extracts meaningful words",
			description: "Specialized in physics, quantum mechanics, and thermodynamics",
			expectMin:   3, // physics, specialized, quantum, mechanics, thermodynamics
		},
		{
			name:        "filters stop words",
			description: "A model for the analysis of data",
			expectMin:   1, // analysis, data
		},
		{
			name:        "empty description",
			description: "",
			expectMin:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			keywords := extractDomainKeywords(tt.description)
			if len(keywords) < tt.expectMin {
				t.Errorf("expected at least %d keywords, got %d: %v", tt.expectMin, len(keywords), keywords)
			}
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float64
		epsilon  float64
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			epsilon:  0.001,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 0.0,
			epsilon:  0.001,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0},
			b:        []float32{-1, 0},
			expected: -1.0,
			epsilon:  0.001,
		},
		{
			name:     "empty vectors",
			a:        []float32{},
			b:        []float32{},
			expected: 0.0,
			epsilon:  0.001,
		},
		{
			name:     "different length vectors",
			a:        []float32{1, 0},
			b:        []float32{1, 0, 0},
			expected: 0.0,
			epsilon:  0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			if diff := result - tt.expected; diff > tt.epsilon || diff < -tt.epsilon {
				t.Errorf("expected %.3f, got %.3f", tt.expected, result)
			}
		})
	}
}

func TestDefaultPRISMConfig(t *testing.T) {
	cfg := DefaultPRISMConfig()

	if cfg.Enabled {
		t.Error("expected default config to be disabled")
	}
	if cfg.Mode != PRISMModeFineFilter {
		t.Errorf("expected default mode %s, got %s", PRISMModeFineFilter, cfg.Mode)
	}
	if cfg.DomainThreshold != 0.3 {
		t.Errorf("expected default threshold 0.3, got %f", cfg.DomainThreshold)
	}
	if cfg.RefusalPolicy != PRISMRefusalReroute {
		t.Errorf("expected default refusal policy %s, got %s", PRISMRefusalReroute, cfg.RefusalPolicy)
	}
	if cfg.MaxRerouteAttempts != 3 {
		t.Errorf("expected default max reroute attempts 3, got %d", cfg.MaxRerouteAttempts)
	}
}
