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
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// StaticConfig configures the static selector
type StaticConfig struct {
	// UseFirstCandidate always selects the first candidate (default behavior)
	UseFirstCandidate bool `yaml:"use_first_candidate"`

	// CategoryScores maps category -> model -> score (from config)
	CategoryScores map[string]map[string]float64 `yaml:"-"`
}

// DefaultStaticConfig returns the default Static configuration
func DefaultStaticConfig() *StaticConfig {
	return &StaticConfig{
		UseFirstCandidate: true,
		CategoryScores:    make(map[string]map[string]float64),
	}
}

// StaticSelector implements static model selection based on configuration scores
// This is the baseline selector that uses pre-configured scores without learning.
type StaticSelector struct {
	config *StaticConfig

	// Category-specific scores from configuration
	categoryScores map[string]map[string]float64
	scoresMu       sync.RWMutex
}

// NewStaticSelector creates a new Static selector
func NewStaticSelector(cfg *StaticConfig) *StaticSelector {
	if cfg == nil {
		cfg = DefaultStaticConfig()
	}
	return &StaticSelector{
		config:         cfg,
		categoryScores: make(map[string]map[string]float64),
	}
}

// Method returns the selection method type
func (s *StaticSelector) Method() SelectionMethod {
	return MethodStatic
}

// InitializeFromConfig sets up static scores from model configuration
func (s *StaticSelector) InitializeFromConfig(categories []config.Category) {
	s.scoresMu.Lock()
	defer s.scoresMu.Unlock()

	for _, category := range categories {
		if s.categoryScores[category.Name] == nil {
			s.categoryScores[category.Name] = make(map[string]float64)
		}
		for _, ms := range category.ModelScores {
			s.categoryScores[category.Name][ms.Model] = ms.Score
		}
	}

	logging.Infof("[StaticSelector] Initialized scores for %d categories", len(s.categoryScores))
}

// SetCategoryScore sets a static score for a model in a category
func (s *StaticSelector) SetCategoryScore(category, model string, score float64) {
	s.scoresMu.Lock()
	defer s.scoresMu.Unlock()

	if s.categoryScores[category] == nil {
		s.categoryScores[category] = make(map[string]float64)
	}
	s.categoryScores[category][model] = score
}

// Select chooses the best model based on static configuration scores
func (s *StaticSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	allScores := make(map[string]float64)
	var bestModel *config.ModelRef
	var bestScore float64

	s.scoresMu.RLock()
	categoryScores := s.categoryScores[selCtx.DecisionName]
	s.scoresMu.RUnlock()

	for i := range selCtx.CandidateModels {
		model := &selCtx.CandidateModels[i]

		// Get static score if available
		score := 1.0 // Default score
		if categoryScores != nil {
			if cs, ok := categoryScores[model.Model]; ok {
				score = cs
			}
		}

		allScores[model.Model] = score

		if score > bestScore || bestModel == nil {
			bestScore = score
			bestModel = model
		}
	}

	// If no scores found and useFirstCandidate is true, use first
	if bestModel == nil || (s.config.UseFirstCandidate && bestScore == 1.0) {
		bestModel = &selCtx.CandidateModels[0]
		bestScore = allScores[bestModel.Model]
	}

	reasoning := "Static selection from configuration"
	if selCtx.DecisionName != "" {
		reasoning = fmt.Sprintf("Static selection for category '%s'", selCtx.DecisionName)
	}

	logging.Infof("[StaticSelector] Candidates: %v → Selected: %s (using first/highest)",
		getModelNames(selCtx.CandidateModels), bestModel.Model)

	return &SelectionResult{
		SelectedModel: bestModel.Model,
		LoRAName:      bestModel.LoRAName,
		Score:         bestScore,
		Confidence:    1.0, // Static selection is always "confident"
		Method:        MethodStatic,
		Reasoning:     reasoning,
		AllScores:     allScores,
	}, nil
}

// UpdateFeedback does nothing for static selector (no learning)
func (s *StaticSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	// Static selector doesn't learn from feedback
	logging.Debugf("[StaticSelector] Ignoring feedback (static selector does not learn)")
	return nil
}

// getModelNames extracts model names from ModelRef slice
func getModelNames(models []config.ModelRef) []string {
	names := make([]string, len(models))
	for i, m := range models {
		names[i] = m.Model
	}
	return names
}
