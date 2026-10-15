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

import "testing"

func TestApplyCostAdjustmentUsesTheConfiguredWeightOnce(t *testing.T) {
	cfg := DefaultHybridConfig()
	cfg.CostWeight = 0.5
	selector := NewHybridSelector(cfg)
	selector.modelCosts = map[string]float64{"model-cheap": 1.0, "model-dear": 2.0}

	scores := CandidateScores{
		{Candidate: createCandidateModels("model-cheap")[0], Score: 1.0},
		{Candidate: createCandidateModels("model-dear")[0], Score: 1.0},
	}
	selector.applyCostAdjustment(&SelectionContext{}, scores)

	// The configured weight applies once: the cheapest candidate earns the
	// full bonus and the dearest none, whatever the selection context carries.
	if scores[0].Score != 1.5 {
		t.Errorf("cheap model score = %v, want 1.5", scores[0].Score)
	}
	if scores[1].Score != 1.0 {
		t.Errorf("dear model score = %v, want 1.0", scores[1].Score)
	}
}

func TestApplyCostAdjustmentPrefersTheSelectionContextOverride(t *testing.T) {
	cfg := DefaultHybridConfig()
	cfg.CostWeight = 0.2
	selector := NewHybridSelector(cfg)
	selector.modelCosts = map[string]float64{"model-cheap": 1.0, "model-dear": 2.0}

	newScores := func() CandidateScores {
		return CandidateScores{
			{Candidate: createCandidateModels("model-cheap")[0], Score: 1.0},
			{Candidate: createCandidateModels("model-dear")[0], Score: 1.0},
		}
	}

	// A decision-level cost_weight override reaches the adjustment exactly
	// once, the way Elo consumes it — not zero times, not squared.
	scores := newScores()
	selector.applyCostAdjustment(&SelectionContext{CostWeight: 0.9}, scores)
	if scores[0].Score != 1.9 {
		t.Errorf("cheap model score = %v, want 1.9", scores[0].Score)
	}

	// A zero override keeps the configured weight.
	scores = newScores()
	selector.applyCostAdjustment(&SelectionContext{}, scores)
	if scores[0].Score != 1.2 {
		t.Errorf("cheap model score = %v, want 1.2", scores[0].Score)
	}
}
