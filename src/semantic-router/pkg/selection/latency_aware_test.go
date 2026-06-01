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
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
)

func resetLatencyAwareStats() {
	latency.ResetTPOT()
	latency.ResetTTFT()
}

func TestLatencyAwareSelectorSelectsFastestTPOT(t *testing.T) {
	resetLatencyAwareStats()

	selector := NewLatencyAwareSelector(nil)
	latency.UpdateTPOT("model-a", 0.06)
	latency.UpdateTPOT("model-a", 0.06)
	latency.UpdateTPOT("model-a", 0.06)
	latency.UpdateTPOT("model-b", 0.03)
	latency.UpdateTPOT("model-b", 0.03)
	latency.UpdateTPOT("model-b", 0.03)

	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels:            createCandidateModels("model-a", "model-b"),
		LatencyAwareTPOTPercentile: 50,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Method != MethodLatencyAware {
		t.Errorf("expected method %s, got %s", MethodLatencyAware, result.Method)
	}
	if result.SelectedModel != "model-b" {
		t.Errorf("expected model-b, got %s", result.SelectedModel)
	}
}

func TestLatencyAwareSelectorSelectsFastestTTFT(t *testing.T) {
	resetLatencyAwareStats()

	selector := NewLatencyAwareSelector(nil)
	latency.UpdateTTFT("model-a", 0.30)
	latency.UpdateTTFT("model-a", 0.30)
	latency.UpdateTTFT("model-a", 0.30)
	latency.UpdateTTFT("model-b", 0.12)
	latency.UpdateTTFT("model-b", 0.12)
	latency.UpdateTTFT("model-b", 0.12)

	result, err := selector.Select(context.Background(), &SelectionContext{
		CandidateModels:            createCandidateModels("model-a", "model-b"),
		LatencyAwareTTFTPercentile: 50,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SelectedModel != "model-b" {
		t.Errorf("expected model-b, got %s", result.SelectedModel)
	}
}

func TestLatencyAwareSelectorSelectsBestCombinedLatency(t *testing.T) {
	resetLatencyAwareStats()
	seedCombinedLatencyStats()

	result, err := NewLatencyAwareSelector(nil).Select(context.Background(), &SelectionContext{
		CandidateModels:            createCandidateModels("model-a", "model-b", "model-c"),
		LatencyAwareTPOTPercentile: 50,
		LatencyAwareTTFTPercentile: 50,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SelectedModel != "model-b" {
		t.Errorf("expected model-b, got %s", result.SelectedModel)
	}
}

func seedCombinedLatencyStats() {
	for i := 0; i < 3; i++ {
		latency.UpdateTPOT("model-a", 0.02)
		latency.UpdateTTFT("model-a", 0.60)
		latency.UpdateTPOT("model-b", 0.05)
		latency.UpdateTTFT("model-b", 0.20)
		latency.UpdateTPOT("model-c", 0.08)
		latency.UpdateTTFT("model-c", 0.10)
	}
}

func TestLatencyAwareSelectorDefaults(t *testing.T) {
	for _, tc := range []struct {
		name   string
		selCtx *SelectionContext
	}{
		{
			name: "no stats available",
			selCtx: &SelectionContext{
				CandidateModels:            createCandidateModels("model-a", "model-b"),
				LatencyAwareTPOTPercentile: 50,
				LatencyAwareTTFTPercentile: 50,
			},
		},
		{
			name: "missing latency aware config",
			selCtx: &SelectionContext{
				CandidateModels: createCandidateModels("model-a", "model-b"),
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resetLatencyAwareStats()
			result, err := NewLatencyAwareSelector(nil).Select(context.Background(), tc.selCtx)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.SelectedModel != "model-a" {
				t.Errorf("expected default model-a, got %s", result.SelectedModel)
			}
			if !strings.Contains(result.Reasoning, "using first candidate as default") {
				t.Errorf("expected default-candidate reasoning, got %q", result.Reasoning)
			}
		})
	}
}

func TestLatencyAwareSelectorRejectsInvalidInput(t *testing.T) {
	t.Run("no candidates", func(t *testing.T) {
		resetLatencyAwareStats()
		_, err := NewLatencyAwareSelector(nil).Select(context.Background(), &SelectionContext{
			LatencyAwareTPOTPercentile: 50,
		})
		if err == nil {
			t.Fatal("expected error but got nil")
		}
	})

	t.Run("invalid percentiles", func(t *testing.T) {
		resetLatencyAwareStats()
		for _, selCtx := range []*SelectionContext{
			{CandidateModels: createCandidateModels("model-a"), LatencyAwareTPOTPercentile: 101},
			{CandidateModels: createCandidateModels("model-a"), LatencyAwareTTFTPercentile: -1},
		} {
			_, err := NewLatencyAwareSelector(nil).Select(context.Background(), selCtx)
			if !errors.Is(err, ErrLatencyAwarePercentileInvalid) {
				t.Fatalf("expected %v, got %v", ErrLatencyAwarePercentileInvalid, err)
			}
		}
	})
}
