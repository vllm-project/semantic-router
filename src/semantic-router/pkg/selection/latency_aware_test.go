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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
)

func TestLatencyAwareSelector_Select(t *testing.T) {
	ctx := context.Background()

	t.Run("select fastest model with TPOT percentile only", func(t *testing.T) {
		latency.ResetTPOT()
		latency.ResetTTFT()

		selector := NewLatencyAwareSelector(nil)
		latency.UpdateTPOT("model-a", 0.06)
		latency.UpdateTPOT("model-a", 0.06)
		latency.UpdateTPOT("model-a", 0.06)

		latency.UpdateTPOT("model-b", 0.03)
		latency.UpdateTPOT("model-b", 0.03)
		latency.UpdateTPOT("model-b", 0.03)

		result, err := selector.Select(ctx, &SelectionContext{
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
	})

	t.Run("select fastest model with TTFT percentile only", func(t *testing.T) {
		latency.ResetTPOT()
		latency.ResetTTFT()

		selector := NewLatencyAwareSelector(nil)
		latency.UpdateTTFT("model-a", 0.30)
		latency.UpdateTTFT("model-a", 0.30)
		latency.UpdateTTFT("model-a", 0.30)

		latency.UpdateTTFT("model-b", 0.12)
		latency.UpdateTTFT("model-b", 0.12)
		latency.UpdateTTFT("model-b", 0.12)

		result, err := selector.Select(ctx, &SelectionContext{
			CandidateModels:            createCandidateModels("model-a", "model-b"),
			LatencyAwareTTFTPercentile: 50,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.SelectedModel != "model-b" {
			t.Errorf("expected model-b, got %s", result.SelectedModel)
		}
	})

	t.Run("select best combined model with TPOT and TTFT percentiles", func(t *testing.T) {
		latency.ResetTPOT()
		latency.ResetTTFT()

		selector := NewLatencyAwareSelector(nil)

		latency.UpdateTPOT("model-a", 0.02)
		latency.UpdateTPOT("model-a", 0.02)
		latency.UpdateTPOT("model-a", 0.02)
		latency.UpdateTTFT("model-a", 0.60)
		latency.UpdateTTFT("model-a", 0.60)
		latency.UpdateTTFT("model-a", 0.60)

		latency.UpdateTPOT("model-b", 0.05)
		latency.UpdateTPOT("model-b", 0.05)
		latency.UpdateTPOT("model-b", 0.05)
		latency.UpdateTTFT("model-b", 0.20)
		latency.UpdateTTFT("model-b", 0.20)
		latency.UpdateTTFT("model-b", 0.20)

		latency.UpdateTPOT("model-c", 0.08)
		latency.UpdateTPOT("model-c", 0.08)
		latency.UpdateTPOT("model-c", 0.08)
		latency.UpdateTTFT("model-c", 0.10)
		latency.UpdateTTFT("model-c", 0.10)
		latency.UpdateTTFT("model-c", 0.10)

		result, err := selector.Select(ctx, &SelectionContext{
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
	})

	t.Run("fallback to first candidate when no stats available", func(t *testing.T) {
		latency.ResetTPOT()
		latency.ResetTTFT()

		selector := NewLatencyAwareSelector(nil)
		result, err := selector.Select(ctx, &SelectionContext{
			CandidateModels:            createCandidateModels("model-a", "model-b"),
			LatencyAwareTPOTPercentile: 50,
			LatencyAwareTTFTPercentile: 50,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.SelectedModel != "model-a" {
			t.Errorf("expected fallback model-a, got %s", result.SelectedModel)
		}
	})

	t.Run("fallback to first candidate when latency_aware config is missing", func(t *testing.T) {
		latency.ResetTPOT()
		latency.ResetTTFT()

		selector := NewLatencyAwareSelector(nil)
		result, err := selector.Select(ctx, &SelectionContext{
			CandidateModels: createCandidateModels("model-a", "model-b"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result.SelectedModel != "model-a" {
			t.Errorf("expected fallback model-a, got %s", result.SelectedModel)
		}
	})

	t.Run("return error when no candidates", func(t *testing.T) {
		latency.ResetTPOT()
		latency.ResetTTFT()

		selector := NewLatencyAwareSelector(nil)
		_, err := selector.Select(ctx, &SelectionContext{
			CandidateModels:            nil,
			LatencyAwareTPOTPercentile: 50,
		})
		if err == nil {
			t.Fatal("expected error but got nil")
		}
	})
}
