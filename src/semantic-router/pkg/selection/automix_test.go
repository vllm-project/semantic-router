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
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEstimateParamSizeRespectsDigitBoundaries(t *testing.T) {
	selector := NewAutoMixSelector(DefaultAutoMixConfig())

	cases := []struct {
		model string
		want  float64
	}{
		{"qwen114b", 114.0},   // "14b" must not match inside "114b"
		{"llama-170b", 170.0}, // "70b" must not match inside "170b"
		{"qwen2.5-72b", 72.0}, // version 2.5 must not become the size
		{"llama-3.1-8b", 8.0}, // version 3.1 must not become the size
		{"llama-3.2-1b", 1.0}, // sizes missing from the old table
		{"phi-4b", 4.0},
		{"qwen3-9b", 9.0},
		{"Qwen3-0.6B", 0.6}, // uppercase unit, sub-1B size
		{"gemma-0.5b-it", 0.5},
		{"gpt-4o", 7.0}, // no size token: default assumption
	}
	for _, tc := range cases {
		if got := selector.estimateParamSize(tc.model); got != tc.want {
			t.Errorf("estimateParamSize(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestGetEscalationChainOrdersByEstimatedSize(t *testing.T) {
	selector := NewAutoMixSelector(DefaultAutoMixConfig())

	chain := selector.GetEscalationChain([]config.ModelRef{
		{Model: "model-114b"},
		{Model: "model-14b"},
		{Model: "model-3b"},
	})

	want := []string{"model-3b", "model-14b", "model-114b"}
	for i, model := range want {
		if chain[i] != model {
			t.Fatalf("escalation chain = %v, want %v", chain, want)
		}
	}
}

func TestGetEscalationChainKeepsConfigOrderForEqualEstimates(t *testing.T) {
	selector := NewAutoMixSelector(DefaultAutoMixConfig())

	// Names without a size token all fall back to the default estimate;
	// the chain must keep their configuration order instead of leaving
	// it to the sort's internal choices.
	chain := selector.GetEscalationChain([]config.ModelRef{
		{Model: "gpt-4o"},
		{Model: "claude"},
		{Model: "model-3b"},
	})

	want := []string{"model-3b", "gpt-4o", "claude"}
	for i, model := range want {
		if chain[i] != model {
			t.Fatalf("escalation chain = %v, want %v", chain, want)
		}
	}
}
