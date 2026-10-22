package config

import "testing"

func TestSameModelRepoFollowsTheHuggingFaceOrgMove(t *testing.T) {
	for _, tc := range []struct {
		a, b string
		same bool
	}{
		{"vllm-sr/Vela-1.0-Omni-Nano", "vllm-sr/Vela-1.0-Omni-Nano", true},
		{"vllm-sr/Vela-1.0-Omni-Nano", "llm-semantic-router/Vela-1.0-Omni-Nano", true},
		{"LLM-Semantic-Router/halugate-sentinel", "vllm-sr/halugate-sentinel", true},
		{"vllm-sr/Vela-1.0-Omni-Nano", "llm-semantic-router/Vela-1.0-Omni-Mini", false},
		{"vllm-sr/Vela-1.0-Omni-Nano", "example/Vela-1.0-Omni-Nano", false},
		{"llm-semantic-router", "vllm-sr", false},
	} {
		if got := SameModelRepo(tc.a, tc.b); got != tc.same {
			t.Errorf("SameModelRepo(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.same)
		}
	}
}
