package config

import (
	"strings"
	"testing"
)

func TestNativeCascadeBudgetBelongsToSelectedAlgorithm(t *testing.T) {
	raw := nativeRoutingTestYAML
	if _, err := ParseYAMLBytes([]byte(raw)); err != nil {
		t.Fatalf("algorithm-local budget rejected: %v", err)
	}
}

func TestNativeLearnedPolicyIsNotAProductAlgorithm(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "type: cascade", "type: policy\n            policy: {source: policy.json, sha256: "+strings.Repeat("a", 64)+"}", 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil {
		t.Fatal("removed native policy accepted")
	}
}

func TestNativeCascadeRejectsChatCandidateMinimum(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "type: cascade", "type: cascade\n            minimum_candidates: 2", 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil || !strings.Contains(err.Error(), "minimum_candidates") {
		t.Fatalf("native cascade accepted an unenforced Chat candidate minimum: %v", err)
	}
}

func TestLegacyRecipeBudgetAndChatAlgorithmBudgetAreRejected(t *testing.T) {
	for _, legacy := range []string{
		strings.Replace(nativeRoutingTestYAML, "    routing:\n", "    routing:\n      budget: {deadline: 3s, max_calls: 4}\n", 1),
		strings.Replace(nativeRoutingTestYAML, "entrypoints:\n", "routing:\n  budget: {deadline: 3s, max_calls: 4}\nentrypoints:\n", 1),
	} {
		if _, err := ParseYAMLBytes([]byte(legacy)); err == nil || !strings.Contains(err.Error(), "budget") {
			t.Fatalf("old routing budget accepted: %v", err)
		}
	}
	algorithm := &AlgorithmConfig{Type: "static", Budget: &AlgorithmBudget{Deadline: "3s", MaxCalls: 1}}
	if err := validateDecisionAlgorithmConfig("chat", []ModelRef{{Model: "chat"}}, algorithm); err == nil || !strings.Contains(err.Error(), "require cascade") {
		t.Fatalf("Chat algorithm silently ignored budget: %v", err)
	}
}
