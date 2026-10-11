package topiccontinuity

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestEvaluateAllNoRulesNeverLoads(t *testing.T) {
	loaded := false
	evaluation := EvaluateAll(context.Background(), func() ([]llmprotocol.Message, bool) {
		loaded = true
		return nil, true
	}, nil)
	if loaded || len(evaluation.Rules) != 0 || len(evaluation.PreparationInputBytes) != 0 {
		t.Fatalf("loaded=%v evaluation=%+v", loaded, evaluation)
	}
}

func TestEvaluateAllPreservesOrderAndSharesExtraction(t *testing.T) {
	messages := conversation(unrelatedHistory(2), user("What is the capital city of Peru and its current population?"))
	userOnly := defaultPolicy
	userOnly.IncludeAssistant = false
	rules := []EvalConfig{
		{Name: "z", Policy: defaultPolicy, Continuation: 0.35, Change: 0.08},
		{Name: "a", Policy: userOnly, Continuation: 0.35, Change: 0.08},
		{Name: "m", Policy: defaultPolicy, Continuation: 0.5, Change: 0.05},
	}
	loads := 0
	evaluation := EvaluateAll(context.Background(), func() ([]llmprotocol.Message, bool) {
		loads++
		return messages, true
	}, rules)
	// One preparation per distinct policy: two here.
	if loads != 1 || len(evaluation.PreparationInputBytes) != 2 {
		t.Fatalf("loads=%d preparations=%d", loads, len(evaluation.PreparationInputBytes))
	}
	for i, rule := range evaluation.Rules {
		if rule.Result.Signal != rules[i].Name {
			t.Fatalf("order: position %d is %q, want %q", i, rule.Result.Signal, rules[i].Name)
		}
		if want := evaluateConfig(messages, rules[i]); !reflect.DeepEqual(rule.Result, want) {
			t.Fatalf("%s: shared evaluation differs from a standalone one", rules[i].Name)
		}
		if rule.ClassifyMicros < 0 || rule.PrepareExtractMicros < 0 {
			t.Fatalf("negative timing: %+v", rule)
		}
	}
	if evaluation.Rules[0].PrepareExtractMicros != evaluation.Rules[2].PrepareExtractMicros {
		t.Fatal("rules in one group should report the group's shared cost")
	}
}

func TestEvaluateAllEightRulesOnePolicyPrepareOnce(t *testing.T) {
	rules := make([]EvalConfig, 8)
	for i := range rules {
		rules[i] = EvalConfig{Name: string(rune('a' + i)), Policy: defaultPolicy, Continuation: 0.35, Change: 0.08}
	}
	evaluation := EvaluateAll(context.Background(), func() ([]llmprotocol.Message, bool) {
		return conversation(unrelatedHistory(1), user("next steps for the loader please")), true
	}, rules)
	if len(evaluation.PreparationInputBytes) != 1 || len(evaluation.Rules) != 8 {
		t.Fatalf("preparations=%d rules=%d, want 1 and 8", len(evaluation.PreparationInputBytes), len(evaluation.Rules))
	}
}

func TestEvaluateAllUnavailableHistory(t *testing.T) {
	evaluation := EvaluateAll(context.Background(), func() ([]llmprotocol.Message, bool) { return nil, false },
		[]EvalConfig{defaultConfig(defaultPolicy)})
	result := evaluation.Rules[0].Result
	assertInvariants(t, result)
	if result.Reason != ReasonHistoryUnavailable || !result.Fallback {
		t.Fatalf("got %+v", result)
	}
}
