package topiccontinuity

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// liveToolSpan builds a tool continuation whose live span holds n assistant
// messages between the user turn and the matched tool result.
func liveToolSpan(n int, blocksPerCall int) []llmprotocol.Message {
	messages := conversation(unrelatedHistory(1), user("run the job"))
	for i := 0; i < n; i++ {
		messages = append(messages, assistant("working"))
	}
	calls := make([]llmprotocol.Content, 0, blocksPerCall)
	for i := 0; i < blocksPerCall; i++ {
		calls = append(calls, toolCall("t1", "job", ""))
	}
	return append(messages, assistantCalls(calls...), toolResult("t1", "ok"))
}

func TestTerminalOutcomes(t *testing.T) {
	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	type row struct {
		name      string
		run       func() Result
		reason    Reason
		coverage  Coverage
		capFlag   bool
		zeroFeats bool
	}
	cfg := defaultConfig(defaultPolicy)
	run := func(ctx context.Context, messages []llmprotocol.Message, available bool) Result {
		prepared := prepare(ctx, messages, available, defaultPolicy)
		return classify(cfg, prepared, extract(ctx, prepared))
	}
	rows := []row{
		{"invalid rule config", func() Result {
			invalid := cfg
			invalid.Continuation = 1
			return EvaluateAll(context.Background(), func() ([]llmprotocol.Message, bool) {
				return conversation(unrelatedHistory(1), user("next step please")), true
			}, []EvalConfig{invalid}).Rules[0].Result
		}, ReasonInternalError, CoveragePartial, false, true},
		{"loader unavailable", func() Result {
			return run(context.Background(), nil, false)
		}, ReasonHistoryUnavailable, CoveragePartial, false, true},
		{"available and empty", func() Result {
			return run(context.Background(), []llmprotocol.Message{}, true)
		}, ReasonNoLiveUserTurn, CoveragePartial, false, true},
		{"final assistant", func() Result {
			return run(context.Background(), unrelatedHistory(1), true)
		}, ReasonNoLiveUserTurn, CoveragePartial, false, true},
		{"orphan tool result", func() Result {
			return run(context.Background(), conversation(user("go"), toolResult("t9", "x")), true)
		}, ReasonOrphanToolResult, CoveragePartial, false, true},
		{"scan cap in live phase", func() Result {
			return run(context.Background(), liveToolSpan(maxScanMessages+10, 1), true)
		}, ReasonOverBudget, CoveragePartial, false, true},
		{"block cap in live phase", func() Result {
			return run(context.Background(), liveToolSpan(0, maxContentBlocksPerPrepare+4), true)
		}, ReasonOverBudget, CoveragePartial, true, true},
		{"cancelled in prepare", func() Result {
			return run(cancelledCtx, conversation(unrelatedHistory(1), user("next step please")), true)
		}, ReasonCancelled, CoveragePartial, false, true},
		{"no prior turn", func() Result {
			return run(context.Background(), conversation(user("first question about routers")), true)
		}, ReasonNoPriorTurn, CoverageFull, false, true},
		{"opaque-only live turn", func() Result {
			return run(context.Background(), conversation(unrelatedHistory(1),
				llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{image()}}), true)
		}, ReasonOpaqueOnly, CoverageFull, false, false},
	}
	for _, tc := range rows {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.run()
			assertInvariants(t, result)
			if result.Class != ClassUnknown || result.Confidence != 0 {
				t.Fatalf("class %s confidence %v", result.Class, result.Confidence)
			}
			if result.Reason != tc.reason || result.Coverage != tc.coverage {
				t.Fatalf("got %s/%s, want %s/%s", result.Reason, result.Coverage, tc.reason, tc.coverage)
			}
			if result.Fallback != fallbackReason(tc.reason) {
				t.Fatalf("fallback %v", result.Fallback)
			}
			if result.Scope.FeatureCapReached != tc.capFlag {
				t.Fatalf("FeatureCapReached = %v, want %v", result.Scope.FeatureCapReached, tc.capFlag)
			}
			if tc.zeroFeats && !reflect.DeepEqual(result.Features, Features{}) {
				t.Fatalf("features not zero: %+v", result.Features)
			}
			if !result.Scope.AssistantIncluded {
				t.Fatal("AssistantIncluded must reflect the policy")
			}
		})
	}
}

func historyWithToolResult() []llmprotocol.Message {
	return conversation(user("Find the config"), assistantCalls(toolCall("t1", "read_file", `{"p":1}`)),
		toolResult("t1", "contents"), assistant("Found it."), user("Explain the routing section"))
}

func TestCancellationKeepsObservedScope(t *testing.T) {
	prepared := prepare(context.Background(), historyWithToolResult(), true, defaultPolicy)
	if !prepared.Scope.ExcludedContentPresent {
		t.Fatal("precondition: excluded content observed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cancelled := classify(defaultConfig(defaultPolicy), prepared, extract(ctx, prepared))
	if cancelled.Reason != ReasonCancelled || !cancelled.Scope.ExcludedContentPresent {
		t.Fatalf("cancellation should keep scope: %+v", cancelled)
	}
}

func evaluateGroups(messages []llmprotocol.Message, configs []EvalConfig) map[string]Result {
	load := func() ([]llmprotocol.Message, bool) { return messages, true }
	results := map[string]Result{}
	for _, rule := range EvaluateAll(context.Background(), load, configs).Rules {
		results[rule.Result.Signal] = rule.Result
	}
	return results
}

func TestDeterminism(t *testing.T) {
	for _, tc := range goldenCases() {
		policy := defaultPolicy
		if tc.policy != nil {
			policy = *tc.policy
		}
		first := evaluate(tc.messages, policy)
		for i := 0; i < 100; i++ {
			if again := evaluate(tc.messages, policy); !reflect.DeepEqual(first, again) {
				t.Fatalf("%s: run %d differs", tc.name, i)
			}
		}
	}
}
