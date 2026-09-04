package shadow

import (
	"context"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBudgetTryEnterCallLimit(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxCalls: 2})
	for i := 0; i < 2; i++ {
		if _, ok := b.tryEnter("a", "m"); !ok {
			t.Fatalf("arm %d should be admitted", i)
		}
	}
	res, ok := b.tryEnter("c", "m")
	if ok {
		t.Fatal("third arm must be rejected")
	}
	if res.Outcome != OutcomeSkipped {
		t.Fatalf("want OutcomeSkipped, got %s", res.Outcome)
	}
}

func TestBudgetReleaseFreesSlot(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxConcurrency: 1})
	if _, ok := b.tryEnter("a", "m"); !ok {
		t.Fatal("first arm should be admitted")
	}
	if _, ok := b.tryEnter("b", "m"); ok {
		t.Fatal("second arm must be rejected under concurrency 1")
	}
	b.release()
	if _, ok := b.tryEnter("b", "m"); !ok {
		t.Fatal("second arm must be admitted after release")
	}
}

func TestBudgetReconcileTokensAndCost(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{PricePerMillionTokens: 2.0})
	b.reconcile(OutcomeCompleted, 100, 50) // 150 tokens
	_, tokens, cost := b.used()
	if tokens != 150 {
		t.Fatalf("tokens = %d, want 150", tokens)
	}
	if want := 150.0 / 1e6 * 2.0; cost != want {
		t.Fatalf("cost = %v, want %v", cost, want)
	}
	b.reconcile(OutcomeFailed, 500, 500) // must not consume
	_, tokens, _ = b.used()
	if tokens != 150 {
		t.Fatalf("failed arm must not consume tokens, got %d", tokens)
	}
}

// TestBudgetTokenSoftLimitAfterReconcile proves token accounting is enforced
// for arms admitted after the limit is exceeded (deterministic under a
// sequential budget unit).
func TestBudgetTokenSoftLimitAfterReconcile(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxTokens: 100})
	b.reconcile(OutcomeCompleted, 60, 60) // 120 tokens > 100
	if _, ok := b.tryEnter("a", "m"); ok {
		t.Fatal("arm must be rejected after token soft limit exceeded")
	}
}

func TestBudgetCostSoftLimit(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxCost: 0.0002, PricePerMillionTokens: 2.0})
	b.reconcile(OutcomeCompleted, 200, 0) // 200/1e6*2 = 0.0004 >= 0.0002
	if _, ok := b.tryEnter("a", "m"); ok {
		t.Fatal("arm must be rejected after cost soft limit exceeded")
	}
}

// TestDispatchBudgetCallLimitSkipsRest proves the aggregate budget is enforced
// deterministically under concurrent dispatch: exactly one arm is admitted and
// the rest are skipped (never failed), regardless of goroutine scheduling.
func TestDispatchBudgetCallLimitSkipsRest(t *testing.T) {
	arm := newArmServer(t, http.StatusOK, 0)
	cfg := config.ShadowComparisonConfig{
		Enabled: true,
		Budget:  config.ShadowBudgetConfig{MaxCalls: 1, MaxConcurrency: 1, MaxTokens: 100},
		Arms: []config.ShadowArmConfig{
			{Name: "arm-1", Model: "model-a", Endpoint: arm.server.URL},
			{Name: "arm-2", Model: "model-b", Endpoint: arm.server.URL},
			{Name: "arm-3", Model: "model-c", Endpoint: arm.server.URL},
		},
	}
	results := Dispatch(context.Background(), cfg, testParams(), nil)

	var completed, skipped, failed int
	for _, res := range results {
		switch res.Outcome {
		case OutcomeCompleted:
			completed++
		case OutcomeSkipped:
			skipped++
		default:
			failed++
		}
	}
	if completed != 1 || skipped != 2 || failed != 0 {
		t.Fatalf("want 1 completed 2 skipped 0 failed, got completed=%d skipped=%d failed=%d (results=%+v)",
			completed, skipped, failed, results)
	}
}
