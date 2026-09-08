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
		if _, ok := b.tryEnter("a", "m", 0); !ok {
			t.Fatalf("arm %d should be admitted", i)
		}
	}
	res, ok := b.tryEnter("c", "m", 0)
	if ok {
		t.Fatal("third arm must be rejected")
	}
	if res.Outcome != OutcomeSkipped {
		t.Fatalf("want OutcomeSkipped, got %s", res.Outcome)
	}
}

func TestBudgetReleaseFreesSlot(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxConcurrency: 1})
	if _, ok := b.tryEnter("a", "m", 0); !ok {
		t.Fatal("first arm should be admitted")
	}
	if _, ok := b.tryEnter("b", "m", 0); ok {
		t.Fatal("second arm must be rejected under concurrency 1")
	}
	b.settle(OutcomeFailed, 0, 0, 0)
	if _, ok := b.tryEnter("b", "m", 0); !ok {
		t.Fatal("second arm must be admitted after release")
	}
}

func TestBudgetReconcileTokensAndCost(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{PricePerMillionTokens: 2.0})
	b.tryEnter("a", "m", 0)
	b.settle(OutcomeCompleted, 100, 50, 0) // 150 tokens
	_, tokens, cost := b.used()
	if tokens != 150 {
		t.Fatalf("tokens = %d, want 150", tokens)
	}
	if want := 150.0 / 1e6 * 2.0; cost != want {
		t.Fatalf("cost = %v, want %v", cost, want)
	}
	b.tryEnter("f", "m", 0)
	b.settle(OutcomeFailed, 500, 500, 0) // must not consume
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
	b.tryEnter("a", "m", 0)
	b.settle(OutcomeCompleted, 60, 60, 0) // 120 tokens > 100
	if _, ok := b.tryEnter("a", "m", 0); ok {
		t.Fatal("arm must be rejected after token soft limit exceeded")
	}
}

func TestBudgetCostSoftLimit(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxCost: 0.0002, PricePerMillionTokens: 2.0})
	b.tryEnter("a", "m", 0)
	b.settle(OutcomeCompleted, 200, 0, 0) // 200/1e6*2 = 0.0004 >= 0.0002
	if _, ok := b.tryEnter("a", "m", 0); ok {
		t.Fatal("arm must be rejected after cost soft limit exceeded")
	}
}

// TestBudgetReservationPreventsConcurrentOvershoot proves concurrent arms
// cannot collectively overshoot token/cost budgets: each admission reserves
// the request's declared output cap, so the limit check runs against the
// reserved total, not the (still-zero) accounted usage.
func TestBudgetReservationPreventsConcurrentOvershoot(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxTokens: 100, PricePerMillionTokens: 2.0})
	if _, ok := b.tryEnter("a", "m", 60); !ok {
		t.Fatal("first arm with reserve 60 must be admitted")
	}
	res, ok := b.tryEnter("b", "m", 60)
	if ok {
		t.Fatal("second arm with reserve 60 must be rejected under MaxTokens=100")
	}
	if res.Outcome != OutcomeSkipped {
		t.Fatalf("want OutcomeSkipped, got %s", res.Outcome)
	}
	if written, _, _ := b.used(); written != 1 {
		t.Fatalf("admitted calls = %d, want 1", written)
	}
}

// TestBudgetSettleSwapsReservationForActual proves a completed arm replaces
// its admission reservation with real usage (net), and a failed arm keeps the
// reservation so accounting never over-reports capacity.
func TestBudgetSettleSwapsReservationForActual(t *testing.T) {
	b := newBudget(config.ShadowBudgetConfig{MaxTokens: 150, PricePerMillionTokens: 2.0})
	b.tryEnter("a", "m", 60) // reserve 60
	b.settle(OutcomeCompleted, 30, 20, 60)
	_, tokens, cost := b.used() // 60 - 60 + (30+20) = 50
	if tokens != 50 {
		t.Fatalf("tokens = %d, want 50 (reservation swapped for actual)", tokens)
	}
	if want := 50.0 / 1e6 * 2.0; cost != want {
		t.Fatalf("cost = %v, want %v", cost, want)
	}
	b.tryEnter("b", "m", 60) // 50 + 60 = 110 <= 150, admitted
	b.settle(OutcomeFailed, 0, 0, 60)
	// A failed arm conservatively keeps its reservation (it did fire), so the
	// accounted total stays 50 (a) + 60 (b reserved) and never exceeds budget.
	if _, tokens, _ := b.used(); tokens != 110 {
		t.Fatalf("failed arm must keep reservation, tokens = %d, want 110", tokens)
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
