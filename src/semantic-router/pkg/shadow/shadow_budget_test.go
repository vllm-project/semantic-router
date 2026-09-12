package shadow

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestShadowBudgetCallLimit(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchPluginConfig{Budget: config.ShadowDispatchBudgetConfig{MaxCallsPerRequest: 1}})
	if _, ok := b.TryEnter("a"); !ok {
		t.Fatal("first arm must be admitted")
	}
	if reason, ok := b.TryEnter("b"); ok {
		t.Fatalf("second arm must be rejected, got admitted")
	} else if reason != "budget_call_limit (1)" {
		t.Fatalf("reason = %q, want budget_call_limit (1)", reason)
	}
}

func TestShadowBudgetReservesTokensAtAdmission(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchPluginConfig{
		Budget: config.ShadowDispatchBudgetConfig{MaxTokensPerRequest: 100, ReserveTokensPerArm: 60},
	})
	if _, ok := b.TryEnter("a"); !ok {
		t.Fatal("first arm with reserve 60 must be admitted")
	}
	if reason, ok := b.TryEnter("b"); ok {
		t.Fatalf("second arm with reserve 60 must be rejected under MaxTokens=100")
	} else if !strings.HasPrefix(reason, "budget_token_limit") {
		t.Fatalf("reason = %q, want budget_token_limit", reason)
	}
}

func TestShadowBudgetReconcileSwapsReservation(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchPluginConfig{
		Budget: config.ShadowDispatchBudgetConfig{MaxTokensPerRequest: 200, PricePerMillionTokens: 2.0, ReserveTokensPerArm: 60},
	})
	b.TryEnter("a")
	b.TryEnter("b")
	b.Reconcile(true, 20, 10) // a: 60 - 60 + 30 = 30
	b.Reconcile(false, 0, 0)  // b: failed keeps reservation, total 30 + 60 = 90
	calls, tokens, cost := b.Total()
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
	if tokens != 90 {
		t.Fatalf("tokens = %d, want 90 (30 accounted + 60 failed reservation)", tokens)
	}
	// Cost follows the same accounting as tokens (reservation is reserved at
	// admission, net-adjusted on completion), so it reflects 90 accounted
	// tokens at the configured price.
	if want := 90.0 / 1e6 * 2.0; cost != want {
		t.Fatalf("cost = %v, want %v (cost reflects accounted tokens)", cost, want)
	}
}

func TestShadowBudgetCostLimit(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchPluginConfig{
		Budget: config.ShadowDispatchBudgetConfig{MaxCostPerRequest: 0.0001, PricePerMillionTokens: 2.0, ReserveTokensPerArm: 100},
	})
	if _, ok := b.TryEnter("a"); ok {
		t.Fatal("first arm reserve 100 = 0.0002 > 0.0001 should be rejected immediately")
	}
}

func TestShadowBudgetNoReserveAccountsOnCompletion(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchPluginConfig{
		Budget: config.ShadowDispatchBudgetConfig{MaxTokensPerRequest: 100},
	})
	// No reserve: both arms admit concurrently; accounting reconciles later.
	if _, ok := b.TryEnter("a"); !ok {
		t.Fatal("first arm must be admitted")
	}
	if _, ok := b.TryEnter("b"); !ok {
		t.Fatal("second arm must be admitted when no reserve is set")
	}
	b.Reconcile(true, 60, 60) // a: 120 tokens
	_, tokens, _ := b.Total()
	if tokens != 120 {
		t.Fatalf("tokens = %d, want 120", tokens)
	}
}
