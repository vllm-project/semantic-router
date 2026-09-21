package shadow

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestShadowBudgetCallLimit(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxCallsPerRequest: 1}, 0)
	if _, ok := b.TryEnter(); !ok { // "a" is the model name, opaque to the budget
		t.Fatal("first arm must be admitted")
	}
	if reason, ok := b.TryEnter(); ok {
		t.Fatalf("second arm must be rejected, got admitted")
	} else if reason != "budget_call_limit" {
		t.Fatalf("reason = %q, want budget_call_limit", reason)
	}
}

func TestShadowBudgetDropReasonsStayValueFree(t *testing.T) {
	for _, reason := range []string{
		droppReasonCallLimit, droppReasonTokenLimit, droppReasonCostLimit,
		droppReasonConcurrencyLimit, droppReasonResponseBytesLimit,
	} {
		if !strings.HasPrefix(reason, "budget_") || strings.Contains(reason, "(") {
			t.Fatalf("drop reason %q must be a fixed label: it tags a metric reason", reason)
		}
	}
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxResponseBytesPerRequest: 700}, 400)
	b.TryEnter()
	if reason, ok := b.TryEnter(); ok || strings.Contains(reason, "(") {
		t.Fatalf("drop reason %q must carry no per-request value", reason)
	}
}

func TestShadowBudgetReservesTokensAtAdmission(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxTokensPerRequest: 100, ReserveTokensPerArm: 60}, 0)
	if _, ok := b.TryEnter(); !ok {
		t.Fatal("first arm with reserve 60 must be admitted")
	}
	if reason, ok := b.TryEnter(); ok {
		t.Fatalf("second arm with reserve 60 must be rejected under MaxTokens=100")
	} else if reason != "budget_token_limit" {
		t.Fatalf("reason = %q, want budget_token_limit", reason)
	}
}

func TestShadowBudgetReservesResponseBytesAtAdmission(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxResponseBytesPerRequest: 700}, 400)
	if _, ok := b.TryEnter(); !ok {
		t.Fatal("first arm with a 400-byte reservation must be admitted under 700")
	}
	if reason, ok := b.TryEnter(); ok {
		t.Fatalf("second arm must be rejected: 400+400 > 700")
	} else if reason != "budget_response_bytes_limit" {
		t.Fatalf("reason = %q, want budget_response_bytes_limit", reason)
	}
}

func TestShadowBudgetReconcileSwapsReservation(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxTokensPerRequest: 200, PricePerMillionTokens: 2.0, ReserveTokensPerArm: 60}, 0)
	b.TryEnter()
	b.TryEnter()
	b.Reconcile(true, 20, 10, 512) // a: 60 - 60 + 30 = 30
	b.Reconcile(false, 0, 0, 128)  // b: failed keeps reservation, total 30 + 60 = 90
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
	// Response bytes are accounted for every outcome, not only completions.
	if bytes := b.TotalBytes(); bytes != 640 {
		t.Fatalf("bytes = %d, want 640", bytes)
	}
}

func TestShadowBudgetReconcileSwapsResponseBytesReservation(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{}, 400)
	b.TryEnter()
	if bytes := b.TotalBytes(); bytes != 400 {
		t.Fatalf("bytes = %d, want 400 reserved at admission", bytes)
	}
	b.Reconcile(true, 0, 0, 128)
	if bytes := b.TotalBytes(); bytes != 128 {
		t.Fatalf("bytes = %d, want 128 (reservation swapped for the observed size)", bytes)
	}
	// An arm that reported no read at all releases its reservation.
	b.TryEnter()
	b.Reconcile(false, 0, 0, 0)
	if bytes := b.TotalBytes(); bytes != 128 {
		t.Fatalf("bytes = %d, want 128 after a zero-byte outcome", bytes)
	}
}

func TestShadowBudgetCostLimit(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxCostPerRequest: 0.0001, PricePerMillionTokens: 2.0, ReserveTokensPerArm: 100}, 0)
	if _, ok := b.TryEnter(); ok {
		t.Fatal("first arm reserve 100 = 0.0002 > 0.0001 should be rejected immediately")
	}
}

func TestShadowBudgetNoReserveAccountsOnCompletion(t *testing.T) {
	// The admission reservation is the dimension that binds early; without it
	// the token cap can only be observed on completion. Configurations that
	// need the cap to bind at admission are rejected at load time by
	// ShadowDispatchPluginConfig.Validate.
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxTokensPerRequest: 100}, 0)
	if _, ok := b.TryEnter(); !ok {
		t.Fatal("first arm must be admitted")
	}
	if _, ok := b.TryEnter(); !ok {
		t.Fatal("second arm must be admitted when no reserve is set")
	}
	b.Reconcile(true, 60, 60, 0) // a: 120 tokens
	_, tokens, _ := b.Total()
	if tokens != 120 {
		t.Fatalf("tokens = %d, want 120", tokens)
	}
}

func TestShadowBudgetRefundReturnsAdmission(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxTokensPerRequest: 200, PricePerMillionTokens: 2.0, ReserveTokensPerArm: 60}, 400)
	if _, ok := b.TryEnter(); !ok {
		t.Fatal("arm must be admitted")
	}
	b.Refund()
	calls, tokens, cost := b.Total()
	if calls != 0 || tokens != 0 || cost != 0 {
		t.Fatalf("after Refund calls=%d tokens=%d cost=%v, want 0/0/0", calls, tokens, cost)
	}
	if bytes := b.TotalBytes(); bytes != 0 {
		t.Fatalf("after Refund bytes = %d, want 0 (the byte reservation is released too)", bytes)
	}
}

func TestShadowBudgetConcurrencyPerRequest(t *testing.T) {
	b := NewShadowBudget(config.ShadowDispatchBudgetConfig{MaxConcurrencyPerRequest: 1}, 0)
	if !b.EnterInflight() {
		t.Fatal("first in-flight slot must be granted")
	}
	if b.EnterInflight() {
		t.Fatal("second in-flight slot must be refused under MaxConcurrencyPerRequest=1")
	}
	b.LeaveInflight()
	if !b.EnterInflight() {
		t.Fatal("slot must be reusable after LeaveInflight")
	}
	b.LeaveInflight()

	unbounded := NewShadowBudget(config.ShadowDispatchBudgetConfig{}, 0)
	for i := range 3 {
		if !unbounded.EnterInflight() {
			t.Fatalf("unbounded budget refused in-flight slot %d", i)
		}
	}
}
