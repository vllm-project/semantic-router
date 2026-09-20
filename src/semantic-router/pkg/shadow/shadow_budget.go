package shadow

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ShadowBudget is the per-request aggregate budget shared by every arm of one
// shadow_dispatch decision (issue #3376 multi-arm). Calls, reserved tokens/cost
// and the in-flight concurrency are checked at admission; tokens, cost and
// response bytes are accounted on arm completion. A zero field is unlimited.
// Wall time is bounded by the shared job dead-line: every arm of one request
// expires at the same instant, so the aggregate window needs no separate knob.
type ShadowBudget struct {
	mu       sync.Mutex
	limit    config.ShadowDispatchBudgetConfig
	inflight int64

	calls int64
	token int64
	cost  float64
	bytes int64
}

// NewShadowBudget builds a per-request budget from a decision's plugin config.
func NewShadowBudget(limit config.ShadowDispatchBudgetConfig) *ShadowBudget {
	return &ShadowBudget{limit: limit}
}

// TryEnter admits one arm when every enforced dimension allows it. On
// rejection it returns a deterministic reason for the drop; the caller must
// not reconcile a rejected arm.
func (b *ShadowBudget) TryEnter(model string) (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.limit.MaxCallsPerRequest > 0 && b.calls >= b.limit.MaxCallsPerRequest {
		return fmt.Sprintf("budget_call_limit (%d)", b.limit.MaxCallsPerRequest), false
	}
	reserveCost := 0.0
	if b.limit.PricePerMillionTokens > 0 {
		reserveCost = float64(b.limit.ReserveTokensPerArm) / 1e6 * b.limit.PricePerMillionTokens
	}
	if b.limit.MaxTokensPerRequest > 0 && b.token+b.limit.ReserveTokensPerArm > b.limit.MaxTokensPerRequest {
		return fmt.Sprintf("budget_token_limit (%d)", b.limit.MaxTokensPerRequest), false
	}
	if b.limit.MaxCostPerRequest > 0 && b.cost+reserveCost > b.limit.MaxCostPerRequest {
		return fmt.Sprintf("budget_cost_limit (%v)", b.limit.MaxCostPerRequest), false
	}
	b.calls++
	b.token += b.limit.ReserveTokensPerArm
	b.cost += reserveCost
	return "", true
}

// EnterInflight acquires one of the per-request in-flight slots. The caller
// must LeaveInflight when the arm attempt ends. This is the concurrency
// dimension of the aggregate budget; the lane semaphore bounds the decision
// across requests instead.
func (b *ShadowBudget) EnterInflight() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.limit.MaxConcurrencyPerRequest > 0 && b.inflight >= b.limit.MaxConcurrencyPerRequest {
		return false
	}
	b.inflight++
	return true
}

// LeaveInflight releases a slot acquired by EnterInflight.
func (b *ShadowBudget) LeaveInflight() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.inflight > 0 {
		b.inflight--
	}
}

// Reconcile accounts a finished arm attempt. A completed arm swaps its
// admission reservation for the real usage; any other outcome keeps the
// reservation (the attempt may still have spent upstream compute, and this was
// already counted at admission). Observed response bytes are accounted for
// every outcome; the per-request upper bound of that dimension is the per-arm
// MaxResponseBytes times admitted calls.
func (b *ShadowBudget) Reconcile(completed bool, inputTokens, outputTokens, responseBytes int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if responseBytes > 0 {
		b.bytes += responseBytes
	}
	if !completed {
		return
	}
	used := inputTokens + outputTokens
	b.token += used - b.limit.ReserveTokensPerArm
	if b.limit.PricePerMillionTokens > 0 {
		b.cost += float64(used-b.limit.ReserveTokensPerArm) / 1e6 * b.limit.PricePerMillionTokens
	}
}

// Total reports the accounted totals (test/observability helper).
func (b *ShadowBudget) Total() (calls, tokens int64, cost float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls, b.token, b.cost
}

// TotalBytes reports the accounted response bytes (test/observability helper).
func (b *ShadowBudget) TotalBytes() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.bytes
}
