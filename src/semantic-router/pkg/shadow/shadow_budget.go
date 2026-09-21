package shadow

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ShadowBudget is the per-request aggregate budget shared by every arm of one
// shadow_dispatch decision (issue #3376 multi-arm). Calls, reserved
// tokens/cost/response bytes and the in-flight concurrency are checked at
// admission; tokens, cost and the observed response size are accounted on arm
// completion. A zero field is unlimited.
// Wall time is bounded by the shared job dead-line: every arm of one request
// expires at the same instant, so the aggregate window needs no separate knob.
//
// Scoped to one shadow decision (issue #3376): this is not the cross-stage
// budget ledger owned by #2861, and it is not exported as a general contract.
type ShadowBudget struct {
	mu                   sync.Mutex
	limit                config.ShadowDispatchBudgetConfig
	reserveResponseBytes int64
	inflight             int64

	calls int64
	token int64
	cost  float64
	bytes int64
}

// NewShadowBudget builds a per-request budget from a decision's plugin config.
// reserveResponseBytes is the per-arm response bound reserved at admission for
// the aggregate byte cap.
func NewShadowBudget(limit config.ShadowDispatchBudgetConfig, reserveResponseBytes int64) *ShadowBudget {
	return &ShadowBudget{limit: limit, reserveResponseBytes: reserveResponseBytes}
}

// TryEnter admits one arm when every enforced dimension allows it. On
// rejection it returns a deterministic reason for the drop; the caller must
// not reconcile a rejected arm.
func (b *ShadowBudget) TryEnter() (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.limit.MaxCallsPerRequest > 0 && b.calls >= b.limit.MaxCallsPerRequest {
		return fmt.Sprintf("budget_call_limit (%d)", b.limit.MaxCallsPerRequest), false
	}
	reserveCost := b.costOf(b.limit.ReserveTokensPerArm)
	if b.limit.MaxTokensPerRequest > 0 && b.token+b.limit.ReserveTokensPerArm > b.limit.MaxTokensPerRequest {
		return fmt.Sprintf("budget_token_limit (%d)", b.limit.MaxTokensPerRequest), false
	}
	if b.limit.MaxCostPerRequest > 0 && b.cost+reserveCost > b.limit.MaxCostPerRequest {
		return fmt.Sprintf("budget_cost_limit (%v)", b.limit.MaxCostPerRequest), false
	}
	if b.limit.MaxResponseBytesPerRequest > 0 && b.bytes+b.reserveResponseBytes > b.limit.MaxResponseBytesPerRequest {
		return fmt.Sprintf("budget_response_bytes_limit (%d)", b.limit.MaxResponseBytesPerRequest), false
	}
	b.calls++
	b.token += b.limit.ReserveTokensPerArm
	b.cost += reserveCost
	b.bytes += b.reserveResponseBytes
	return "", true
}

// Refund returns one admission reservation for an arm that was admitted but
// never dispatched because the in-flight bound refused it.
func (b *ShadowBudget) Refund() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.calls > 0 {
		b.calls--
	}
	b.token -= b.limit.ReserveTokensPerArm
	b.cost -= b.costOf(b.limit.ReserveTokensPerArm)
	b.bytes -= b.reserveResponseBytes
}

// costOf converts tokens to cost at the configured price; 0 when unpriced.
func (b *ShadowBudget) costOf(tokens int64) float64 {
	if b.limit.PricePerMillionTokens <= 0 {
		return 0
	}
	return float64(tokens) / 1e6 * b.limit.PricePerMillionTokens
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

// Reconcile accounts a finished arm attempt. A completed arm swaps its token
// and cost admission reservations for the real usage; any other outcome keeps
// them (the attempt may still have spent upstream compute, and this was
// already counted at admission). Response bytes are always swapped for the
// observed size, so a read that never happened releases its reservation.
func (b *ShadowBudget) Reconcile(completed bool, inputTokens, outputTokens, responseBytes int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.bytes += responseBytes - b.reserveResponseBytes
	if !completed {
		return
	}
	used := inputTokens + outputTokens
	b.token += used - b.limit.ReserveTokensPerArm
	b.cost += b.costOf(used - b.limit.ReserveTokensPerArm)
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
