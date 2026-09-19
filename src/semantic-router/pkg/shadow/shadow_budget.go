package shadow

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ShadowBudget is the per-request aggregate budget shared by every arm of one
// shadow_dispatch decision (issue #3376 multi-arm). Hard limits (calls,
// reserved tokens/cost) are checked at admission; accounted tokens/cost are
// reconciled on arm completion. A zero field is unlimited.
type ShadowBudget struct {
	mu      sync.Mutex
	limit   config.ShadowDispatchBudgetConfig
	reserve int64

	calls int64
	token int64
	cost  float64
}

// NewShadowBudget builds a per-request budget from a decision's plugin config.
func NewShadowBudget(cfg config.ShadowDispatchPluginConfig) *ShadowBudget {
	return &ShadowBudget{
		limit:   cfg.Budget,
		reserve: cfg.Budget.ReserveTokensPerArm,
	}
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
		reserveCost = float64(b.reserve) / 1e6 * b.limit.PricePerMillionTokens
	}
	if b.limit.MaxTokensPerRequest > 0 && b.token+b.reserve > b.limit.MaxTokensPerRequest {
		return fmt.Sprintf("budget_token_limit (%d)", b.limit.MaxTokensPerRequest), false
	}
	if b.limit.MaxCostPerRequest > 0 && b.cost+reserveCost > b.limit.MaxCostPerRequest {
		return fmt.Sprintf("budget_cost_limit (%v)", b.limit.MaxCostPerRequest), false
	}
	b.calls++
	b.token += b.reserve
	b.cost += reserveCost
	return "", true
}

// Reconcile accounts a finished arm. Completed arms swap their admission
// reservation for the real usage; any other outcome conservatively keeps the
// reservation, so aggregate accounting never exceeds reserved + actual.
func (b *ShadowBudget) Reconcile(completed bool, inputTokens, outputTokens int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !completed {
		return
	}
	used := inputTokens + outputTokens
	b.token += used - b.reserve
	if b.limit.PricePerMillionTokens > 0 {
		b.cost += float64(used-b.reserve) / 1e6 * b.limit.PricePerMillionTokens
	}
}

// Total reports the accounted totals (test/observability helper).
func (b *ShadowBudget) Total() (calls, tokens int64, cost float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls, b.token, b.cost
}
