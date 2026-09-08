package shadow

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Outcome is the normalized lifecycle result of one shadow arm attempt,
// reconcilable into the aggregate budget and later persisted to Replay
// (issue #3376).
type Outcome string

const (
	OutcomeCompleted Outcome = "completed"
	OutcomeFailed    Outcome = "failed"
	OutcomeTimedOut  Outcome = "timed_out"
	OutcomeCancelled Outcome = "cancelled"
	// OutcomeSkipped marks an arm never admitted because an aggregate budget
	// limit was already reached (deterministic consume, insufficient_arms).
	OutcomeSkipped Outcome = "skipped"
)

// Budget is the per-request aggregate resource budget for shadow dispatch.
// Hard limits (calls, concurrency) are reserved before an arm starts; token
// and cost are also reserved at admission when the request declares an output
// cap (see outputReserve), so concurrent arms cannot collectively overshoot
// those limits. Tokens/cost without a declared cap are accounted on completion
// and enforced for subsequently admitted arms.
type Budget struct {
	mu         sync.Mutex
	cfg        config.ShadowBudgetConfig
	usedCalls  int64
	usedTokens int64
	usedCost   float64
	active     int
}

func newBudget(cfg config.ShadowBudgetConfig) *Budget {
	return &Budget{cfg: cfg}
}

// tryEnter reserves one call, one concurrency slotainer, and reserveTokens of
// token/cost headroom for an arm when every enforced dimension allows it. On
// rejection it returns the deterministic skipped result; the caller must not
// invoke settle for a rejected arm. Admission failure leaves any prior
// reservation untouched.
func (b *Budget) tryEnter(armName, model string, reserveTokens int64) (ArmResult, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cfg.MaxCalls > 0 && b.usedCalls >= b.cfg.MaxCalls {
		return ArmResult{
			Arm: armName, Model: model, Outcome: OutcomeSkipped,
			Err: fmt.Sprintf("budget: call limit reached (%d)", b.cfg.MaxCalls),
		}, false
	}
	if b.cfg.MaxConcurrency > 0 && b.active >= b.cfg.MaxConcurrency {
		return ArmResult{
			Arm: armName, Model: model, Outcome: OutcomeSkipped,
			Err: fmt.Sprintf("budget: concurrency limit reached (%d)", b.cfg.MaxConcurrency),
		}, false
	}
	reserveCost := 0.0
	if b.cfg.PricePerMillionTokens > 0 {
		reserveCost = float64(reserveTokens) / 1e6 * b.cfg.PricePerMillionTokens
	}
	if b.cfg.MaxTokens > 0 && b.usedTokens+reserveTokens > b.cfg.MaxTokens {
		return ArmResult{
			Arm: armName, Model: model, Outcome: OutcomeSkipped,
			Err: fmt.Sprintf("budget: token limit reached (%d)", b.cfg.MaxTokens),
		}, false
	}
	if b.cfg.MaxCost > 0 && b.usedCost+reserveCost > b.cfg.MaxCost {
		return ArmResult{
			Arm: armName, Model: model, Outcome: OutcomeSkipped,
			Err: fmt.Sprintf("budget: cost limit reached (%v)", b.cfg.MaxCost),
		}, false
	}
	b.usedCalls++
	b.active++
	b.usedTokens += reserveTokens
	b.usedCost += reserveCost
	return ArmResult{}, true
}

// settle frees one concurrency slot after an admitted arm finishes and swaps
// the admission-time token/cost reservation for the accounted reality: a
// completed arm records its actual usage (net of the reservation); any other
// outcome keeps the reservation (the arm did fire and conservatively consumed
// budget), so aggregate accounting never exceeds reserved + actual totals.
func (b *Budget) settle(outcome Outcome, promptTokens, completionTokens, reserveTokens int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.active > 0 {
		b.active--
	}
	if outcome != OutcomeCompleted {
		return
	}
	tokens := promptTokens + completionTokens
	b.usedTokens += tokens - reserveTokens
	if b.cfg.PricePerMillionTokens > 0 {
		b.usedCost += float64(tokens-reserveTokens) / 1e6 * b.cfg.PricePerMillionTokens
	}
}

// used reports the accounted totals (test/observability helper).
func (b *Budget) used() (calls, tokens int64, cost float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.usedCalls, b.usedTokens, b.usedCost
}
