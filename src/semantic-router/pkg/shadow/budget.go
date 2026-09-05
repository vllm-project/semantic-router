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
// Hard limits (calls, concurrency) are reserved before an arm starts; soft
// limits (tokens, cost) are accounted on completion and enforced for
// subsequently admitted arms.
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

// tryEnter reserves one call and one concurrency slot for an arm when every
// enforced dimension allows it. On rejection it returns the deterministic
// skipped result; the caller must not invoke release for a rejected arm.
func (b *Budget) tryEnter(armName, model string) (ArmResult, bool) {
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
	if b.cfg.MaxTokens > 0 && b.usedTokens >= b.cfg.MaxTokens {
		return ArmResult{
			Arm: armName, Model: model, Outcome: OutcomeSkipped,
			Err: fmt.Sprintf("budget: token limit reached (%d)", b.cfg.MaxTokens),
		}, false
	}
	if b.cfg.MaxCost > 0 && b.usedCost >= b.cfg.MaxCost {
		return ArmResult{
			Arm: armName, Model: model, Outcome: OutcomeSkipped,
			Err: fmt.Sprintf("budget: cost limit reached (%v)", b.cfg.MaxCost),
		}, false
	}
	b.usedCalls++
	b.active++
	return ArmResult{}, true
}

// release frees one concurrency slot after an admitted arm finishes.
func (b *Budget) release() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.active > 0 {
		b.active--
	}
}

// reconcile accounts an arm outcome under the aggregate budget. Only completed
// arms consume tokens and cost; failed/timed_out/cancelled arms still consumed
// their admitted call slot (already counted by tryEnter).
func (b *Budget) reconcile(outcome Outcome, promptTokens, completionTokens int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if outcome != OutcomeCompleted {
		return
	}
	tokens := promptTokens + completionTokens
	b.usedTokens += tokens
	if b.cfg.PricePerMillionTokens > 0 {
		b.usedCost += float64(tokens) / 1e6 * b.cfg.PricePerMillionTokens
	}
}

// used reports the accounted totals (test/observability helper).
func (b *Budget) used() (calls, tokens int64, cost float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.usedCalls, b.usedTokens, b.usedCost
}
