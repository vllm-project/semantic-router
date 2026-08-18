/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
)

// BudgetExhaustionReason is a stable, typed reason code for why a
// BudgetLedger stopped granting further work. Values are a wire/metric
// contract for logging and replay: existing tokens must not be renamed once
// shipped, and any addition should be documented alongside these.
type BudgetExhaustionReason string

const (
	// BudgetReasonNone means the ledger is not exhausted.
	BudgetReasonNone BudgetExhaustionReason = ""
	// BudgetReasonPromptTokens means cumulative prompt tokens reached
	// ComputeBudget.MaxPromptTokens.
	BudgetReasonPromptTokens BudgetExhaustionReason = "prompt_tokens_exhausted" //nolint:gosec // enum value, not a credential
	// BudgetReasonCompletionTokens means cumulative completion tokens
	// reached ComputeBudget.MaxCompletionTokens.
	BudgetReasonCompletionTokens BudgetExhaustionReason = "completion_tokens_exhausted"
	// BudgetReasonTotalTokens means cumulative prompt+completion tokens
	// reached ComputeBudget.MaxTotalTokens.
	BudgetReasonTotalTokens BudgetExhaustionReason = "total_tokens_exhausted" //nolint:gosec // enum value, not a credential
	// BudgetReasonEstimatedCost means cumulative estimated cost reached
	// ComputeBudget.MaxEstimatedCost.
	BudgetReasonEstimatedCost BudgetExhaustionReason = "estimated_cost_exhausted"
	// BudgetReasonWallTime means the request's wall-clock deadline
	// (ComputeBudget.MaxWallTimeMs) has passed. BudgetLedger itself does not
	// set this reason - it is enforced separately via context.WithTimeout in
	// ExecuteWithLatency - but the token is reserved here so every budget
	// exhaustion reason lives in one enum.
	BudgetReasonWallTime BudgetExhaustionReason = "wall_time_exhausted"
)

// BudgetLedger is the mutable, concurrency-safe counterpart to
// ComputeBudget: it tracks what has actually been consumed by completed
// model calls during one Looper execution and reports whether the budget is
// exhausted. A single BudgetLedger is shared by every goroutine an algorithm
// spawns for one request (Fusion/ReMoM/Workflows dispatch calls
// concurrently), so all access goes through mu.
//
// A nil *BudgetLedger behaves as "no budget configured": every method is
// nil-safe and Exhausted always reports false.
type BudgetLedger struct {
	mu       sync.Mutex
	budget   *ComputeBudget
	consumed TokenUsage
	cost     float64
	reason   BudgetExhaustionReason
}

// NewBudgetLedger creates a BudgetLedger tracking consumption against
// budget. It returns nil when budget is nil or declares no limits, so
// callers can attach the result to Request.Ledger unconditionally and rely
// on nil-safety everywhere else.
func NewBudgetLedger(budget *ComputeBudget) *BudgetLedger {
	if budget.IsZero() {
		return nil
	}
	return &BudgetLedger{budget: budget}
}

// Exhausted reports whether the ledger's budget has been used up, and if so,
// which dimension tripped first. Nil-safe: a nil ledger is never exhausted.
func (l *BudgetLedger) Exhausted() (BudgetExhaustionReason, bool) {
	if l == nil {
		return BudgetReasonNone, false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.reason, l.reason != BudgetReasonNone
}

// Record adds one completed model call's usage and estimated cost to the
// ledger, and evaluates whether the budget is now exhausted. Nil-safe: a nil
// ledger discards the record.
func (l *BudgetLedger) Record(usage TokenUsage, cost float64) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	l.consumed.PromptTokens += usage.PromptTokens
	l.consumed.CompletionTokens += usage.CompletionTokens
	l.consumed.TotalTokens += usage.TotalTokens
	l.cost += cost

	if l.reason != BudgetReasonNone {
		return
	}
	l.reason = exhaustionReason(l.budget, l.consumed, l.cost)
}

func exhaustionReason(budget *ComputeBudget, consumed TokenUsage, cost float64) BudgetExhaustionReason {
	switch {
	case budget.MaxPromptTokens > 0 && consumed.PromptTokens >= budget.MaxPromptTokens:
		return BudgetReasonPromptTokens
	case budget.MaxCompletionTokens > 0 && consumed.CompletionTokens >= budget.MaxCompletionTokens:
		return BudgetReasonCompletionTokens
	case budget.MaxTotalTokens > 0 && consumed.TotalTokens >= budget.MaxTotalTokens:
		return BudgetReasonTotalTokens
	case budget.MaxEstimatedCost > 0 && cost >= budget.MaxEstimatedCost:
		return BudgetReasonEstimatedCost
	default:
		return BudgetReasonNone
	}
}

// CheckBudget is the pre-flight guard every algorithm calls before starting
// another model call, round, or step. It reports whether the caller should
// stop escalating and, if so, why. Nil-safe via Request.Ledger's own
// nil-safety: a Request with no configured budget always returns (false,
// BudgetReasonNone).
func CheckBudget(req *Request) (stop bool, reason BudgetExhaustionReason) {
	if req == nil {
		return false, BudgetReasonNone
	}
	reason, exhausted := req.Ledger.Exhausted()
	return exhausted, reason
}

// RecordBudgetUsage records one completed model call's usage against
// req.Ledger, estimating cost from req.ModelParams[modelName].Pricing via
// modelpricing.Cost. Nil-safe and a no-op when req, req.Ledger, or pricing
// for modelName is unavailable (missing pricing simply means cost is not
// tracked for that call; token/wall-time budgets are unaffected).
func RecordBudgetUsage(req *Request, usage TokenUsage, modelName string) {
	if req == nil || req.Ledger == nil {
		return
	}
	req.Ledger.Record(usage, estimateCost(req, usage, modelName))
}

// RecordBudgetUsageForResponse records one completed call's response against
// req.Ledger, for algorithm call sites that already hold a *ModelResponse
// (rather than separate usage/modelName values). Nil-safe: a nil resp, req,
// or req.Ledger is a no-op.
func RecordBudgetUsageForResponse(req *Request, resp *ModelResponse) {
	if req == nil || req.Ledger == nil || resp == nil {
		return
	}
	RecordBudgetUsage(req, resp.Usage, resp.Model)
}

// RecordBudgetUsageForResponses records every non-nil response in responses
// against req.Ledger. It is the shared entry point for algorithms (ReMoM,
// Fusion, Workflows) that gather a batch of concurrent-round responses
// rather than recording one call at a time; batches from result collectors
// are expected to already exclude failed/cancelled calls. Nil-safe and a
// no-op when req or req.Ledger is nil.
func RecordBudgetUsageForResponses(req *Request, responses []*ModelResponse) {
	for _, resp := range responses {
		RecordBudgetUsageForResponse(req, resp)
	}
}

func estimateCost(req *Request, usage TokenUsage, modelName string) float64 {
	if req.ModelParams == nil {
		return 0
	}
	params, ok := req.ModelParams[modelName]
	if !ok {
		return 0
	}
	return modelpricing.Cost(modelpricing.Usage{
		PromptTokens:     int(usage.PromptTokens),
		CompletionTokens: int(usage.CompletionTokens),
	}, pricingRates(params.Pricing))
}

// pricingRates adapts config.ModelPricing to modelpricing.Rates. Kept local
// to avoid importing pkg/extproc (which already imports pkg/looper) purely
// for its equivalent private helper.
func pricingRates(pricing config.ModelPricing) modelpricing.Rates {
	return modelpricing.Rates{
		Currency:         pricing.Currency,
		PromptPer1M:      pricing.PromptPer1M,
		CachedInputPer1M: pricing.CachedInputPer1M,
		CacheWritePer1M:  pricing.CacheWritePer1M,
		CompletionPer1M:  pricing.CompletionPer1M,
	}
}
