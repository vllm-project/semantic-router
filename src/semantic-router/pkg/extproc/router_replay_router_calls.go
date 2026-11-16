package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// looperRouterCalls costs the calls a Looper request made, the model calls
// the Router made itself to answer it. The priced attempts in the first
// priced attempt's currency are summed. An attempt without rates or usage, in
// another currency, or that no trace kept counts as unpriced, never free.
// calls is how many calls the algorithm reports, for one that traces none.
// The tokens are the request's, which already sum every call.
func looperRouterCalls(diagnostics *routerreplay.LooperDiagnostics, usage looper.TokenUsage, calls int) store.RouterCallCost {
	call := store.RouterCallCost{
		Kind:             store.RouterCallKindLooper,
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
	}
	if diagnostics == nil || len(diagnostics.Attempts)+diagnostics.DroppedAttemptCount == 0 {
		// Only some algorithms trace their attempts. Without a trace no call
		// has a price; at least one call was made.
		addUnpricedRouterCalls(&call, store.UnpricedAttemptNotTraced, max(calls, 1))
		return call
	}
	cost := 0.0
	for _, attempt := range diagnostics.Attempts {
		if reason := looperAttemptUnpricedReason(attempt, call.Currency); reason != "" {
			addUnpricedRouterCalls(&call, reason, 1)
			continue
		}
		call.Calls++
		call.Currency = normalizeReplayCurrency(attempt.Currency)
		cost += *attempt.ActualCost
	}
	if call.Currency != "" {
		call.Cost = replayFloat64Ptr(cost)
	}
	if diagnostics.DroppedAttemptCount > 0 {
		addUnpricedRouterCalls(&call, store.UnpricedAttemptNotTraced, diagnostics.DroppedAttemptCount)
	}
	return call
}

func looperAttemptUnpricedReason(attempt routerreplay.LooperAttempt, currency string) string {
	switch {
	case attempt.Currency == "":
		return store.UnpricedPricingNotConfigured
	case attempt.ActualCost == nil && attempt.Usage == (routerreplay.LooperUsage{}):
		return store.UnpricedNoUsage
	case !finiteReplayCost(attempt.ActualCost):
		return store.UnpricedUsageIncomplete
	case currency != "" && normalizeReplayCurrency(attempt.Currency) != currency:
		return store.UnpricedCurrencyMismatch
	}
	return ""
}

func addUnpricedRouterCalls(call *store.RouterCallCost, reason string, count int) {
	call.Calls += count
	call.UnpricedCalls += count
	if call.UnpricedReasons == nil {
		call.UnpricedReasons = make(map[string]int)
	}
	call.UnpricedReasons[reason] += count
}
