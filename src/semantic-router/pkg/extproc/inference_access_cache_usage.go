package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"

// recordCacheHitSettlementUsage retains only normalized authoritative usage
// from the cached neutral response. Backend evidence remains known-zero; this
// record describes tokens served to the caller and never fabricates cost.
func recordCacheHitSettlementUsage(ctx *RequestContext, usage responseUsageMetrics) {
	if ctx == nil || ctx.InferenceAccess == nil {
		return
	}
	recorded := usageFromResponse(usage)
	state := ctx.InferenceAccess
	state.mu.Lock()
	defer state.mu.Unlock()
	state.cacheHitServedUsage = nil
	if recorded.State != usageaccounting.EvidenceKnownActual {
		return
	}
	served := recorded.Usage
	state.cacheHitServedUsage = &served
}

func cacheHitSettlementUsage(ctx *RequestContext) *usageaccounting.ActualUsage {
	if ctx == nil || !ctx.VSRCacheHit || ctx.InferenceAccess == nil {
		return nil
	}
	state := ctx.InferenceAccess
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.cacheHitServedUsage == nil {
		return nil
	}
	served := *state.cacheHitServedUsage
	return &served
}
