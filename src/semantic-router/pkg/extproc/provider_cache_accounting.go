package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func buildProviderCacheAccounting(ctx *RequestContext, pricing *config.ModelPricing) *store.ProviderCacheAccounting {
	if ctx == nil || ctx.SemanticResponse == nil {
		return nil
	}
	usage := ctx.SemanticResponse.Usage
	result := &store.ProviderCacheAccounting{State: "partial"}
	fields := []struct {
		count     llmprotocol.TokenCount
		component *store.TokenCostComponent
	}{
		{usage.InputUncached, &result.UncachedInput},
		{usage.InputCacheRead, &result.CacheRead},
		{usage.InputCacheWrite, &result.CacheWrite},
		{usage.OutputTotal, &result.Output},
		{usage.InputCacheWrite5m, &result.CacheWrite5m},
		{usage.InputCacheWrite1h, &result.CacheWrite1h},
	}
	for _, field := range fields {
		field.component.Provenance = string(field.count.Provenance)
		if field.count.Value == nil {
			field.component.Provenance = string(llmprotocol.UsageUnknown)
			continue
		}
		if *field.count.Value >= 0 && *field.count.Value <= int64(^uint(0)>>1) {
			field.component.Tokens = replayIntPtr(int(*field.count.Value))
		}
	}
	if usage.State == llmprotocol.UsageUnavailable {
		result.State = "unavailable"
		return result
	}
	if err := llmprotocol.ValidateUsage(usage); err != nil {
		result.State = "invalid"
		return result
	}
	inputKnown := usage.InputTotal.Value != nil &&
		(usage.InputTotal.Provenance == llmprotocol.UsageAuthoritative || usage.InputTotal.Provenance == llmprotocol.UsageDerived)
	if result.UncachedInput.Tokens == nil && inputKnown &&
		priceableProviderComponent(result.CacheRead) && priceableProviderComponent(result.CacheWrite) {
		uncached := *usage.InputTotal.Value - int64(*result.CacheRead.Tokens) - int64(*result.CacheWrite.Tokens)
		result.UncachedInput = store.TokenCostComponent{Tokens: replayIntPtr(int(uncached)), Provenance: string(llmprotocol.UsageDerived)}
	}
	complete := priceableProviderComponent(result.UncachedInput) && priceableProviderComponent(result.CacheRead) &&
		priceableProviderComponent(result.CacheWrite) && priceableProviderComponent(result.Output)
	for _, tier := range []store.TokenCostComponent{result.CacheWrite5m, result.CacheWrite1h} {
		if tier.Tokens != nil && !priceableProviderComponent(tier) {
			complete = false
		}
	}
	if complete {
		result.State = "complete"
	}
	if pricing == nil {
		return result
	}
	rates := modelPricingRates(*pricing)
	for _, field := range []struct {
		component *store.TokenCostComponent
		rate      float64
	}{
		{&result.UncachedInput, rates.PromptPer1M},
		{&result.CacheRead, rates.CachedInputPer1M},
		{&result.Output, rates.CompletionPer1M},
		{&result.CacheWrite5m, rates.EffectiveCacheWritePer1M()},
		{&result.CacheWrite1h, rates.EffectiveCacheWrite1hPer1M()},
	} {
		if priceableProviderComponent(*field.component) {
			field.component.Cost = replayFloat64Ptr(float64(*field.component.Tokens) * field.rate / 1_000_000)
		}
	}
	if priceableProviderComponent(result.CacheWrite) {
		writes := modelpricing.Usage{PromptTokens: *result.CacheWrite.Tokens, CacheWriteTokens: *result.CacheWrite.Tokens}
		if priceableProviderComponent(result.CacheWrite5m) {
			writes.CacheWrite5mTokens = *result.CacheWrite5m.Tokens
		}
		if priceableProviderComponent(result.CacheWrite1h) {
			writes.CacheWrite1hTokens = *result.CacheWrite1h.Tokens
		}
		result.CacheWrite.Cost = replayFloat64Ptr(modelpricing.Costs(writes, rates).CacheWrite)
	}
	if !complete {
		return result
	}
	input := *result.UncachedInput.Tokens + *result.CacheRead.Tokens + *result.CacheWrite.Tokens
	noCache := modelpricing.Cost(modelpricing.Usage{PromptTokens: input, CompletionTokens: *result.Output.Tokens}, rates)
	observed := modelpricing.Usage{
		PromptTokens: input, CachedInputTokens: *result.CacheRead.Tokens,
		CacheWriteTokens: *result.CacheWrite.Tokens, CompletionTokens: *result.Output.Tokens,
	}
	if priceableProviderComponent(result.CacheWrite5m) {
		observed.CacheWrite5mTokens = *result.CacheWrite5m.Tokens
	}
	if priceableProviderComponent(result.CacheWrite1h) {
		observed.CacheWrite1hTokens = *result.CacheWrite1h.Tokens
	}
	actual := modelpricing.Cost(observed, rates)
	result.NoCacheCost = replayFloat64Ptr(noCache)
	result.CacheSavings = replayFloat64Ptr(noCache - actual)
	return result
}

func priceableProviderComponent(component store.TokenCostComponent) bool {
	return component.Tokens != nil &&
		(component.Provenance == string(llmprotocol.UsageAuthoritative) || component.Provenance == string(llmprotocol.UsageDerived))
}
