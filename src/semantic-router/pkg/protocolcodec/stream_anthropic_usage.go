package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

func decodeAnthropicStreamUsage(
	wire anthropicUsageWire,
	initial bool,
	policy llmprotocol.Policy,
	diagnostics *llmprotocol.Diagnostics,
) llmprotocol.Usage {
	usage := llmprotocol.Usage{
		State:           llmprotocol.UsageAvailable,
		InputCacheRead:  unknownCount(),
		InputCacheWrite: unknownCount(),
	}
	cacheWrite := wire.CacheCreationInputTokens
	cacheWriteAggregate := int64(0)
	if cacheWrite != nil {
		cacheWriteAggregate = *cacheWrite
	}
	cacheWrite5m, cacheWrite1h, derivedAggregate := decodeAnthropicCacheCreationTiers(
		wire.CacheCreation, cacheWriteAggregate, policy, diagnostics,
	)
	if derivedAggregate != nil {
		cacheWrite = derivedAggregate
	}
	hasCacheTokens := wire.CacheReadInputTokens != nil && *wire.CacheReadInputTokens > 0 ||
		cacheWrite != nil && *cacheWrite > 0
	if initial || wire.InputTokens > 0 || hasCacheTokens || wire.CacheCreation != nil {
		inputTotal := wire.InputTokens
		if wire.CacheReadInputTokens != nil {
			inputTotal += *wire.CacheReadInputTokens
		}
		if cacheWrite != nil {
			inputTotal += *cacheWrite
		}
		usage.InputUncached = authoritative(wire.InputTokens)
		usage.InputTotal = authoritative(inputTotal)
	}
	if wire.CacheReadInputTokens != nil {
		usage.InputCacheRead = optionalAuthoritative(wire.CacheReadInputTokens)
	}
	if cacheWrite != nil {
		usage.InputCacheWrite = optionalAuthoritative(cacheWrite)
	}
	usage.InputCacheWrite5m = cacheWrite5m
	usage.InputCacheWrite1h = cacheWrite1h
	if wire.OutputTokens > 0 || !initial {
		reasoning := wire.OutputTokensDetails.ThinkingTokens
		other := wire.OutputTokens - reasoning
		if other < 0 {
			other = 0
		}
		usage.OutputReasoning = authoritative(reasoning)
		usage.OutputOther = authoritative(other)
		usage.OutputTotal = authoritative(wire.OutputTokens)
	}
	return usage
}
