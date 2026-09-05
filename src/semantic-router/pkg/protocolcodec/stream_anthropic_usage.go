package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

func decodeAnthropicStreamUsage(wire anthropicUsageWire, initial bool) llmprotocol.Usage {
	usage := llmprotocol.Usage{
		State:           llmprotocol.UsageAvailable,
		InputCacheRead:  unknownCount(),
		InputCacheWrite: unknownCount(),
	}
	hasCacheTokens := wire.CacheReadInputTokens != nil && *wire.CacheReadInputTokens > 0 ||
		wire.CacheCreationInputTokens != nil && *wire.CacheCreationInputTokens > 0
	if initial || wire.InputTokens > 0 || hasCacheTokens {
		inputTotal := wire.InputTokens
		if wire.CacheReadInputTokens != nil {
			inputTotal += *wire.CacheReadInputTokens
		}
		if wire.CacheCreationInputTokens != nil {
			inputTotal += *wire.CacheCreationInputTokens
		}
		usage.InputUncached = authoritative(wire.InputTokens)
		usage.InputTotal = authoritative(inputTotal)
	}
	if wire.CacheReadInputTokens != nil {
		usage.InputCacheRead = optionalAuthoritative(wire.CacheReadInputTokens)
	}
	if wire.CacheCreationInputTokens != nil {
		usage.InputCacheWrite = optionalAuthoritative(wire.CacheCreationInputTokens)
	}
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
