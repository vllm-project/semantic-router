package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// takeNeutralResponseUsage consumes provider-reported usage from the response
// already decoded by the neutral protocol engine. Estimated request tokens are
// never substituted for missing terminal evidence.
func (r *OpenAIRouter) takeNeutralResponseUsage(ctx *RequestContext) responseUsageMetrics {
	if r == nil || ctx == nil || ctx.SemanticResponse == nil {
		return invalidResponseTerminalUsage("authoritative_usage_missing")
	}
	return responseUsageFromSemanticUsage(ctx.SemanticResponse.Usage)
}

func invalidResponseTerminalUsage(reason string) responseUsageMetrics {
	return responseUsageMetrics{invalid: true, invalidReason: reason}
}

func responseUsageFromSemanticUsage(usage llmprotocol.Usage) responseUsageMetrics {
	if usage.State != llmprotocol.UsageAvailable {
		return invalidResponseTerminalUsage("authoritative_usage_invalid")
	}
	input, inputReported, inputOK := authoritativeTerminalTokenCount(usage.InputTotal)
	output, outputReported, outputOK := authoritativeTerminalTokenCount(usage.OutputTotal)
	total, totalReported, totalOK := settlementTotalTokenCount(usage, input, output)
	cacheRead, cacheReadReported, cacheReadOK := authoritativeTerminalTokenCount(usage.InputCacheRead)
	cacheWrite, cacheWriteReported, cacheWriteOK := authoritativeTerminalTokenCount(usage.InputCacheWrite)
	if !inputOK || !outputOK || !totalOK || !cacheReadOK || !cacheWriteOK ||
		!inputReported || !outputReported {
		return invalidResponseTerminalUsage("response_terminal_invalid")
	}
	return normalizeResponseUsage(responseUsageMetrics{
		promptTokens: input, promptTokensReported: inputReported,
		completionTokens: output, completionTokensReported: outputReported,
		totalTokens: total, totalTokensReported: totalReported,
		cachedPromptTokens: cacheRead, cachedPromptTokensReported: cacheReadReported,
		cacheWriteTokens: cacheWrite, cacheWriteTokensReported: cacheWriteReported,
	})
}

func authoritativeTerminalTokenCount(count llmprotocol.TokenCount) (int, bool, bool) {
	if count.Value == nil {
		return 0, false, count.Provenance == "" || count.Provenance == llmprotocol.UsageUnknown
	}
	if count.Provenance != llmprotocol.UsageAuthoritative {
		return 0, true, false
	}
	value := *count.Value
	maximum := int64(^uint(0) >> 1)
	if value < 0 || value > maximum {
		return 0, true, false
	}
	return int(value), true, true
}

func settlementTotalTokenCount(usage llmprotocol.Usage, input, output int) (int, bool, bool) {
	if usage.Total.Value == nil {
		if input > int(^uint(0)>>1)-output {
			return 0, false, false
		}
		return input + output, true, true
	}
	value := *usage.Total.Value
	maximum := int64(^uint(0) >> 1)
	if value < 0 || value > maximum {
		return 0, true, false
	}
	switch usage.Total.Provenance {
	case llmprotocol.UsageAuthoritative:
		return int(value), true, true
	case llmprotocol.UsageDerived:
		if input > int(^uint(0)>>1)-output || int(value) != input+output {
			return 0, true, false
		}
		return int(value), true, true
	default:
		return 0, true, false
	}
}
