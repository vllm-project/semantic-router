package extproc

import "github.com/openai/openai-go"

func mergeProviderResponseUsage(
	ctx *RequestContext,
	usage responseUsageMetrics,
) responseUsageMetrics {
	cacheRead, cacheWrite := providerCacheUsageFromExtensions(ctx)
	if cacheRead > 0 && !usage.cachedPromptTokensReported {
		usage.promptTokens += cacheRead
		usage.cachedPromptTokens = cacheRead
		usage.cachedPromptTokensReported = true
	}
	if cacheWrite > 0 && !usage.cacheWriteTokensReported {
		usage.promptTokens += cacheWrite
		usage.cacheWriteTokens = cacheWrite
		usage.cacheWriteTokensReported = true
	}
	return normalizeResponseUsage(usage)
}

func mergeProviderStreamingUsage(
	ctx *RequestContext,
	usage openai.CompletionUsage,
) openai.CompletionUsage {
	cacheRead, cacheWrite := providerCacheUsageFromExtensions(ctx)
	cacheReadReported, cacheWriteReported := streamingCacheDetailsReported(ctx)
	if cacheRead > 0 && !cacheReadReported {
		usage.PromptTokens += int64(cacheRead)
		usage.TotalTokens += int64(cacheRead)
	}
	if cacheWrite > 0 && !cacheWriteReported {
		usage.PromptTokens += int64(cacheWrite)
		usage.TotalTokens += int64(cacheWrite)
	}
	return usage
}

func streamingCacheDetailsReported(ctx *RequestContext) (cacheRead bool, cacheWrite bool) {
	if ctx == nil || ctx.StreamingMetadata == nil {
		return false, false
	}
	usageMap, ok := ctx.StreamingMetadata["usage"].(map[string]interface{})
	if !ok {
		return false, false
	}
	for _, key := range []string{"prompt_tokens_details", "input_tokens_details"} {
		details, ok := usageMap[key].(map[string]interface{})
		if !ok {
			continue
		}
		_, hasCacheRead := details["cached_tokens"]
		_, hasCacheWrite := details["cache_write_tokens"]
		cacheRead = cacheRead || hasCacheRead
		cacheWrite = cacheWrite || hasCacheWrite
	}
	return cacheRead, cacheWrite
}

func providerCacheUsageFromExtensions(ctx *RequestContext) (cacheRead int, cacheWrite int) {
	if ctx == nil || ctx.IRExtensions == nil {
		return 0, 0
	}
	return int(ctx.IRExtensions.CacheReadInputTokens),
		int(ctx.IRExtensions.CacheCreationInputTokens)
}

func mergeProviderStreamingTokenDetails(
	ctx *RequestContext,
	cached int,
	cachedReported bool,
	cacheWrite int,
	cacheWriteReported bool,
) (int, bool, int, bool) {
	cacheRead, extensionCacheWrite := providerCacheUsageFromExtensions(ctx)
	if !cachedReported && cacheRead > 0 {
		cached = cacheRead
		cachedReported = true
	}
	if !cacheWriteReported && extensionCacheWrite > 0 {
		cacheWrite = extensionCacheWrite
		cacheWriteReported = true
	}
	return cached, cachedReported, cacheWrite, cacheWriteReported
}
