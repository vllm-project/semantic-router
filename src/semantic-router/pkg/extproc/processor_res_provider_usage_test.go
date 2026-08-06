package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

func TestMergeProviderResponseUsageIncludesAnthropicCacheTokens(t *testing.T) {
	ctx := &RequestContext{IRExtensions: &ir.IRExtensions{
		CacheReadInputTokens:     30,
		CacheCreationInputTokens: 20,
	}}

	usage := mergeProviderResponseUsage(ctx, responseUsageMetrics{
		promptTokens:     50,
		completionTokens: 10,
	})

	if usage.promptTokens != 100 ||
		usage.cachedPromptTokens != 30 ||
		usage.cacheWriteTokens != 20 ||
		!usage.cachedPromptTokensReported ||
		!usage.cacheWriteTokensReported {
		t.Fatalf("unexpected provider usage: %#v", usage)
	}
}

func TestExtractStreamingUsageIncludesAnthropicCacheTokens(t *testing.T) {
	ctx := &RequestContext{
		IRExtensions: &ir.IRExtensions{
			CacheReadInputTokens:     30,
			CacheCreationInputTokens: 20,
		},
		StreamingMetadata: map[string]interface{}{
			"usage": map[string]interface{}{
				"prompt_tokens":     float64(50),
				"completion_tokens": float64(10),
				"total_tokens":      float64(60),
			},
		},
	}

	usage := extractStreamingUsage(ctx)
	cached, cachedReported, cacheWrite, cacheWriteReported := streamingPromptTokenDetails(
		ctx,
		int(usage.PromptTokens),
	)

	if usage.PromptTokens != 100 || usage.TotalTokens != 110 {
		t.Fatalf("unexpected streaming totals: %#v", usage)
	}
	if cached != 30 || cacheWrite != 20 || !cachedReported || !cacheWriteReported {
		t.Fatalf(
			"unexpected streaming cache details: cached=%d/%v write=%d/%v",
			cached,
			cachedReported,
			cacheWrite,
			cacheWriteReported,
		)
	}
}
