package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (r *OpenAIRouter) performExactCacheLookup(
	ctx *RequestContext,
	categoryName string,
	cacheEnabled bool,
) (*ext_proc.ProcessingResponse, bool) {
	if !cacheEnabled || r.Cache == nil || !r.Cache.IsEnabled() ||
		ctx == nil || ctx.CacheExactFingerprint == "" ||
		!exactCacheEnabledForRequest(ctx) {
		return nil, false
	}
	exactBackend, ok := r.Cache.(cache.ExactCacheBackend)
	if !ok {
		return nil, false
	}
	partition := semanticCachePartition(ctx, ctx.CacheRequestModel)
	start := time.Now()
	result, err := exactBackend.FindExact(partition, ctx.CacheExactFingerprint)
	lookupTime := time.Since(start).Milliseconds()
	if err != nil {
		logging.ComponentWarnEvent("extproc", "exact_cache_lookup_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"latency_ms": lookupTime,
			"error":      err.Error(),
		})
		return nil, false
	}
	if !result.Found {
		return nil, false
	}

	ctx.VSRCacheHit = true
	ctx.VSRCacheSimilarity = 1
	if categoryName != "" {
		ctx.VSRSelectedDecisionName = categoryName
	}
	metrics.RecordCachePluginHit(requestDecisionStateKey(ctx), "exact-cache")
	r.startRouterReplay(ctx, ctx.CacheRequestModel, ctx.CacheSelectedModel, categoryName)
	response := r.createCacheHitResponse(
		ctx,
		result.ResponseBody,
		"",
		ctx.VSRSelectedDecisionName,
		nil,
		0,
	)
	r.updateRouterReplayStatus(ctx, 200, ctx.ExpectStreamingResponse)
	r.attachRouterReplayResponse(ctx, result.ResponseBody, true)
	logging.ComponentDebugEvent("extproc", "exact_cache_hit", map[string]interface{}{
		"request_id": ctx.RequestID,
		"latency_ms": lookupTime,
	})
	return response, true
}

func (r *OpenAIRouter) updateExactResponseCache(
	ctx *RequestContext,
	responseBody []byte,
	ttlSeconds int,
) {
	if r == nil || r.Cache == nil || ctx == nil ||
		ctx.CacheExactFingerprint == "" || len(responseBody) == 0 ||
		!exactCacheEnabledForRequest(ctx) || ctx.CacheWriteBypass {
		return
	}
	exactBackend, ok := r.Cache.(cache.ExactCacheBackend)
	if !ok {
		return
	}
	partition := semanticCachePartition(ctx, ctx.CacheRequestModel)
	if err := exactBackend.AddExact(
		partition,
		ctx.CacheExactFingerprint,
		responseBody,
		ttlSeconds,
	); err != nil {
		logging.ComponentWarnEvent("extproc", "exact_cache_write_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      err.Error(),
		})
	}
}
