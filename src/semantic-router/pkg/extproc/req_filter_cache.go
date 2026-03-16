package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// handleCaching handles cache lookup and storage with category-specific settings
func (r *OpenAIRouter) handleCaching(ctx *RequestContext, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	if !r.extractCacheRequest(ctx) {
		return nil, false
	}

	if ctx.LooperRequest {
		return r.handleLooperCaching(ctx, categoryName)
	}

	if response, found := r.lookupCachedResponse(ctx, categoryName); found {
		return response, true
	}

	r.addPendingCacheRequest(ctx, categoryName)
	return nil, false
}

func (r *OpenAIRouter) extractCacheRequest(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}

	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery
	ctx.CacheQuery = cache.ScopeQueryToUser(requestQuery, extractUserID(ctx))
	return true
}

func (r *OpenAIRouter) handleLooperCaching(
	ctx *RequestContext,
	categoryName string,
) (*ext_proc.ProcessingResponse, bool) {
	logging.Debugf("[Cache] Skipping cache read for looper internal request")
	r.addPendingCacheRequest(ctx, categoryName)
	return nil, false
}

func (r *OpenAIRouter) lookupCachedResponse(
	ctx *RequestContext,
	categoryName string,
) (*ext_proc.ProcessingResponse, bool) {
	cacheQuery := cacheQueryForContext(ctx)
	if !r.shouldLookupSemanticCache(cacheQuery, categoryName) {
		return nil, false
	}

	threshold := r.cacheSimilarityThreshold(categoryName)
	logging.Infof("handleCaching: Performing cache lookup - model=%s, query_len=%d, threshold=%.2f",
		ctx.RequestModel, len(ctx.RequestQuery), threshold)

	spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, "semantic-cache", categoryName)
	startTime := time.Now()
	cachedResponse, found, cacheErr := r.Cache.FindSimilarWithThreshold(ctx.RequestModel, cacheQuery, threshold)
	lookupTime := time.Since(startTime).Milliseconds()

	logging.Infof("FindSimilarWithThreshold returned: found=%v, error=%v, lookupTime=%dms", found, cacheErr, lookupTime)
	tracing.SetSpanAttributes(span,
		attribute.String(tracing.AttrCacheKey, ctx.RequestQuery),
		attribute.Bool(tracing.AttrCacheHit, found),
		attribute.Int64(tracing.AttrCacheLookupTimeMs, lookupTime),
		attribute.String(tracing.AttrCategoryName, categoryName),
		attribute.Float64("cache.threshold", float64(threshold)))

	ctx.TraceContext = spanCtx
	if cacheErr != nil {
		logging.Errorf("Error searching cache: %v", cacheErr)
		tracing.RecordError(span, cacheErr)
		tracing.EndPluginSpan(span, "error", lookupTime, "lookup_failed")
		return nil, false
	}
	if !found {
		metrics.RecordCachePluginMiss(categoryName, "semantic-cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_miss")
		return nil, false
	}

	ctx.VSRCacheHit = true
	if categoryName != "" {
		ctx.VSRSelectedDecisionName = categoryName
	}

	metrics.RecordCachePluginHit(categoryName, "semantic-cache")
	tracing.EndPluginSpan(span, "success", lookupTime, "cache_hit")
	r.startRouterReplay(ctx, ctx.RequestModel, ctx.RequestModel, categoryName)
	logging.LogEvent("cache_hit", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      ctx.RequestModel,
		"query":      ctx.RequestQuery,
		"category":   categoryName,
		"threshold":  threshold,
	})

	response := http.CreateCacheHitResponse(cachedResponse, ctx.ExpectStreamingResponse, categoryName, ctx.VSRSelectedDecisionName, ctx.VSRMatchedKeywords)
	r.updateRouterReplayStatus(ctx, 200, ctx.ExpectStreamingResponse)
	r.attachRouterReplayResponse(ctx, cachedResponse, true)
	return response, true
}

func (r *OpenAIRouter) shouldLookupSemanticCache(requestQuery string, categoryName string) bool {
	cacheEnabled := r.isCacheEnabledForDecision(categoryName)
	logging.Infof("handleCaching: query_len=%d, cacheEnabled=%v, r.Cache.IsEnabled()=%v",
		len(requestQuery), cacheEnabled, r.Cache.IsEnabled())
	return requestQuery != "" && r.Cache.IsEnabled() && cacheEnabled
}

func (r *OpenAIRouter) addPendingCacheRequest(ctx *RequestContext, categoryName string) {
	cacheQuery := cacheQueryForContext(ctx)
	if !r.shouldLookupSemanticCache(cacheQuery, categoryName) {
		return
	}

	ttlSeconds := r.Config.GetCacheTTLSecondsForDecision(categoryName)
	err := r.Cache.AddPendingRequest(ctx.RequestID, ctx.RequestModel, cacheQuery, ctx.OriginalRequestBody, ttlSeconds)
	if err != nil {
		logging.Errorf("Error adding pending request to cache: %v", err)
	}
}

func (r *OpenAIRouter) isCacheEnabledForDecision(categoryName string) bool {
	cacheEnabled := r.Config.SemanticCache.Enabled
	if categoryName != "" {
		cacheEnabled = r.Config.IsCacheEnabledForDecision(categoryName)
	}
	return cacheEnabled
}

func (r *OpenAIRouter) cacheSimilarityThreshold(categoryName string) float32 {
	threshold := r.Config.GetCacheSimilarityThreshold()
	if categoryName != "" {
		threshold = r.Config.GetCacheSimilarityThresholdForDecision(categoryName)
	}
	return threshold
}

func cacheQueryForContext(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	if ctx.CacheQuery != "" {
		return ctx.CacheQuery
	}
	return ctx.RequestQuery
}
