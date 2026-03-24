package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// decisionWillPersonalize checks whether the matched decision is configured
// with plugins (RAG, memory) that inject user-specific context. When true,
// we skip the entire cache path — both reads and writes — because:
//   - reads would serve a generic cached answer instead of the personalized one
//   - writes would cache a personalized answer that could leak to other users
//
// This avoids orphaned pending cache entries and unnecessary embedding work.
func decisionWillPersonalize(ctx *RequestContext, cfg *config.RouterConfig) bool {
	d := ctx.VSRSelectedDecision
	if d == nil {
		return false
	}
	if ragCfg := d.GetRAGConfig(); ragCfg != nil && ragCfg.Enabled {
		return true
	}
	if memCfg := d.GetMemoryConfig(); memCfg != nil && memCfg.Enabled {
		return true
	}
	if cfg != nil && cfg.Memory.Enabled {
		return true
	}
	return false
}

// handleCaching handles cache lookup and storage with category-specific settings
func (r *OpenAIRouter) handleCaching(ctx *RequestContext, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	// Skip entire cache path for decisions that will inject user-specific context.
	// Both reads (would serve stale generic answers) and writes (would leak
	// personalized data) are wrong when RAG or memory is enabled.
	if decisionWillPersonalize(ctx, r.Config) {
		logging.Debugf("[Cache] Skipping cache for decision '%s': RAG or memory enabled", categoryName)
		return nil, false
	}

	// Skip cache read for looper internal requests
	// Looper requests should not return cached responses, but should still write to cache
	if ctx.LooperRequest {
		logging.Debugf("[Cache] Skipping cache read for looper internal request")
		// Still extract model and query for potential cache write later
		requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
		if err != nil {
			logging.Errorf("Error extracting query from request: %v", err)
			return nil, false
		}
		ctx.RequestModel = requestModel
		ctx.RequestQuery = requestQuery

		// Add pending request for cache write only when semantic-cache is in scope.
		cacheEnabled := r.semanticCacheEnabledForScope(categoryName)
		r.storePendingCacheRequest(ctx, categoryName, requestModel, requestQuery, cacheEnabled)
		return nil, false
	}

	// Extract the model and query for cache lookup
	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		// Continue without caching
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	// Semantic-cache is decision-scoped when routing decisions exist.
	cacheEnabled := r.semanticCacheEnabledForScope(categoryName)

	logging.Infof("handleCaching: requestQuery='%s' (len=%d), cacheEnabled=%v, r.Cache.IsEnabled()=%v",
		requestQuery, len(requestQuery), cacheEnabled, r.Cache.IsEnabled())

	if response, shouldReturn := r.performCacheLookup(ctx, categoryName, requestModel, requestQuery, cacheEnabled); shouldReturn {
		return response, true
	}

	// Cache miss, store the request for later (only if caching is enabled for this decision)
	r.storePendingCacheRequest(ctx, categoryName, requestModel, requestQuery, cacheEnabled)

	return nil, false
}

// storePendingCacheRequest adds a pending cache request if caching is enabled for this decision.
func (r *OpenAIRouter) storePendingCacheRequest(ctx *RequestContext, categoryName, requestModel, requestQuery string, cacheEnabled bool) {
	if requestQuery == "" || !r.Cache.IsEnabled() || !cacheEnabled {
		return
	}
	ttlSeconds := r.Config.GetCacheTTLSecondsForDecision(categoryName)
	if err := r.Cache.AddPendingRequest(ctx.RequestID, requestModel, requestQuery, ctx.OriginalRequestBody, ttlSeconds); err != nil {
		logging.Errorf("Error adding pending request to cache: %v", err)
	}
}

// performCacheLookup searches for a cached response matching the request query.
// Returns the cached response and true on cache hit, or nil and false on miss/error/skip.
func (r *OpenAIRouter) performCacheLookup(
	ctx *RequestContext, categoryName, requestModel, requestQuery string, cacheEnabled bool,
) (*ext_proc.ProcessingResponse, bool) {
	if requestQuery == "" || !r.Cache.IsEnabled() || !cacheEnabled {
		return nil, false
	}

	// Get decision-specific threshold
	threshold := r.Config.GetCacheSimilarityThreshold()
	if categoryName != "" {
		threshold = r.Config.GetCacheSimilarityThresholdForDecision(categoryName)
	}

	logging.Infof("handleCaching: Performing cache lookup - model=%s, query='%s', threshold=%.2f",
		requestModel, requestQuery, threshold)

	// Start semantic-cache plugin span
	spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, "semantic-cache", categoryName)

	startTime := time.Now()
	// Try to find a similar cached response using category-specific threshold
	cachedResponse, found, cacheErr := r.Cache.FindSimilarWithThreshold(requestModel, requestQuery, threshold)
	lookupTime := time.Since(startTime).Milliseconds()

	logging.Infof("FindSimilarWithThreshold returned: found=%v, error=%v, lookupTime=%dms", found, cacheErr, lookupTime)

	// Keep legacy attributes for backward compatibility
	tracing.SetSpanAttributes(span,
		attribute.String(tracing.AttrCacheKey, requestQuery),
		attribute.Bool(tracing.AttrCacheHit, found),
		attribute.Int64(tracing.AttrCacheLookupTimeMs, lookupTime),
		attribute.String(tracing.AttrCategoryName, categoryName),
		attribute.Float64("cache.threshold", float64(threshold)))

	if cacheErr != nil {
		logging.Errorf("Error searching cache: %v", cacheErr)
		tracing.RecordError(span, cacheErr)
		tracing.EndPluginSpan(span, "error", lookupTime, "lookup_failed")
	} else if found {
		// Mark this request as a cache hit
		ctx.VSRCacheHit = true

		// Set VSR decision context even for cache hits so headers are populated
		// The categoryName passed here is the decision name from classification
		if categoryName != "" {
			ctx.VSRSelectedDecisionName = categoryName
		}

		// Record cache plugin hit with decision name
		metrics.RecordCachePluginHit(categoryName, "semantic-cache")

		// End plugin span with cache hit status
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_hit")

		// Start router replay capture if enabled, even when serving from cache
		r.startRouterReplay(ctx, requestModel, requestModel, categoryName)
		// Log cache hit
		logging.LogEvent("cache_hit", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      requestModel,
			"query":      requestQuery,
			"category":   categoryName,
			"threshold":  threshold,
		})
		// Return immediate response from cache
		response := http.CreateCacheHitResponse(cachedResponse, ctx.ExpectStreamingResponse, categoryName, ctx.VSRSelectedDecisionName, ctx.VSRMatchedKeywords)
		r.updateRouterReplayStatus(ctx, 200, ctx.ExpectStreamingResponse)
		r.attachRouterReplayResponse(ctx, cachedResponse, true)
		ctx.TraceContext = spanCtx
		return response, true
	} else {
		// Cache miss - record cache plugin miss with decision name
		metrics.RecordCachePluginMiss(categoryName, "semantic-cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_miss")
	}
	ctx.TraceContext = spanCtx

	return nil, false
}
