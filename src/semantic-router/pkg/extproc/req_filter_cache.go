package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// handleCaching handles cache lookup and storage with category-specific settings
func (r *OpenAIRouter) handleCaching(ctx *RequestContext, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	// Extract the model and query for cache lookup
	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		// Continue without caching
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	// Check if caching is enabled for this decision
	cacheEnabled := r.Config.SemanticCache.Enabled
	if categoryName != "" {
		cacheEnabled = r.Config.IsCacheEnabledForDecision(categoryName)
	}

	logging.Infof("handleCaching: requestQuery='%s' (len=%d), cacheEnabled=%v, r.Cache.IsEnabled()=%v",
		requestQuery, len(requestQuery), cacheEnabled, r.Cache.IsEnabled())

	if requestQuery != "" && r.Cache.IsEnabled() && cacheEnabled {
		// Get decision-specific threshold
		threshold := r.Config.GetCacheSimilarityThreshold()
		if categoryName != "" {
			threshold = r.Config.GetCacheSimilarityThresholdForDecision(categoryName)
		}

		logging.Infof("handleCaching: Performing cache lookup - model=%s, query='%s', threshold=%.2f",
			requestModel, requestQuery, threshold)

		// Start cache lookup span
		spanCtx, span := tracing.StartSpan(ctx.TraceContext, tracing.SpanCacheLookup)
		defer span.End()

		startTime := time.Now()
		// Try to find a similar cached response using category-specific threshold
		cachedResponse, found, cacheErr := r.Cache.FindSimilarWithThreshold(requestModel, requestQuery, threshold)
		lookupTime := time.Since(startTime).Milliseconds()

		logging.Infof("FindSimilarWithThreshold returned: found=%v, error=%v, lookupTime=%dms", found, cacheErr, lookupTime)

		tracing.SetSpanAttributes(span,
			attribute.String(tracing.AttrCacheKey, requestQuery),
			attribute.Bool(tracing.AttrCacheHit, found),
			attribute.Int64(tracing.AttrCacheLookupTimeMs, lookupTime),
			attribute.String(tracing.AttrCategoryName, categoryName),
			attribute.Float64("cache.threshold", float64(threshold)))

		if cacheErr != nil {
			logging.Errorf("Error searching cache: %v", cacheErr)
			tracing.RecordError(span, cacheErr)
		} else if found {
			// Mark this request as a cache hit
			ctx.VSRCacheHit = true
			// Log cache hit
			logging.LogEvent("cache_hit", map[string]interface{}{
				"request_id": ctx.RequestID,
				"model":      requestModel,
				"query":      requestQuery,
				"category":   categoryName,
				"threshold":  threshold,
			})
			// Return immediate response from cache
			response := http.CreateCacheHitResponse(cachedResponse, ctx.ExpectStreamingResponse)
			ctx.TraceContext = spanCtx
			return response, true
		}
		ctx.TraceContext = spanCtx
	}

	// Cache miss, store the request for later
	err = r.Cache.AddPendingRequest(ctx.RequestID, requestModel, requestQuery, ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error adding pending request to cache: %v", err)
		// Continue without caching
	}

	return nil, false
}
