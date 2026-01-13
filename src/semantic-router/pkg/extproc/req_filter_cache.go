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
	logging.Infof("handleCaching: ENTRY - categoryName=%s, requestID=%s", categoryName, ctx.RequestID)

	// Extract the model and query for cache lookup
	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		// Continue without caching
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery
	logging.Infof("handleCaching: Extracted query='%s', model='%s'", requestQuery, requestModel)

	// Determine domain from category (default to "general" if no category)
	domain := categoryName
	if domain == "" {
		domain = "general"
	}
	logging.Infof("handleCaching: Using domain='%s'", domain)

	// Check if the decision has semantic-cache plugin enabled
	decisionCacheEnabled := r.Config.IsCacheEnabledForDecision(categoryName)
	logging.Infof("handleCaching: decisionCacheEnabled=%v for decision='%s'", decisionCacheEnabled, categoryName)
	if !decisionCacheEnabled {
		logging.Infof("Semantic-cache plugin is NOT enabled for decision '%s' - skipping cache", categoryName)
		return nil, false
	}

	// Check if caching is enabled for this domain
	cacheEnabled := r.CacheManager.IsEnabled(domain)
	logging.Infof("handleCaching: CacheManager.IsEnabled(%s)=%v", domain, cacheEnabled)
	if !cacheEnabled {
		logging.Infof("Caching is disabled for domain '%s' - skipping cache", domain)
		return nil, false
	}

	// Get domain-specific cache
	domainCache, err := r.CacheManager.GetCache(domain)
	if err != nil {
		logging.Errorf("Failed to get cache for domain '%s': %v", domain, err)
		// Try fallback to general cache
		domainCache, err = r.CacheManager.GetCache("general")
		if err != nil {
			logging.Errorf("Failed to get fallback general cache: %v", err)
			return nil, false
		}
		logging.Infof("Using fallback general cache for domain '%s'", domain)
	}

	logging.Infof("handleCaching: requestQuery='%s' (len=%d), domain='%s', cacheEnabled=%v",
		requestQuery, len(requestQuery), domain, cacheEnabled)

	if requestQuery != "" && domainCache.IsEnabled() {
		// Get decision-specific threshold
		threshold := r.Config.GetCacheSimilarityThreshold()
		if categoryName != "" {
			threshold = r.Config.GetCacheSimilarityThresholdForDecision(categoryName)
		}

		logging.Infof("handleCaching: Performing cache lookup - domain=%s, model=%s, query='%s', threshold=%.2f",
			domain, requestModel, requestQuery, threshold)

		// Start cache lookup span
		spanCtx, span := tracing.StartSpan(ctx.TraceContext, tracing.SpanCacheLookup)
		defer span.End()

		startTime := time.Now()
		// Use domain as namespace for cache isolation
		namespace := domain
		// Try to find a similar cached response using category-specific threshold and namespace
		cachedResponse, found, cacheErr := domainCache.FindSimilarWithThreshold(namespace, requestModel, requestQuery, threshold)
		lookupTime := time.Since(startTime).Milliseconds()

		logging.Infof("FindSimilarWithThreshold returned: found=%v, error=%v, lookupTime=%dms", found, cacheErr, lookupTime)

		tracing.SetSpanAttributes(span,
			attribute.String(tracing.AttrCacheKey, requestQuery),
			attribute.Bool(tracing.AttrCacheHit, found),
			attribute.Int64(tracing.AttrCacheLookupTimeMs, lookupTime),
			attribute.String(tracing.AttrCategoryName, categoryName),
			attribute.String("cache.domain", domain),
			attribute.Float64("cache.threshold", float64(threshold)))

		if cacheErr != nil {
			logging.Errorf("Error searching cache: %v", cacheErr)
			tracing.RecordError(span, cacheErr)
		} else if found {
			// Mark this request as a cache hit
			ctx.VSRCacheHit = true

			// Set VSR decision context even for cache hits so headers are populated
			// The categoryName passed here is the decision name from classification
			if categoryName != "" {
				ctx.VSRSelectedDecisionName = categoryName
			}

			// Log cache hit
			logging.LogEvent("cache_hit", map[string]interface{}{
				"request_id": ctx.RequestID,
				"model":      requestModel,
				"query":      requestQuery,
				"category":   categoryName,
				"domain":     domain,
				"threshold":  threshold,
			})
			// Return immediate response from cache
			response := http.CreateCacheHitResponse(cachedResponse, ctx.ExpectStreamingResponse, categoryName, ctx.VSRSelectedDecisionName, ctx.VSRMatchedKeywords)
			ctx.TraceContext = spanCtx
			return response, true
		}
		ctx.TraceContext = spanCtx
	}

	// Cache miss, store the request for later with namespace
	namespace := domain
	logging.Infof("handleCaching: CACHE MISS - Adding pending request: requestID=%s, namespace=%s, model=%s, query='%s'",
		ctx.RequestID, namespace, requestModel, requestQuery)
	err = domainCache.AddPendingRequest(ctx.RequestID, namespace, requestModel, requestQuery, ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("handleCaching: ERROR adding pending request to cache: %v", err)
		// Continue without caching
	} else {
		logging.Infof("handleCaching: SUCCESS - Pending request added to cache for requestID=%s", ctx.RequestID)
	}

	return nil, false
}
