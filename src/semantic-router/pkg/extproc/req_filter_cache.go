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
	// Per-decision memory plugin takes priority over global setting.
	if memCfg := d.GetMemoryConfig(); memCfg != nil {
		return memCfg.Enabled
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

	if ctx.LooperRequest {
		return r.handleLooperCacheSkip(ctx, categoryName)
	}

	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(cacheRequestBodyForContext(ctx))
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery
	ctx.CacheQuery = cache.ScopeQueryToUser(requestQuery, cacheScopeUserID(ctx))

	cacheEnabled := r.semanticCacheEnabledForRequest(ctx)

	if response, shouldReturn := r.performCacheLookup(ctx, categoryName, requestModel, cacheEnabled); shouldReturn {
		return response, true
	}

	r.storePendingCacheRequest(ctx, categoryName, requestModel, cacheEnabled)

	return nil, false
}

// handleLooperCacheSkip extracts the query for a looper request (skipping read)
// and registers a pending cache write if caching is enabled.
func (r *OpenAIRouter) handleLooperCacheSkip(ctx *RequestContext, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	logging.Debugf("[Cache] Skipping cache read for looper internal request")

	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(cacheRequestBodyForContext(ctx))
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}
	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery
	ctx.CacheQuery = cache.ScopeQueryToUser(requestQuery, cacheScopeUserID(ctx))

	cacheEnabled := r.semanticCacheEnabledForRequest(ctx)
	r.storePendingCacheRequest(ctx, categoryName, requestModel, cacheEnabled)
	return nil, false
}

// semanticCachePartition keeps recipe identity out of the text passed to the
// embedding model. Cache backends treat model as an exact partition key, so
// named recipes cannot read each other's entries while the default recipe
// preserves the legacy model key and semantic similarity behavior.
func semanticCachePartition(ctx *RequestContext, model string) string {
	if ctx == nil {
		return model
	}
	return config.RoutingNamespaceKey(ctx.Routing.RecipeName(), model)
}

// storePendingCacheRequest adds a pending cache request if caching is enabled for this decision.
func (r *OpenAIRouter) storePendingCacheRequest(ctx *RequestContext, categoryName, requestModel string, cacheEnabled bool) {
	cacheQuery := cacheQueryForContext(ctx)
	if cacheQuery == "" || !r.Cache.IsEnabled() || !cacheEnabled {
		return
	}
	ttlSeconds := r.Config.GetCacheTTLSecondsForDecisionObject(ctx.VSRSelectedDecision)
	partition := semanticCachePartition(ctx, requestModel)
	if err := r.Cache.AddPendingRequest(ctx.TraceContext, ctx.RequestID, partition, cacheQuery, ctx.OriginalRequestBody, ttlSeconds); err != nil {
		logging.Errorf("Error adding pending request to cache: %v", err)
	}
}

// performCacheLookup searches for a cached response matching the request query.
// Returns the cached response and true on cache hit, or nil and false on miss/error/skip.
func (r *OpenAIRouter) performCacheLookup(
	ctx *RequestContext, categoryName, requestModel string, cacheEnabled bool,
) (*ext_proc.ProcessingResponse, bool) {
	cacheQuery := cacheQueryForContext(ctx)
	if cacheQuery == "" || !r.Cache.IsEnabled() || !cacheEnabled {
		return nil, false
	}

	threshold := r.Config.GetCacheSimilarityThreshold()
	if ctx.VSRSelectedDecision != nil {
		threshold = r.Config.GetCacheSimilarityThresholdForDecisionObject(ctx.VSRSelectedDecision)
	}

	logging.Infof("handleCaching: Performing cache lookup - model=%s, query=%s, threshold=%.2f",
		requestModel, logging.ContentDescriptor(ctx.RequestQuery), threshold)

	spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, "semantic-cache", categoryName)

	startTime := time.Now()
	partition := semanticCachePartition(ctx, requestModel)
	lookupResult, cacheErr := r.Cache.FindSimilarWithThreshold(spanCtx, partition, cacheQuery, threshold)
	lookupTime := time.Since(startTime).Milliseconds()
	cachedResponse := lookupResult.Body
	found := lookupResult.Found

	logging.Infof("FindSimilarWithThreshold returned: found=%v, error=%v, lookupTime=%dms", found, cacheErr, lookupTime)

	tracing.SetSpanAttributes(span,
		attribute.String(tracing.AttrCacheKey, ctx.RequestQuery),
		attribute.Bool(tracing.AttrCacheHit, found),
		attribute.Int64(tracing.AttrCacheLookupTimeMs, lookupTime),
		attribute.String(tracing.AttrCategoryName, categoryName),
		attribute.Float64("cache.threshold", float64(threshold)))

	if cacheErr != nil {
		logging.Errorf("Error searching cache: %v", cacheErr)
		tracing.RecordError(span, cacheErr)
		tracing.EndPluginSpan(span, "error", lookupTime, "lookup_failed")
	} else if found {
		ctx.VSRCacheHit = true
		ctx.VSRCacheSimilarity = lookupResult.Similarity

		if categoryName != "" {
			ctx.VSRSelectedDecisionName = categoryName
		}

		metrics.RecordCachePluginHit(requestDecisionStateKey(ctx), "semantic-cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_hit")

		r.startRouterReplay(ctx, requestModel, requestModel, categoryName)
		logging.LogEvent("cache_hit", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      requestModel,
			"query":      ctx.RequestQuery,
			"category":   categoryName,
			"threshold":  threshold,
		})
		// Intermediate cache detail (category, matched keywords, similarity) is
		// demoted to the x-vsr-debug surface (#2205).
		cacheCategory, cacheKeywords, cacheSimilarity := cacheDetailForSurface(ctx, categoryName)
		response := r.createCacheHitResponse(ctx, cachedResponse, cacheCategory, ctx.VSRSelectedDecisionName, cacheKeywords, cacheSimilarity)
		r.updateRouterReplayStatus(ctx, 200, ctx.ExpectStreamingResponse)
		r.attachRouterReplayResponse(ctx, cachedResponse, true)
		ctx.TraceContext = spanCtx
		return response, true
	} else {
		// No similarity is recorded: a miss leaves the debug/Replay surface
		// without a score (see LookupResult).
		metrics.RecordCachePluginMiss(requestDecisionStateKey(ctx), "semantic-cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_miss")
	}
	ctx.TraceContext = spanCtx

	return nil, false
}

func (r *OpenAIRouter) createCacheHitResponse(
	ctx *RequestContext,
	cachedResponse []byte,
	category string,
	decisionName string,
	matchedKeywords []string,
	similarity float32,
) *ext_proc.ProcessingResponse {
	response := http.CreateCacheHitResponse(
		cachedResponse,
		ctx.ExpectStreamingResponse,
		category,
		decisionName,
		matchedKeywords,
		similarity,
	)
	if isResponseAPIRequest(ctx) {
		if responseAPIResponse, ok := r.createResponseAPICacheHitResponse(
			ctx,
			cachedResponse,
			category,
			decisionName,
			matchedKeywords,
			similarity,
		); ok {
			response = responseAPIResponse
		}
	}
	appendRecipeHeaderToImmediateResponse(response, ctx)
	return response
}

func cacheRequestBodyForContext(ctx *RequestContext) []byte {
	if ctx != nil && ctx.ResponseAPICtx != nil && len(ctx.ResponseAPICtx.TranslatedBody) > 0 {
		return ctx.ResponseAPICtx.TranslatedBody
	}
	if ctx != nil {
		return ctx.OriginalRequestBody
	}
	return nil
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

// cacheDetailForSurface returns the intermediate cache-hit detail (category,
// matched keywords, similarity) when the request opted into x-vsr-debug, and
// empty values otherwise. CreateCacheHitResponse omits the empties, demoting
// the detail off the lean default surface (#2205).
func cacheDetailForSurface(ctx *RequestContext, categoryName string) (string, []string, float32) {
	if !debugHeadersRequested(ctx) {
		return "", nil, 0
	}
	return categoryName, ctx.VSRMatchedKeywords, ctx.VSRCacheSimilarity
}
