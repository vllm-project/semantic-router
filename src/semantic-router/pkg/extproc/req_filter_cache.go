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

	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	if skip, reason := r.shouldSkipSemanticCache(ctx, ctx.RequestQuery); skip {
		logging.Infof("[Cache] Skipping semantic cache lookup and write for personalized request: reason=%s decision=%s",
			reason, categoryName)
		return nil, false
	}

	cacheEnabled := r.semanticCacheEnabledForScope(categoryName)

	if response, shouldReturn := r.performCacheLookup(ctx, categoryName, requestModel, requestQuery, cacheEnabled); shouldReturn {
		return response, true
	}

	r.storePendingCacheRequest(ctx, categoryName, requestModel, requestQuery, cacheEnabled)

	return nil, false
}

// handleLooperCacheSkip extracts the query for a looper request (skipping read)
// and registers a pending cache write if caching is enabled.
func (r *OpenAIRouter) handleLooperCacheSkip(ctx *RequestContext, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	logging.Debugf("[Cache] Skipping cache read for looper internal request")

	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}
	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	if skip, reason := r.shouldSkipSemanticCache(ctx, ctx.RequestQuery); skip {
		logging.Infof("[Cache] Skipping semantic cache write for personalized looper request: reason=%s decision=%s",
			reason, categoryName)
		return nil, false
	}

	cacheEnabled := r.semanticCacheEnabledForScope(categoryName)
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

	threshold := r.Config.GetCacheSimilarityThreshold()
	if categoryName != "" {
		threshold = r.Config.GetCacheSimilarityThresholdForDecision(categoryName)
	}

	logging.Infof("handleCaching: Performing cache lookup - model=%s, query='%s', threshold=%.2f",
		requestModel, requestQuery, threshold)

	spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, "semantic-cache", categoryName)

	startTime := time.Now()
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
		tracing.EndPluginSpan(span, "error", lookupTime, "lookup_failed")
	} else if found {
		ctx.VSRCacheHit = true
		ctx.VSRCacheSimilarity = r.Cache.LastSimilarity()

		if categoryName != "" {
			ctx.VSRSelectedDecisionName = categoryName
		}

		metrics.RecordCachePluginHit(categoryName, "semantic-cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_hit")

		r.startRouterReplay(ctx, requestModel, requestModel, categoryName)
		logging.LogEvent("cache_hit", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      requestModel,
			"query":      requestQuery,
			"category":   categoryName,
			"threshold":  threshold,
		})
		response := http.CreateCacheHitResponse(cachedResponse, ctx.ExpectStreamingResponse, categoryName, ctx.VSRSelectedDecisionName, ctx.VSRMatchedKeywords, ctx.VSRCacheSimilarity)
		r.updateRouterReplayStatus(ctx, 200, ctx.ExpectStreamingResponse)
		r.attachRouterReplayResponse(ctx, cachedResponse, true)
		ctx.TraceContext = spanCtx
		return response, true
	} else {
		ctx.VSRCacheSimilarity = r.Cache.LastSimilarity()
		metrics.RecordCachePluginMiss(categoryName, "semantic-cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_miss")
	}
	ctx.TraceContext = spanCtx

	return nil, false
}

func (r *OpenAIRouter) shouldSkipSemanticCache(ctx *RequestContext, requestQuery string) (bool, string) {
	if r == nil || r.Config == nil {
		return false, ""
	}

	if decisionUsesRAG(ctx) {
		return true, "rag"
	}

	if r.requestCanUsePersonalMemory(ctx, requestQuery) {
		return true, "memory"
	}

	return false, ""
}

func decisionUsesRAG(ctx *RequestContext) bool {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return false
	}

	ragConfig := ctx.VSRSelectedDecision.GetRAGConfig()
	return ragConfig != nil && ragConfig.Enabled
}

func (r *OpenAIRouter) requestCanUsePersonalMemory(ctx *RequestContext, requestQuery string) bool {
	if ctx == nil {
		return false
	}
	if !r.memoryEnabledForRequest(ctx) {
		return false
	}
	if extractUserID(ctx) == "" {
		return false
	}
	if requestQuery == "" {
		return false
	}
	return ShouldSearchMemory(ctx, requestQuery)
}

func (r *OpenAIRouter) memoryEnabledForRequest(ctx *RequestContext) bool {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		memoryConfig := ctx.VSRSelectedDecision.GetMemoryConfig()
		if memoryConfig != nil {
			return memoryConfig.Enabled
		}
	}
	return r.Config.Memory.Enabled
}

func (r *OpenAIRouter) shouldSkipSemanticCacheWrite(ctx *RequestContext) (bool, string) {
	requestQuery := ""
	if ctx != nil {
		requestQuery = ctx.RequestQuery
		if ctx.RAGRetrievedContext != "" {
			return true, "rag_context"
		}
		if ctx.MemoryContext != "" {
			return true, "memory_context"
		}
	}

	return r.shouldSkipSemanticCache(ctx, requestQuery)
}
