package extproc

import (
	"context"
	"strings"
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
func (r *OpenAIRouter) handleCaching(
	ctx *RequestContext,
	categoryName string,
	selectedModels ...string,
) (*ext_proc.ProcessingResponse, bool) {
	// Skip entire cache path for decisions that will inject user-specific context.
	// Both reads (would serve stale generic answers) and writes (would leak
	// personalized data) are wrong when RAG or memory is enabled.
	if decisionWillPersonalize(ctx, r.Config) {
		logging.Debugf("[Cache] Skipping cache for decision '%s': RAG or memory enabled", categoryName)
		return nil, false
	}

	if ctx.LooperRequest {
		return r.handleLooperCacheSkip(ctx, categoryName, selectedModels...)
	}

	identity, err := cache.BuildRequestIdentity(cacheRequestBodyForContext(ctx))
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}

	ctx.RequestModel = identity.Model
	ctx.RequestQuery = identity.Query
	ctx.CacheRequestModel = identity.Model
	ctx.CacheQuery = identity.Query
	policyFingerprint := responseCachePolicyFingerprint(ctx)
	ctx.CacheExactFingerprint = cache.CombineFingerprints(
		identity.ExactFingerprint,
		policyFingerprint,
	)
	ctx.CacheCompatibilityFingerprint = cache.CombineFingerprints(
		identity.CompatibilityFingerprint,
		policyFingerprint,
	)
	ctx.CacheSelectedModel = selectedCacheModel(identity.Model, selectedModels)
	ctx.CacheSemanticSafe = identity.SemanticSafe
	ctx.CacheIdentity = responseCacheIdentity(ctx, identity.Model)
	cacheEnabled := r.semanticCacheEnabledForRequest(ctx)
	applyRequestCacheControls(ctx)
	if !ctx.CacheReadBypass {
		if response, shouldReturn := r.performExactCacheLookup(ctx, categoryName, cacheEnabled); shouldReturn {
			return response, true
		}
	}
	if !identity.SemanticSafe || !semanticLookupEnabledForRequest(ctx) {
		logging.ComponentDebugEvent("extproc", "semantic_cache_lookup_skipped", map[string]interface{}{
			"request_id": ctx.RequestID,
			"reason":     semanticCacheSkipReason(identity.SemanticSafe, ctx),
		})
		return nil, false
	}

	if !ctx.CacheReadBypass {
		if response, shouldReturn := r.performCacheLookup(ctx, categoryName, identity.Model, cacheEnabled); shouldReturn {
			return response, true
		}
	}

	return nil, false
}

func semanticCacheSkipReason(semanticSafe bool, ctx *RequestContext) string {
	if !semanticSafe {
		return "unsupported_current_user_content"
	}
	if !semanticLookupEnabledForRequest(ctx) {
		return "exact_only_mode"
	}
	return "disabled"
}

// handleLooperCacheSkip extracts the query for a looper request (skipping read)
// and registers a pending cache write if caching is enabled.
func (r *OpenAIRouter) handleLooperCacheSkip(
	ctx *RequestContext,
	_ string,
	selectedModels ...string,
) (*ext_proc.ProcessingResponse, bool) {
	logging.Debugf("[Cache] Skipping cache read for looper internal request")

	identity, err := cache.BuildRequestIdentity(cacheRequestBodyForContext(ctx))
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}
	ctx.RequestModel = identity.Model
	ctx.RequestQuery = identity.Query
	ctx.CacheRequestModel = identity.Model
	ctx.CacheQuery = identity.Query
	policyFingerprint := responseCachePolicyFingerprint(ctx)
	ctx.CacheExactFingerprint = cache.CombineFingerprints(
		identity.ExactFingerprint,
		policyFingerprint,
	)
	ctx.CacheCompatibilityFingerprint = cache.CombineFingerprints(
		identity.CompatibilityFingerprint,
		policyFingerprint,
	)
	ctx.CacheSelectedModel = selectedCacheModel(identity.Model, selectedModels)
	ctx.CacheSemanticSafe = identity.SemanticSafe
	ctx.CacheIdentity = responseCacheIdentity(ctx, identity.Model)
	return nil, false
}

func selectedCacheModel(requestModel string, selectedModels []string) string {
	if len(selectedModels) > 0 && strings.TrimSpace(selectedModels[0]) != "" {
		return strings.TrimSpace(selectedModels[0])
	}
	return strings.TrimSpace(requestModel)
}

func responseCachePolicyFingerprint(ctx *RequestContext) string {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return ""
	}
	fingerprint, err := cache.FingerprintValue(map[string]interface{}{
		"decision":             ctx.VSRSelectedDecision.Name,
		"model_refs":           ctx.VSRSelectedDecision.ModelRefs,
		"plugins":              ctx.VSRSelectedDecision.Plugins,
		"output_contract":      ctx.VSRSelectedDecision.OutputContract,
		"output_contract_spec": ctx.VSRSelectedDecision.OutputContractSpec,
		"client_protocol":      ctx.ClientProtocol,
	})
	if err != nil {
		return ""
	}
	return fingerprint
}

func semanticCachePartition(ctx *RequestContext, model string) string {
	if ctx == nil {
		return model
	}
	return responseCacheIdentity(ctx, model).Partition.Key()
}

func responseCacheIdentity(ctx *RequestContext, model string) cache.CacheIdentity {
	protocol := strings.TrimSpace(ctx.ClientProtocol)
	if protocol == "" {
		protocol = "openai"
	}
	if ctx.ExpectStreamingResponse {
		protocol += ":stream"
	} else {
		protocol += ":body"
	}
	decision := ""
	revision := ""
	if ctx.VSRSelectedDecision != nil {
		decision = ctx.VSRSelectedDecision.Name
		if plugin := ctx.VSRSelectedDecision.GetResponseCacheConfig(); plugin != nil &&
			plugin.Revision != nil {
			if value, err := cache.FingerprintValue(plugin.Revision); err == nil {
				revision = value
			}
		}
	}
	scope := responseCacheScope(ctx)
	scopeIdentity := responseCacheScopeIdentity(ctx)
	if scopeIdentity != "" {
		scopeIdentity = scope + ":" + scopeIdentity
	}
	return cache.CacheIdentity{
		Partition: cache.CachePartition{
			Recipe:        string(ctx.Routing.RecipeName()),
			Decision:      decision,
			RequestModel:  strings.TrimSpace(model),
			SelectedModel: strings.TrimSpace(ctx.CacheSelectedModel),
			Protocol:      protocol,
			Namespace:     cache.UserScopeNamespace(scopeIdentity),
			Epoch:         revision,
		},
		ExactFingerprint:         ctx.CacheExactFingerprint,
		CompatibilityFingerprint: ctx.CacheCompatibilityFingerprint,
		SemanticQuery:            cacheQueryForContext(ctx),
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

	spanCtx, span := tracing.StartPluginSpan(ctx.TraceContext, "response_cache", categoryName)

	startTime := time.Now()
	identity := ctx.CacheIdentity
	if identity.ExactFingerprint == "" {
		identity = responseCacheIdentity(ctx, requestModel)
	}
	identity.SemanticQuery = cacheQuery
	lookupContext := ctx.TraceContext
	if lookupContext == nil {
		lookupContext = context.Background()
	}
	service := r.responseCacheService()
	if service == nil {
		return nil, false
	}
	lookupResult, cacheErr := service.LookupSemantic(lookupContext, cache.SemanticLookup{
		Identity:  identity,
		Threshold: threshold,
	})
	cachedResponse := lookupResult.ResponseBody
	found := lookupResult.Found
	lookupDuration := time.Since(startTime)
	lookupTime := lookupDuration.Milliseconds()

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
		ctx.VSRCacheHitKind = string(lookupResult.HitKind)
		ctx.VSRCacheSource = string(lookupResult.Source)
		ctx.VSRCacheEntryAgeSeconds = lookupResult.Age.Seconds()
		applyCacheHitSelectedModel(ctx)

		if categoryName != "" {
			ctx.VSRSelectedDecisionName = categoryName
		}

		metrics.RecordCachePluginHit(requestDecisionStateKey(ctx), "response_cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_hit")

		r.startRouterReplay(ctx, requestModel, requestModel, categoryName)
		r.reportCacheHitTelemetry(ctx, cachedResponse, lookupDuration)
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
		// A semantic miss may expose this lookup's rejected-candidate score on the
		// debug and Replay surfaces; LookupResult keeps it request-owned.
		ctx.VSRCacheSimilarity = lookupResult.Similarity
		metrics.RecordCachePluginMiss(requestDecisionStateKey(ctx), "response_cache")
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
	if ctx.ClientProtocol == config.ClientProtocolAnthropic {
		hydrateAnthropicCacheUsage(ctx, cachedResponse)
	}
	response := http.CreateCacheHitResponse(
		cachedResponse,
		ctx.ExpectStreamingResponse,
		category,
		decisionName,
		matchedKeywords,
		similarity,
	)
	if ctx.ClientProtocol == config.ClientProtocolAnthropic {
		response = translateAnthropicCacheHit(ctx, response)
	}
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

func applyCacheHitSelectedModel(ctx *RequestContext) {
	if ctx != nil && ctx.CacheSelectedModel != "" {
		ctx.RequestModel = ctx.CacheSelectedModel
	}
}

func cacheRequestBodyForContext(ctx *RequestContext) []byte {
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
