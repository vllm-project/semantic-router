package extproc

import (
	"context"
	"fmt"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// decisionWillPersonalize checks whether the matched decision is configured
// with plugins (RAG, memory) that inject user-specific context. When true,
// we skip the entire cache path — both reads and writes — because:
//   - reads would serve a generic cached answer instead of the personalized one
//   - writes would cache a personalized answer that could leak to other users
//
// This avoids orphaned pending cache entries and unnecessary embedding work.
func decisionWillPersonalize(ctx *RequestContext) bool {
	plugin := config.ContextPluginOf(requestPlugins(nil, ctx))
	return plugin != nil && len(plugin.Sources) > 0
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
	if decisionWillPersonalize(ctx) {
		logging.Debugf("[Cache] Skipping cache for decision '%s': RAG or memory enabled", categoryName)
		return nil, false
	}

	if !r.cacheBackendEnabled() {
		applyRequestIdentityWithoutCache(ctx)
		if !ctx.LooperRequest {
			applyRequestCacheControls(ctx)
		}
		return nil, false
	}

	if ctx.LooperRequest {
		return r.handleLooperCacheSkip(ctx, categoryName, selectedModels...)
	}

	identity, err := cacheIdentityForContext(ctx)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}
	applyCacheIdentity(r.Config, ctx, identity, selectedModels)
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

	identity, err := cacheIdentityForContext(ctx)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		return nil, false
	}
	applyCacheIdentity(r.Config, ctx, identity, selectedModels)
	return nil, false
}

func applyCacheIdentity(cfg *config.RouterConfig, ctx *RequestContext, identity cache.RequestIdentity, selectedModels []string) {
	ctx.RequestModel = identity.Model
	ctx.RequestQuery = identity.Query
	ctx.CacheRequestModel = identity.Model
	ctx.CacheQuery = identity.Query
	policyFingerprint := responseCachePolicyFingerprint(cfg, ctx)
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
}

func selectedCacheModel(requestModel string, selectedModels []string) string {
	if len(selectedModels) > 0 && strings.TrimSpace(selectedModels[0]) != "" {
		return strings.TrimSpace(selectedModels[0])
	}
	return strings.TrimSpace(requestModel)
}

// responseCachePolicyFingerprint is what besides the request decides its
// answer: the plugins it runs, a decision's or, without one, the layered
// defaults, and the decision's models and output contract.
func responseCachePolicyFingerprint(cfg *config.RouterConfig, ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	policy := map[string]interface{}{
		"plugins":         requestPlugins(cfg, ctx),
		"client_protocol": string(ctx.SourceFormat),
	}
	if decision := ctx.VSRSelectedDecision; decision != nil {
		policy["decision"] = decision.Name
		policy["model_refs"] = decision.ModelRefs
		policy["output_contract"] = decision.OutputContract
		policy["output_contract_spec"] = decision.OutputContractSpec
	}
	fingerprint, err := cache.FingerprintValue(policy)
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
	protocol := strings.TrimSpace(string(ctx.SourceFormat))
	if protocol == "" {
		protocol = "openai"
	}
	if ctx.ExpectStreamingResponse {
		protocol += ":stream"
	} else {
		protocol += ":body"
	}
	decision := ""
	epoch := ""
	if ctx.VSRSelectedDecision != nil {
		decision = ctx.VSRSelectedDecision.Name
	}
	guard := requestResponseGuard(ctx)
	if plugin := requestResponseCache(ctx); plugin != nil && plugin.Revision != nil {
		if value, err := cache.FingerprintValue(plugin.Revision); err == nil {
			epoch = value
		}
	}
	// An entry is served only under the response_guard that passed it:
	// adding or changing the request's guard starts a new partition.
	if guard != nil {
		if value, err := cache.FingerprintValue(guard); err == nil {
			epoch = cache.CombineFingerprints(epoch, value)
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
			Epoch:         epoch,
		},
		ExactFingerprint:         ctx.CacheExactFingerprint,
		CompatibilityFingerprint: ctx.CacheCompatibilityFingerprint,
		SemanticQuery:            cacheQueryForContext(ctx),
	}
}

// performCacheLookup serves the semantic tier: the decision's semantic.match
// recalls the cached requests most similar to this one and, when it writes a
// judge, keeps those that ask the same thing. The first kept entry is served.
// Returns the cached response and true on cache hit, or nil and false on miss/error/skip.
func (r *OpenAIRouter) performCacheLookup(
	ctx *RequestContext, categoryName, requestModel string, cacheEnabled bool,
) (*ext_proc.ProcessingResponse, bool) {
	cacheQuery := cacheQueryForContext(ctx)
	if cacheQuery == "" || !r.cacheBackendEnabled() || !cacheEnabled {
		return nil, false
	}
	service := r.responseCacheService()
	if service == nil {
		return nil, false
	}

	block := requestResponseCache(ctx).SemanticMatch()
	recall := config.MatchResponseCache.Effective(block).Recall
	threshold := 0.0
	if recall != nil && recall.MinSimilarity != nil {
		threshold = *recall.MinSimilarity
	}
	logging.Infof("handleCaching: Performing cache lookup - model=%s, query=%s, threshold=%.2f",
		requestModel, logging.ContentDescriptor(ctx.RequestQuery), threshold)

	_, span := tracing.StartPluginSpan(ctx.TraceContext, "response_cache", categoryName)

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
	index := &responseCacheIndex{service: service, identity: identity, maxAge: responseCacheMaxAgeDuration(ctx)}
	matched, cacheErr := r.matcher().Match(lookupContext, config.MatchResponseCache, block, match.Input{Request: cacheQuery, Index: index})
	lookupResult := index.served(matched, cacheErr)
	service.RecordSemantic(identity, lookupResult, cacheErr, startTime)
	cachedResponse := lookupResult.ResponseBody
	found := lookupResult.Found
	lookupDuration := time.Since(startTime)
	lookupTime := lookupDuration.Milliseconds()

	logging.Infof("response cache semantic match: found=%v, recalled=%d, error=%v, lookupTime=%dms", found, len(index.recalled), cacheErr, lookupTime)

	tracing.SetSpanAttributes(span,
		attribute.Bool(tracing.AttrCacheHit, found),
		attribute.Int64(tracing.AttrCacheLookupTimeMs, lookupTime),
		attribute.String(tracing.AttrDecisionName, categoryName),
		attribute.Float64("cache.threshold", threshold),
		attribute.Int("cache.recalled", len(index.recalled)))
	if judge, judged := judgeStage(matched); judged {
		tracing.SetSpanAttributes(span,
			attribute.String("cache.judge.preset", judge.Method),
			attribute.String("cache.judge.outcome", judge.Outcome),
			attribute.Int("cache.judge.kept", judge.Kept))
	}

	if cacheErr != nil {
		logging.Errorf("Error searching cache: %v", cacheErr)
		tracing.RecordError(span, "cache_lookup_failed")
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
		tracing.SetSpanAttributes(span, attribute.String(tracing.AttrCacheNegationGuard, string(lookupResult.NegationGuard)))
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_hit")

		// The cache partition includes the selected backend, independently of
		// the public entrypoint and any provider alias in the cached body.
		r.startRouterReplay(ctx, ctx.CacheRequestModel, ctx.CacheSelectedModel, categoryName)
		r.reportCacheHitTelemetry(ctx, cachedResponse, lookupDuration)
		logging.LogEvent("cache_hit", map[string]interface{}{
			"request_id":     ctx.RequestID,
			"model":          requestModel,
			"query":          ctx.RequestQuery,
			"category":       categoryName,
			"threshold":      threshold,
			"negation_guard": string(lookupResult.NegationGuard),
		})
		response := r.createCacheHitResponse(ctx, cachedResponse)
		r.updateRouterReplayStatus(ctx, 200, ctx.ExpectStreamingResponse)
		return response, true
	} else {
		// A semantic miss may expose this lookup's rejected-candidate score on the
		// debug and Replay surfaces; LookupResult keeps it request-owned.
		ctx.VSRCacheSimilarity = lookupResult.Similarity
		metrics.RecordCachePluginMiss(requestDecisionStateKey(ctx), "response_cache")
		tracing.EndPluginSpan(span, "success", lookupTime, "cache_miss")
	}

	return nil, false
}

//nolint:nestif // Cache-hit output must preserve separate buffered and streaming protocol lifecycles.
func (r *OpenAIRouter) createCacheHitResponse(
	ctx *RequestContext,
	cachedResponse []byte,
) *ext_proc.ProcessingResponse {
	semanticResponse, err := r.decodeCachedClientResponse(cachedResponse, ctx)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "cache_response_decode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return r.createErrorResponse(502, "The cached response is invalid")
	}
	refreshCachedSemanticResponse(semanticResponse, ctx)
	contentType := "application/json"
	var responseBody []byte
	if ctx.ExpectStreamingResponse {
		engine, engineErr := r.protocolEngine()
		if engineErr == nil {
			responseBody, ctx.ProtocolDiagnostics, engineErr = encodeCachedSemanticStream(
				engine, ctx, *semanticResponse, ctx.ProtocolDiagnostics,
			)
		}
		if engineErr != nil {
			logging.ComponentErrorEvent("extproc", "cache_stream_encode_failed", map[string]interface{}{
				"request_id": ctx.RequestID,
				"format":     ctx.SourceFormat,
				"error":      engineErr.Error(),
			})
			return r.createErrorResponse(502, "The cached response cannot be streamed")
		}
		contentType = "text/event-stream"
	} else {
		responseBody, err = r.encodeClientResponse(*semanticResponse, ctx)
		if err != nil {
			logging.ComponentErrorEvent("extproc", "cache_response_encode_failed", map[string]interface{}{
				"request_id": ctx.RequestID,
				"format":     ctx.SourceFormat,
				"error":      err.Error(),
			})
			return r.createErrorResponse(502, "The cached response is invalid")
		}
	}
	response := http.CreateCacheHitResponseWithBody(responseBody, contentType)
	ctx.ImmediateResponseEncoded = true
	ctx.SemanticResponse = semanticResponse
	r.attachRouterReplayResponse(ctx, responseBody, true)
	return response
}

func refreshCachedSemanticResponse(response *llmprotocol.Response, ctx *RequestContext) {
	if response == nil {
		return
	}
	response.Generation++
	requestID := ""
	if ctx != nil {
		requestID = ctx.RequestID
	}
	response.ID = llmprotocol.StableID("cache-response", requestID)
	if ctx != nil && ctx.SourceFormat == llmprotocol.OpenAIResponsesV1 {
		response.ID = "resp_" + strings.TrimPrefix(response.ID, "item_")
	}
	if publicID := responseObjectPublicID(ctx); publicID != "" {
		response.ID = publicID
	}
	response.CreatedAt = time.Now().UTC()
	for index := range response.Output {
		response.Output[index].ID = llmprotocol.StableID(response.ID, fmt.Sprint(index))
	}
}

func encodeCachedSemanticStream(
	engine *protocolcodec.Engine,
	ctx *RequestContext,
	response llmprotocol.Response,
	diagnostics llmprotocol.Diagnostics,
) ([]byte, llmprotocol.Diagnostics, error) {
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	body, emitted, err := engine.EncodeResponseStream(format, response, llmprotocol.StreamContext{
		Context: ctx.TraceContext, Options: clientStreamOptions(ctx),
		PublicModel: response.Model, ResponseID: response.ID,
		PreviousResponseID: responseObjectPreviousID(ctx),
	})
	return body, append(diagnostics, emitted...), err
}

func applyCacheHitSelectedModel(ctx *RequestContext) {
	if ctx != nil && ctx.CacheSelectedModel != "" {
		ctx.RequestModel = ctx.CacheSelectedModel
	}
}

func (r *OpenAIRouter) cacheBackendEnabled() bool {
	return r != nil && r.Cache != nil && r.Cache.IsEnabled()
}

func applyRequestIdentityWithoutCache(ctx *RequestContext) {
	if ctx == nil || ctx.SemanticRequest == nil {
		return
	}
	ctx.RequestModel = ctx.SemanticRequest.Model
	ctx.RequestQuery = cache.SemanticRequestQuery(*ctx.SemanticRequest)
	if ctx.RoutingInput.Changed() {
		ctx.RequestQuery = ctx.UserContent
	}
	ctx.CacheRequestModel = ctx.RequestModel
	ctx.CacheQuery = ctx.RequestQuery
}

func cacheIdentityForContext(ctx *RequestContext) (cache.RequestIdentity, error) {
	if ctx == nil || ctx.SemanticRequest == nil {
		return cache.RequestIdentity{}, fmt.Errorf("neutral request is unavailable")
	}
	if routing := ctx.RoutingInput; routing.Changed() {
		return cache.BuildRoutedRequestIdentity(*ctx.SemanticRequest, cache.RoutedQuery{
			Text: ctx.UserContent, MessageIndex: routing.MessageIndex, Retained: routing.Stripped,
		})
	}
	return cache.BuildSemanticRequestIdentity(*ctx.SemanticRequest)
}

func cacheRequestBodyForContext(ctx *RequestContext) []byte {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil
	}
	body, err := cache.MarshalSemanticRequest(*ctx.SemanticRequest)
	if err != nil {
		return nil
	}
	return body
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
