package extproc

import (
	"context"
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
	if !canPerformExactCacheLookup(r, ctx, cacheEnabled) {
		return nil, false
	}
	service := r.responseCacheService()
	if service == nil {
		return nil, false
	}
	identity, maxAge, lookupContext := exactCacheLookupInput(ctx)
	start := time.Now()
	result, lease, err := service.LookupExactCoalesced(lookupContext, cache.ExactLookup{
		Identity: identity,
		MaxAge:   maxAge,
	})
	ctx.CacheIdentity = lease.Identity
	lookupDuration := time.Since(start)
	lookupTime := lookupDuration.Milliseconds()
	return r.finishExactCacheLookup(ctx, categoryName, result, err, lookupDuration, lookupTime)
}

func canPerformExactCacheLookup(
	r *OpenAIRouter,
	ctx *RequestContext,
	cacheEnabled bool,
) bool {
	return cacheEnabled &&
		r != nil &&
		r.Cache != nil &&
		r.Cache.IsEnabled() &&
		ctx != nil &&
		ctx.CacheExactFingerprint != "" &&
		exactCacheEnabledForRequest(ctx)
}

func exactCacheLookupInput(
	ctx *RequestContext,
) (cache.CacheIdentity, *time.Duration, context.Context) {
	identity := ctx.CacheIdentity
	if identity.ExactFingerprint == "" {
		identity = responseCacheIdentity(ctx, ctx.CacheRequestModel)
	}
	var maxAge *time.Duration
	if ctx.CacheMaxAgeSeconds != nil {
		value := time.Duration(*ctx.CacheMaxAgeSeconds) * time.Second
		maxAge = &value
	}
	lookupContext := ctx.TraceContext
	if lookupContext == nil {
		lookupContext = context.Background()
	}
	return identity, maxAge, lookupContext
}

func (r *OpenAIRouter) finishExactCacheLookup(
	ctx *RequestContext,
	categoryName string,
	result cache.CacheResult,
	err error,
	lookupDuration time.Duration,
	lookupTime int64,
) (*ext_proc.ProcessingResponse, bool) {
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
	ctx.VSRCacheHitKind = string(result.HitKind)
	ctx.VSRCacheSource = string(result.Source)
	ctx.VSRCacheEntryAgeSeconds = result.Age.Seconds()
	if !result.ExpiresAt.IsZero() {
		ctx.VSRCacheTTLSeconds = int(time.Until(result.ExpiresAt).Seconds())
	}
	applyCacheHitSelectedModel(ctx)
	if categoryName != "" {
		ctx.VSRSelectedDecisionName = categoryName
	}
	metrics.RecordCachePluginHit(requestDecisionStateKey(ctx), "response_cache")
	r.startRouterReplay(ctx, ctx.CacheRequestModel, ctx.CacheSelectedModel, categoryName)
	r.reportCacheHitTelemetry(ctx, result.ResponseBody, lookupDuration)
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
	service := r.responseCacheService()
	if service == nil {
		return
	}
	identity := ctx.CacheIdentity
	if identity.ExactFingerprint == "" {
		identity = responseCacheIdentity(ctx, ctx.CacheRequestModel)
	}
	writeContext := ctx.TraceContext
	if writeContext == nil {
		writeContext = context.Background()
	}
	if err := service.StoreExact(writeContext, cache.CacheWrite{
		Identity:     identity,
		RequestID:    ctx.RequestID,
		RequestBody:  cacheRequestBodyForContext(ctx),
		ResponseBody: responseBody,
		TTL:          cache.TTLPolicyFromLegacySeconds(ttlSeconds),
	}); err != nil {
		logging.ComponentWarnEvent("extproc", "exact_cache_write_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      err.Error(),
		})
	}
}
