package extproc

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// updateResponseCache stores a buffered public response. Streaming responses
// are first reconstructed as neutral IR and encoded into this same buffered
// representation, so cache state never contains provider-specific stream
// accumulator state.
func (r *OpenAIRouter) updateResponseCache(ctx *RequestContext, responseBody []byte) {
	if ctx.RequestID == "" || responseBody == nil || !r.cacheBackendEnabled() {
		return
	}
	if !r.semanticCacheEnabledForRequest(ctx) {
		return
	}
	// A streamed request's hit streams the entry back, so an answer the stream
	// encoder cannot write, such as a generated image, would fail every hit.
	if ctx.ExpectStreamingResponse && ctx.SemanticResponse != nil && !r.streamsFromCache(ctx) {
		metrics.RecordCacheWriteSkipped("stream_content_unsupported")
		return
	}
	if ctx.CacheWriteBypass {
		metrics.RecordCacheWriteSkipped("request_no_store")
		return
	}
	// A cache hit runs no guard, so an answer is kept only as the guard let it
	// through.
	if reason := ctx.ResponseGuard.cacheSkipReason(); reason != "" {
		metrics.RecordCacheWriteSkipped(reason)
		return
	}

	// A cached non-2xx body would be replayed as a success and freeze a
	// transient upstream failure for the entry lifetime.
	if skip, reason := shouldSkipCacheWriteForStatus(ctx); skip {
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping cache write for request ID %s: upstream status %d is not 2xx (reason=%s)", ctx.RequestID, ctx.UpstreamStatusCode, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return
	}

	if skip, reason := ShouldSkipCacheWrite(ctx); skip {
		metrics.RecordCacheWriteSkipped(reason)
		logRetentionSkip(ctx, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return
	}

	personalized, personalizedReason := ctx.HasPersonalizedContext()
	if personalized && !personalizedCacheWriteAllowed(ctx, personalizedReason) {
		metrics.RecordCacheWriteSkipped(personalizedReason)
		logging.Infof("Skipping cache write for request ID %s: response has personalized context (reason=%s)", ctx.RequestID, personalizedReason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, personalizedReason))
		}
		return
	}

	// A fallback candidate answered, so the answer belongs to the candidate's
	// selected-model partition, not to the primary's the lookup used.
	if ctx.ResponsePath == headers.ResponsePathFallback && ctx.CacheSelectedModel != ctx.VSRSelectedModel {
		ctx.CacheSelectedModel = ctx.VSRSelectedModel
		ctx.CacheIdentity = responseCacheIdentity(ctx, ctx.CacheRequestModel)
	}

	// Without a plugin ttl_seconds the entry lives the store's ttl_seconds,
	// which a caller's ttl and ttl_turns may only shorten.
	ttlSeconds := -1
	if plugin := requestResponseCache(ctx); plugin != nil && plugin.TTLSeconds != nil {
		ttlSeconds = *plugin.TTLSeconds
	} else if r.Config != nil && r.Config.SemanticCache.TTLSeconds > 0 {
		ttlSeconds = r.Config.SemanticCache.TTLSeconds
	}
	ttlSeconds = applyResponseCacheRequestTTL(ctx, ttlSeconds)
	ttlSeconds = applyRetentionTTLOverride(ttlSeconds, ctx)
	if !personalized && semanticCacheWriteAllowed(ctx) {
		if err := r.addSemanticCacheEntry(ctx, responseBody, ttlSeconds); err != nil {
			logging.Errorf("Error adding semantic cache entry: %v", err)
		}
	}
	r.updateExactResponseCache(ctx, responseBody, ttlSeconds)
	logging.Infof("Cache updated for request ID: %s", ctx.RequestID)
}

func semanticCacheWriteAllowed(ctx *RequestContext) bool {
	return ctx != nil && semanticLookupEnabledForRequest(ctx) &&
		(ctx.CacheSemanticSafe || ctx.CacheExactFingerprint == "")
}

// cacheWriteContext detaches a completed cache fill from client cancellation
// while retaining trace values. Read-side work remains cancellation-aware.
func cacheWriteContext(ctx *RequestContext) context.Context {
	if ctx == nil || ctx.TraceContext == nil {
		return context.Background()
	}
	return context.WithoutCancel(ctx.TraceContext)
}

func (r *OpenAIRouter) addSemanticCacheEntry(
	ctx *RequestContext,
	responseBody []byte,
	ttlSeconds int,
) error {
	query := cacheQueryForContext(ctx)
	if query == "" {
		return nil
	}
	requestModel := ctx.CacheRequestModel
	if requestModel == "" {
		requestModel = ctx.RequestModel
	}
	identity := ctx.CacheIdentity
	if identity.ExactFingerprint == "" {
		identity = responseCacheIdentity(ctx, requestModel)
	}
	identity.SemanticQuery = query
	service := r.responseCacheService()
	if service == nil {
		return nil
	}
	return service.StoreSemantic(cacheWriteContext(ctx), cache.CacheWrite{
		Identity:     identity,
		RequestID:    ctx.RequestID,
		RequestBody:  cacheRequestBodyForContext(ctx),
		ResponseBody: responseBody,
		TTL:          cache.TTLPolicyFromLegacySeconds(ttlSeconds),
	})
}

// streamsFromCache reports whether a cache hit can stream the request's
// answer back.
func (r *OpenAIRouter) streamsFromCache(ctx *RequestContext) bool {
	engine, err := r.protocolEngine()
	if err == nil {
		_, _, err = encodeCachedSemanticStream(engine, ctx, *ctx.SemanticResponse, nil)
	}
	return err == nil
}
