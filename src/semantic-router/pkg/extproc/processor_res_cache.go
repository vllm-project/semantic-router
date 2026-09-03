package extproc

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// updateResponseCache stores a buffered public response. Streaming responses
// are first reconstructed as neutral IR and encoded into this same buffered
// representation, so cache state never contains provider-specific stream
// accumulator state.
func (r *OpenAIRouter) updateResponseCache(ctx *RequestContext, responseBody []byte) {
	if ctx.RequestID == "" || responseBody == nil {
		return
	}
	if !r.semanticCacheEnabledForRequest(ctx) {
		return
	}
	if ctx.CacheWriteBypass {
		metrics.RecordCacheWriteSkipped("request_no_store")
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

	ttlSeconds := -1
	if r != nil && r.Config != nil {
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecisionObject(ctx.VSRSelectedDecision)
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
	return ctx != nil && (ctx.CacheSemanticSafe || ctx.CacheExactFingerprint == "")
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
