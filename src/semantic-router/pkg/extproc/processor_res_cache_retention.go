package extproc

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// retentionDropReason is the canonical skip-reason value used in metrics, log
// and trace attributes when a cache write is skipped because the matched
// decision declared `EMIT retention { drop: true }`. Sharing the same key
// (tracing.AttrCacheWriteSkippedReason) with other skip causes lets operators
// query a single attribute to see why a cache write was skipped.
const retentionDropReason = "retention.drop"

// ShouldSkipCacheWrite reports whether the cache write must be skipped for
// the given request context, based solely on the emitted retention directive.
// The boolean indicates skip; the string is the canonical reason for
// metrics/log/trace observability.
//
// Contract (issues #1747, #2009):
//   - Only `retention.drop=true` triggers a write skip here.
//   - `ttl_turns` is consumed separately as a per-entry TTL override
//     (applyRetentionTTLOverride), not as a skip; `drop` and `ttl_turns` are
//     mutually exclusive by validation.
//   - `keep_current_model` is consumed by Router Learning protection;
//     `prefer_prefix_retention` is emitted to the pool as a response header.
//   - This gate does NOT affect cache reads; read-side gating lives in
//     req_filter_cache.go.
func ShouldSkipCacheWrite(ctx *RequestContext) (bool, string) {
	if ctx == nil || ctx.EmittedRetention == nil {
		return false, ""
	}
	if ctx.EmittedRetention.Drop != nil && *ctx.EmittedRetention.Drop {
		return true, retentionDropReason
	}
	return false, ""
}

// retentionDefaultSecondsPerTurn approximates one conversation turn as this many
// wall-clock seconds, bridging the turn-based `ttl_turns` directive to the
// seconds-based semantic-cache expiry. A configurable knob is follow-up work.
const retentionDefaultSecondsPerTurn = 60

// retentionTTLOverrideSeconds converts an emitted retention `ttl_turns` into a
// per-entry cache TTL in seconds. Only a positive ttl_turns produces an override
// (nil and zero fall through); returns (seconds, true) only when one applies.
func retentionTTLOverrideSeconds(ctx *RequestContext) (int, bool) {
	if ctx == nil || ctx.EmittedRetention == nil || ctx.EmittedRetention.TTLTurns == nil {
		return 0, false
	}
	turns := *ctx.EmittedRetention.TTLTurns
	if turns <= 0 {
		return 0, false
	}
	return turns * retentionDefaultSecondsPerTurn, true
}

// applyRetentionTTLOverride returns the cache-write TTL (seconds) to use,
// preferring an emitted retention `ttl_turns` override over the decision/global
// default. Callers compute the base TTL (GetCacheTTLSecondsForDecision) first
// and pass it here. This runs only after the drop gate (ShouldSkipCacheWrite);
// drop and ttl_turns are mutually exclusive by validation.
func applyRetentionTTLOverride(base int, ctx *RequestContext) int {
	if override, ok := retentionTTLOverrideSeconds(ctx); ok {
		return override
	}
	return base
}

// observeRetentionDirective records every declared field of the emitted
// retention directive to log + trace. It is invoked once per matched decision
// (right after the deep clone in applyDecisionResultToContext). Fields that
// are nil are not recorded — preserving the tri-state (unset vs explicit zero)
// semantics of the schema.
//
// This records every declared field for audit (DSL → AST → config → ctx →
// log/trace) regardless of which fields the runtime consumes, complementing the
// x-vsr-retention-* response headers so operators can verify wiring end-to-end.
func observeRetentionDirective(ctx *RequestContext) {
	if ctx == nil || ctx.EmittedRetention == nil {
		return
	}
	r := ctx.EmittedRetention
	span := trace.SpanFromContext(ctx.TraceContext)
	fields := map[string]interface{}{"request_id": ctx.RequestID}
	if ctx.VSRSelectedDecisionName != "" {
		fields["decision"] = ctx.VSRSelectedDecisionName
	}

	if r.Drop != nil {
		fields["vsr.retention.drop"] = *r.Drop
		if span.IsRecording() {
			span.SetAttributes(attribute.Bool("vsr.retention.drop", *r.Drop))
		}
	}
	if r.TTLTurns != nil {
		fields["vsr.retention.ttl_turns"] = *r.TTLTurns
		if span.IsRecording() {
			span.SetAttributes(attribute.Int("vsr.retention.ttl_turns", *r.TTLTurns))
		}
	}
	if r.KeepCurrentModel != nil {
		fields["vsr.retention.keep_current_model"] = *r.KeepCurrentModel
		if span.IsRecording() {
			span.SetAttributes(attribute.Bool("vsr.retention.keep_current_model", *r.KeepCurrentModel))
		}
	}
	if r.PreferPrefixRetention != nil {
		fields["vsr.retention.prefer_prefix_retention"] = *r.PreferPrefixRetention
		if span.IsRecording() {
			span.SetAttributes(attribute.Bool("vsr.retention.prefer_prefix_retention", *r.PreferPrefixRetention))
		}
	}
	logging.ComponentDebugEvent("extproc", "retention_directive_observed", fields)
}

// logRetentionSkip records a structured log entry when the cache write is
// skipped due to a retention directive. Kept separate from the trace attr so
// that callers can choose to log + set-attr at distinct points (the trace
// attr is set inline in updateResponseCache / cacheStreamingResponse to share
// the AttrCacheWriteSkippedReason key with other skip causes).
func logRetentionSkip(ctx *RequestContext, reason string) {
	if ctx == nil {
		return
	}
	logging.ComponentEvent("extproc", "cache_write_skipped_retention", map[string]interface{}{
		"request_id": ctx.RequestID,
		"decision":   ctx.VSRSelectedDecisionName,
		"reason":     reason,
	})
}

// applyEmittedRetention extracts the first `retention` EmitDirective from the
// matched decision and stores a deep clone on the request context. The clone
// is required because EmitDirective.Retention holds tri-state pointers; a
// shallow copy would alias the read-only config tree and any downstream
// mutation could poison configuration shared by other in-flight requests.
//
// Returns the cloned directive (nil if the decision did not emit retention).
func applyEmittedRetention(decision *config.Decision, ctx *RequestContext) *config.RetentionDirective {
	if decision == nil || ctx == nil {
		return nil
	}
	for _, e := range decision.Emits {
		if e.Kind != "retention" || e.Retention == nil {
			continue
		}
		clone := &config.RetentionDirective{
			Drop:                  cloneBoolPtr(e.Retention.Drop),
			TTLTurns:              cloneIntPtr(e.Retention.TTLTurns),
			KeepCurrentModel:      cloneBoolPtr(e.Retention.KeepCurrentModel),
			PreferPrefixRetention: cloneBoolPtr(e.Retention.PreferPrefixRetention),
		}
		ctx.EmittedRetention = clone
		return clone
	}
	return nil
}

func cloneBoolPtr(p *bool) *bool {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func cloneIntPtr(p *int) *int {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
