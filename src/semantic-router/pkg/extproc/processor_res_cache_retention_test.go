package extproc

import (
	"context"
	"testing"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func retBool(b bool) *bool { return &b }

func TestShouldSkipCacheWrite(t *testing.T) {
	cases := []struct {
		name       string
		ctx        *RequestContext
		wantSkip   bool
		wantReason string
	}{
		{name: "nil_ctx", ctx: nil, wantSkip: false},
		{name: "no_retention", ctx: &RequestContext{}, wantSkip: false},
		{
			name:     "retention_without_drop",
			ctx:      &RequestContext{Retention: &config.DecisionRetention{TTLTurns: intPtr(3)}},
			wantSkip: false,
		},
		{
			name:     "drop_false",
			ctx:      &RequestContext{Retention: &config.DecisionRetention{Drop: boolPtr(false)}},
			wantSkip: false,
		},
		{
			name:       "drop_true",
			ctx:        &RequestContext{Retention: &config.DecisionRetention{Drop: boolPtr(true)}},
			wantSkip:   true,
			wantReason: "retention.drop",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			skip, reason := ShouldSkipCacheWrite(tc.ctx)
			if skip != tc.wantSkip || reason != tc.wantReason {
				t.Fatalf("got (%v,%q), want (%v,%q)", skip, reason, tc.wantSkip, tc.wantReason)
			}
		})
	}
}

func TestApplyDecisionRetentionDeepClone(t *testing.T) {
	src := &config.Decision{
		Name: "r",
		Retention: &config.DecisionRetention{
			Drop:                  boolPtr(true),
			TTLTurns:              intPtr(5),
			KeepCurrentModel:      boolPtr(true),
			PreferPrefixRetention: boolPtr(false),
		},
	}
	ctx := &RequestContext{}
	got := applyDecisionRetention(src, ctx)
	if got == nil || ctx.Retention != got {
		t.Fatalf("expected ctx.Retention to be set to returned clone")
	}
	if ctx.Retention.Drop == nil || !*ctx.Retention.Drop {
		t.Fatalf("drop missing on clone")
	}

	// Mutate the clone; source must remain untouched (no pointer aliasing).
	*ctx.Retention.Drop = false
	*ctx.Retention.TTLTurns = 99
	if !*src.Retention.Drop {
		t.Fatalf("source Drop polluted by clone mutation")
	}
	if *src.Retention.TTLTurns != 5 {
		t.Fatalf("source TTLTurns polluted by clone mutation, got %d", *src.Retention.TTLTurns)
	}
}

func TestApplyDecisionRetentionNoMatch(t *testing.T) {
	src := &config.Decision{Name: "r"} // no retention
	ctx := &RequestContext{}
	if got := applyDecisionRetention(src, ctx); got != nil {
		t.Fatalf("expected nil when decision has no retention")
	}
	if ctx.Retention != nil {
		t.Fatalf("expected Retention to stay nil")
	}
}

func TestApplyEmittedRetentionNilInputs(t *testing.T) {
	if got := applyDecisionRetention(nil, &RequestContext{}); got != nil {
		t.Fatalf("nil decision must yield nil")
	}
	if got := applyDecisionRetention(&config.Decision{}, nil); got != nil {
		t.Fatalf("nil ctx must yield nil")
	}
}

func TestObserveRetentionDirectiveNoOp(t *testing.T) {
	// Must not panic on nil ctx / nil retention.
	observeDecisionRetention(nil)
	observeDecisionRetention(&RequestContext{})
}

func TestObserveRetentionDirectiveRecordsLogAndTraceFields(t *testing.T) {
	core, logs := observer.New(zapcore.DebugLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(core))
	defer restoreLogger()

	spanRecorder := tracetest.NewSpanRecorder()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRecorder))
	traceCtx, span := tracerProvider.Tracer("retention-test").Start(context.Background(), "retention")

	ctx := &RequestContext{
		RequestID:               "req-retention-observe",
		TraceContext:            traceCtx,
		VSRSelectedDecisionName: "retention-decision",
		Retention: &config.DecisionRetention{
			Drop:                  boolPtr(false),
			TTLTurns:              intPtr(2),
			KeepCurrentModel:      boolPtr(true),
			PreferPrefixRetention: boolPtr(true),
		},
	}

	observeDecisionRetention(ctx)
	span.End()

	entries := logs.FilterMessage("retention_directive_observed").All()
	if len(entries) != 1 {
		t.Fatalf("expected one retention log entry, got %d", len(entries))
	}
	fields := entries[0].ContextMap()
	assertLogField(t, fields, "request_id", "req-retention-observe")
	assertLogField(t, fields, "decision", "retention-decision")
	assertLogField(t, fields, "vsr.retention.drop", false)
	assertLogField(t, fields, "vsr.retention.ttl_turns", int64(2))
	assertLogField(t, fields, "vsr.retention.keep_current_model", true)
	assertLogField(t, fields, "vsr.retention.prefer_prefix_retention", true)

	spans := spanRecorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected one recorded span, got %d", len(spans))
	}
	attrs := map[string]interface{}{}
	for _, attr := range spans[0].Attributes() {
		attrs[string(attr.Key)] = attr.Value.AsInterface()
	}
	assertLogField(t, attrs, "vsr.retention.drop", false)
	assertLogField(t, attrs, "vsr.retention.ttl_turns", int64(2))
	assertLogField(t, attrs, "vsr.retention.keep_current_model", true)
	assertLogField(t, attrs, "vsr.retention.prefer_prefix_retention", true)
}

func TestObserveRetentionDirectiveOmitsUnsetFields(t *testing.T) {
	core, logs := observer.New(zapcore.DebugLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(core))
	defer restoreLogger()

	spanRecorder := tracetest.NewSpanRecorder()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRecorder))
	traceCtx, span := tracerProvider.Tracer("retention-test").Start(context.Background(), "retention")

	observeDecisionRetention(&RequestContext{
		TraceContext: traceCtx,
		Retention:    &config.DecisionRetention{Drop: boolPtr(true)},
	})
	span.End()

	entries := logs.FilterMessage("retention_directive_observed").All()
	if len(entries) != 1 {
		t.Fatalf("expected one retention log entry, got %d", len(entries))
	}
	fields := entries[0].ContextMap()
	assertLogField(t, fields, "vsr.retention.drop", true)
	assertFieldAbsent(t, fields, "vsr.retention.ttl_turns")
	assertFieldAbsent(t, fields, "vsr.retention.keep_current_model")
	assertFieldAbsent(t, fields, "vsr.retention.prefer_prefix_retention")

	spans := spanRecorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected one recorded span, got %d", len(spans))
	}
	attrs := map[string]interface{}{}
	for _, attr := range spans[0].Attributes() {
		attrs[string(attr.Key)] = attr.Value.AsInterface()
	}
	assertLogField(t, attrs, "vsr.retention.drop", true)
	assertFieldAbsent(t, attrs, "vsr.retention.ttl_turns")
	assertFieldAbsent(t, attrs, "vsr.retention.keep_current_model")
	assertFieldAbsent(t, attrs, "vsr.retention.prefer_prefix_retention")
}

func TestUpdateResponseCacheSkipsRetentionDrop(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(retentionCacheDecision("cache-decision", true))
	ctx := withSelectedDecision(&RequestContext{
		RequestID: "req-retention-drop",
		Retention: &config.DecisionRetention{Drop: retBool(true)},
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
	if mockCache.addEntryCalled {
		t.Fatalf("retention.drop must skip non-streaming cache write")
	}
}

func TestUpdateResponseCacheWritesWhenRetentionDropFalse(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(retentionCacheDecision("cache-decision", true))
	ctx := withSelectedDecision(&RequestContext{
		RequestID:       "req-retention-keep",
		RequestModel:    "test-model",
		RequestQuery:    "hello",
		SemanticRequest: testNeutralRequest("test-model", "hello"),
		Retention:       &config.DecisionRetention{Drop: retBool(false)},
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
	if !mockCache.addEntryCalled {
		t.Fatalf("drop=false must preserve non-streaming cache write")
	}
}

func TestRetentionTTLOverrideSeconds(t *testing.T) {
	cases := []struct {
		name    string
		ctx     *RequestContext
		wantSec int
		wantOK  bool
	}{
		{name: "nil_ctx", ctx: nil, wantSec: 0, wantOK: false},
		{name: "no_retention", ctx: &RequestContext{}, wantSec: 0, wantOK: false},
		{
			name:    "ttl_turns_unset",
			ctx:     &RequestContext{Retention: &config.DecisionRetention{Drop: retBool(false)}},
			wantSec: 0, wantOK: false,
		},
		{
			name:    "ttl_turns_zero_is_noop",
			ctx:     &RequestContext{Retention: &config.DecisionRetention{TTLTurns: intPtr(0)}},
			wantSec: 0, wantOK: false,
		},
		{
			name:    "ttl_turns_positive",
			ctx:     &RequestContext{Retention: &config.DecisionRetention{TTLTurns: intPtr(3)}},
			wantSec: 3 * retentionDefaultSecondsPerTurn, wantOK: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sec, ok := retentionTTLOverrideSeconds(tc.ctx)
			if sec != tc.wantSec || ok != tc.wantOK {
				t.Fatalf("got (%d,%v), want (%d,%v)", sec, ok, tc.wantSec, tc.wantOK)
			}
		})
	}
}

func TestApplyRetentionTTLOverride(t *testing.T) {
	const base = 42
	if got := applyRetentionTTLOverride(base, &RequestContext{}); got != base {
		t.Fatalf("no retention: got %d, want base %d", got, base)
	}
	ctx := &RequestContext{Retention: &config.DecisionRetention{TTLTurns: intPtr(2)}}
	if got := applyRetentionTTLOverride(base, ctx); got != base {
		t.Fatalf("ttl_turns longer than base: got %d, want base %d", got, base)
	}
	if got := applyRetentionTTLOverride(-1, ctx); got != 2*retentionDefaultSecondsPerTurn {
		t.Fatalf("ttl_turns over a store without expiry: got %d, want %d", got, 2*retentionDefaultSecondsPerTurn)
	}
}

func TestUpdateResponseCacheAppliesRetentionTTL(t *testing.T) {
	mockCache, router, decision := cacheRouterForDecision(retentionCacheDecision("cache-decision", true))
	ctx := withSelectedDecision(&RequestContext{
		RequestID:       "req-retention-ttl",
		RequestModel:    "test-model",
		RequestQuery:    "hello",
		SemanticRequest: testNeutralRequest("test-model", "hello"),
		Retention:       &config.DecisionRetention{TTLTurns: intPtr(2)},
	}, decision)

	router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
	if !mockCache.addEntryCalled {
		t.Fatalf("ttl_turns must still write the cache entry")
	}
	if mockCache.lastTTLSeconds != 2*retentionDefaultSecondsPerTurn {
		t.Fatalf("non-streaming ttl_turns override: got %d, want %d", mockCache.lastTTLSeconds, 2*retentionDefaultSecondsPerTurn)
	}
}

// ttl_turns shortens a cache write's lifetime and never lengthens it: the
// entry lives the least of the plugin's ttl_seconds, the caller's ttl and
// ttl_turns, and a zero from the plugin or the caller still stores nothing.
func TestUpdateResponseCacheRetentionTTLOnlyShortensTheLifetime(t *testing.T) {
	for _, tt := range []struct {
		name       string
		pluginTTL  *int
		callerTTL  *int
		wantStored bool
		wantTTL    int
	}{
		{name: "caller ttl shorter than ttl_turns", pluginTTL: intPtr(60), callerTTL: intPtr(5), wantStored: true, wantTTL: 5},
		{name: "plugin ttl shorter than ttl_turns", pluginTTL: intPtr(60), wantStored: true, wantTTL: 60},
		{name: "ttl_turns under a store without expiry", wantStored: true, wantTTL: 10 * retentionDefaultSecondsPerTurn},
		{name: "caller no-store", pluginTTL: intPtr(60), callerTTL: intPtr(0)},
		{name: "plugin no-store", pluginTTL: intPtr(0)},
		{name: "plugin no-store with a caller ttl", pluginTTL: intPtr(0), callerTTL: intPtr(5)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			decision := retentionCacheDecision("cache-decision", true)
			if tt.pluginTTL != nil {
				decision.Plugins[0].Configuration = config.MustStructuredPayload(map[string]interface{}{"enabled": true, "ttl_seconds": *tt.pluginTTL})
			}
			mockCache, router, selected := cacheRouterForDecision(decision)
			ctx := withSelectedDecision(&RequestContext{
				RequestID:            "req-retention-min-ttl",
				RequestModel:         "test-model",
				RequestQuery:         "hello",
				SemanticRequest:      testNeutralRequest("test-model", "hello"),
				CacheWriteTTLSeconds: tt.callerTTL,
				Retention:            &config.DecisionRetention{TTLTurns: intPtr(10)},
			}, selected)

			router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
			if mockCache.addEntryCalled != tt.wantStored || mockCache.lastTTLSeconds != tt.wantTTL {
				t.Fatalf("stored=%v ttl=%d, want stored=%v ttl=%d", mockCache.addEntryCalled, mockCache.lastTTLSeconds, tt.wantStored, tt.wantTTL)
			}
		})
	}
}

// A plugin without ttl_seconds stores for the store's ttl_seconds, and a
// caller's ttl and ttl_turns only shorten that: with a 60 s store, a caller
// ttl=600 or ttl_turns: 10 (600 s) still leaves an entry the store expires
// within 60 s.
func TestUpdateResponseCacheTTLNeverOutlivesTheStoreTTL(t *testing.T) {
	callerTTL := 600
	for name, request := range map[string]*RequestContext{
		"caller ttl": {CacheWriteTTLSeconds: &callerTTL},
		"ttl_turns":  {Retention: &config.DecisionRetention{TTLTurns: intPtr(10)}},
	} {
		t.Run(name, func(t *testing.T) {
			backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, MaxEntries: 8, TTLSeconds: 60, EmbeddingProvider: storagetest.Vectors{Size: 8}})
			t.Cleanup(func() { _ = backend.Close() })
			_, router, decision := cacheRouterForDecision(retentionCacheDecision("cache-decision", true))
			router.Cache = backend
			router.Config.SemanticCache.TTLSeconds = 60
			request.RequestID, request.RequestModel, request.RequestQuery = "req-store-ttl", "test-model", "hello"
			request.SemanticRequest = testNeutralRequest("test-model", "hello")
			ctx := withSelectedDecision(request, decision)

			router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
			identity := responseCacheIdentity(ctx, ctx.RequestModel)
			recall, err := router.responseCacheService().LookupSemantic(context.Background(),
				cache.SemanticLookup{Identity: identity, Bounds: cache.SimilarityBounds{MinSimilarity: -1, TopK: 1}})
			if err != nil || len(recall.Candidates) != 1 {
				t.Fatalf("recalled %d entries, err %v", len(recall.Candidates), err)
			}
			if left := time.Until(recall.Candidates[0].ExpiresAt); left > time.Minute {
				t.Fatalf("the entry expires in %v, past the store's 60 s", left.Round(time.Second))
			}
		})
	}
}

func retentionCacheDecision(name string, cacheEnabled bool) config.Decision {
	decision := config.Decision{
		Name:      name,
		ModelRefs: []config.ModelRef{{Model: "test-model"}},
	}
	if cacheEnabled {
		decision.Plugins = []config.DecisionPlugin{{
			Type:          config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
		}}
	}
	return decision
}

func assertLogField(t *testing.T, fields map[string]interface{}, key string, want interface{}) {
	t.Helper()
	got, ok := fields[key]
	if !ok {
		t.Fatalf("expected field %q to be present in %v", key, fields)
	}
	if got != want {
		t.Fatalf("field %q = %v (%T), want %v (%T)", key, got, got, want, want)
	}
}

func assertFieldAbsent(t *testing.T, fields map[string]interface{}, key string) {
	t.Helper()
	if _, ok := fields[key]; ok {
		t.Fatalf("expected field %q to be absent from %v", key, fields)
	}
}
