package extproc

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	replaystore "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// TestALooperRequestRecordsItsTemplateNodeDurations serves a Looper decision
// through the ext_proc adapter and reads the request-graph node durations its
// built-in template recorded.
func TestALooperRequestRecordsItsTemplateNodeDurations(t *testing.T) {
	var fixture looperFixture
	for _, candidate := range looperFixtures {
		if candidate.name == "confidence" {
			fixture = candidate
		}
	}
	observations := func(nodeType string) uint64 {
		histogram, ok := metrics.RequestGraphNodeDuration.WithLabelValues(nodeType, "looper.confidence").(prometheus.Metric)
		require.True(t, ok)
		var sample dto.Metric
		require.NoError(t, histogram.Write(&sample))
		return sample.GetHistogram().GetSampleCount()
	}
	algorithm, responded := observations(looper.StepType), observations(graph.TypeRespond)

	harness := newLooperHarness(t, fixture)
	record := harness.serve(t, looperAdapters["extproc"](harness.router), fixture.request)

	require.Equal(t, 200, record.Response.Status)
	assert.Equal(t, uint64(1), observations(looper.StepType)-algorithm, "the algorithm step")
	assert.Equal(t, uint64(1), observations(graph.TypeRespond)-responded, "the respond step")
}

func TestRecordSuccessfulLooperExecutionRecordsAggregateSessionUsageWithoutModelPricing(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"synthesizer": {
				Pricing: config.ModelPricing{Currency: "USD", PromptPer1M: 100, CompletionPer1M: 200},
			},
		}},
	}}
	// The production caller applies the looper's routing facts before reaching
	// this recorder, so the test establishes the same precondition rather than
	// having the recorder re-apply them.
	ctx := &RequestContext{
		RequestID:    "req-looper-session",
		SessionID:    "session-looper",
		RequestModel: "synthesizer",
	}
	decision := &config.Decision{Name: "panel"}
	response := &looper.Response{
		Model:         "synthesizer",
		AlgorithmType: config.DecisionAlgorithmFusion,
		Usage: looper.TokenUsage{
			PromptTokens:     1_000,
			CompletionTokens: 200,
			TotalTokens:      1_200,
		},
	}

	router.recordSuccessfulLooperExecution(response, "vllm-sr/auto", decision, ctx, nil, nil)

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("session-looper", time.Now())
	require.True(t, ok)
	assert.Equal(t, "synthesizer", snapshot.CurrentModel)
	assert.Equal(t, int64(1_000), snapshot.CumulativePromptTokens)
	assert.Equal(t, int64(200), snapshot.CumulativeCompletionTokens)
	assert.Zero(t, snapshot.CumulativeCost, "aggregate Looper usage must not inherit final-model pricing")
}

func TestLooperResponseUsageMetricsPreservesCacheAndCompleteness(t *testing.T) {
	complete := looperResponseUsageMetrics(looper.TokenUsage{
		PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120,
		CachedInputTokens: 40, CacheWriteTokens: 10,
	})
	assert.False(t, complete.invalid)
	assert.Equal(t, 40, complete.cachedPromptTokens)
	assert.True(t, complete.cachedPromptTokensReported)
	assert.Equal(t, 10, complete.cacheWriteTokens)
	assert.True(t, complete.cacheWriteTokensReported)
	assert.True(t, complete.completionTokensReported)

	unreported := looperResponseUsageMetrics(looper.TokenUsage{PromptTokens: 100, Unreported: true})
	assert.True(t, unreported.invalid)
	assert.Equal(t, "looper_usage_unreported", unreported.invalidReason)
	assert.Zero(t, unreported.promptTokens)

	incomplete := looperResponseUsageMetrics(looper.TokenUsage{
		PromptTokens: 100, CompletionTokens: 20, TotalTokens: 999,
		CachedInputTokens: 40,
	})
	assert.True(t, incomplete.invalid)
	assert.Equal(t, "looper_usage_incomplete", incomplete.invalidReason)
	assert.Equal(t, 40, incomplete.cachedPromptTokens)

	empty := looperResponseUsageMetrics(looper.TokenUsage{})
	assert.False(t, empty.invalid)
	assert.True(t, empty.completionTokensReported)
	assert.Zero(t, empty.completionTokens)
}

func TestRecordSuccessfulLooperExecutionRecordsCacheTokenBuckets(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{RequestID: "req-looper-cache", SessionID: "session-cache", RequestModel: "synthesizer"}
	router.recordSuccessfulLooperExecution(&looper.Response{
		Model: "synthesizer",
		Usage: looper.TokenUsage{
			PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120,
			CachedInputTokens: 40, CacheWriteTokens: 10,
		},
	}, "vllm-sr/auto", &config.Decision{Name: "panel"}, ctx, nil, nil)

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("session-cache", time.Now())
	require.True(t, ok)
	assert.Equal(t, int64(100), snapshot.CumulativePromptTokens)
	assert.Equal(t, int64(40), snapshot.CumulativeCachedTokens)
	assert.Equal(t, int64(10), snapshot.CumulativeCacheWriteTokens)
	assert.Equal(t, int64(20), snapshot.CumulativeCompletionTokens)
}

func TestRecordSuccessfulLooperExecutionSkipsUnreportedUsageTotals(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{RequestID: "req-looper-unreported", SessionID: "session-unreported", RequestModel: "synthesizer"}
	router.recordSuccessfulLooperExecution(&looper.Response{
		Model: "synthesizer",
		Usage: looper.TokenUsage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120, Unreported: true},
	}, "vllm-sr/auto", &config.Decision{Name: "panel"}, ctx, nil, nil)

	_, ok := sessiontelemetry.GetRouterSessionSnapshot("session-unreported", time.Now())
	assert.False(t, ok, "unreported looper usage must not create complete-looking session totals")
}

func TestRecordSuccessfulLooperExecutionRecordsReportedZeroOutput(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		RequestID: "req-looper-empty", SessionID: "session-empty", RequestModel: "synthesizer",
		VSRProgressGateConfig: &config.ProgressGateConfig{Enabled: true, WindowSize: 8, WindowTTLSeconds: 900},
	}
	router.recordSuccessfulLooperExecution(&looper.Response{
		Model: "synthesizer",
		Usage: looper.TokenUsage{},
	}, "vllm-sr/auto", &config.Decision{Name: "panel"}, ctx, nil, nil)

	window := sessiontelemetry.RecentTurnOutcomes(routingLearningStateKey(ctx), time.Now())
	require.Len(t, window, 1)
	assert.Equal(t, sessiontelemetry.TurnNoProgress, window[0].Category)
}

func TestUpdateLooperReplayUsageCopiesCacheBuckets(t *testing.T) {
	recorder := routerreplay.NewRecorder(replaystore.NewMemoryStore(10, 0))
	id, err := recorder.AddRecord(routerreplay.RoutingRecord{RequestID: "replay-looper"})
	require.NoError(t, err)

	router := &OpenAIRouter{ReplayRecorder: recorder}
	ctx := &RequestContext{RouterReplayID: id, RouterReplayRecorder: recorder}
	router.updateLooperReplayUsage(ctx, looper.TokenUsage{
		PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120,
		CachedInputTokens: 40, CacheWriteTokens: 10,
	})

	record, ok := recorder.GetRecord(id)
	require.True(t, ok)
	require.NotNil(t, record.CachedPromptTokens)
	require.NotNil(t, record.CacheWriteTokens)
	assert.Equal(t, 40, *record.CachedPromptTokens)
	assert.Equal(t, 10, *record.CacheWriteTokens)
	assert.Equal(t, 120, *record.TotalTokens)
}

func TestUpdateLooperReplayUsageSkipsUnreportedAggregates(t *testing.T) {
	recorder := routerreplay.NewRecorder(replaystore.NewMemoryStore(10, 0))
	id, err := recorder.AddRecord(routerreplay.RoutingRecord{RequestID: "replay-unreported"})
	require.NoError(t, err)

	router := &OpenAIRouter{ReplayRecorder: recorder}
	ctx := &RequestContext{RouterReplayID: id, RouterReplayRecorder: recorder}
	router.updateLooperReplayUsage(ctx, looper.TokenUsage{
		PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120, Unreported: true,
	})

	record, ok := recorder.GetRecord(id)
	require.True(t, ok)
	assert.Nil(t, record.TotalTokens)
	assert.Nil(t, record.CachedPromptTokens)
}

func TestUpdateLooperReplayUsageSkipsIncompleteAggregates(t *testing.T) {
	recorder := routerreplay.NewRecorder(replaystore.NewMemoryStore(10, 0))
	id, err := recorder.AddRecord(routerreplay.RoutingRecord{RequestID: "replay-incomplete"})
	require.NoError(t, err)

	router := &OpenAIRouter{ReplayRecorder: recorder}
	ctx := &RequestContext{RouterReplayID: id, RouterReplayRecorder: recorder}
	router.updateLooperReplayUsage(ctx, looper.TokenUsage{
		PromptTokens: 100, CompletionTokens: 20, TotalTokens: 999,
	})

	record, ok := recorder.GetRecord(id)
	require.True(t, ok)
	assert.Nil(t, record.PromptTokens)
	assert.Nil(t, record.CompletionTokens)
	assert.Nil(t, record.TotalTokens)
	assert.Nil(t, record.CachedPromptTokens)
	assert.Nil(t, record.CacheWriteTokens)
}
