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
