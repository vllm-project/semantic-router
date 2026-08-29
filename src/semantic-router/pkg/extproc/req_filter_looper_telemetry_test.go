package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

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
	ctx := &RequestContext{
		RequestID: "req-looper-session",
		SessionID: "session-looper",
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

	router.recordSuccessfulLooperExecution(response, "auto", decision, ctx)

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("session-looper", time.Now())
	require.True(t, ok)
	assert.Equal(t, "synthesizer", snapshot.CurrentModel)
	assert.Equal(t, int64(1_000), snapshot.CumulativePromptTokens)
	assert.Equal(t, int64(200), snapshot.CumulativeCompletionTokens)
	assert.Zero(t, snapshot.CumulativeCost, "aggregate Looper usage must not inherit final-model pricing")
}
