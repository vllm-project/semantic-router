package services

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func TestBuildEvalResponseDefaultsExecutionAlgorithmToStatic(t *testing.T) {
	service := &ClassificationService{}
	decisionResult := &decision.DecisionResult{Decision: &config.Decision{Name: "direct"}}
	response := service.buildEvalResponse(
		"hello",
		&classification.SignalResults{Metrics: &classification.SignalMetricsCollection{}},
		decisionResult,
	)

	require.NotNil(t, response.DecisionResult)
	assert.Equal(t, config.DecisionAlgorithmStatic, response.DecisionResult.Algorithm)
	assert.Empty(t, response.DecisionResult.Plugins)
}
