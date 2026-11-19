package services

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func assertIntentScore(t *testing.T, response *IntentResponse, category string, scored bool) {
	t.Helper()
	require.NotNil(t, response)
	require.Equal(t, category, response.Classification.Category)
	raw, err := json.Marshal(response)
	require.NoError(t, err)
	var decoded map[string]interface{}
	require.NoError(t, json.Unmarshal(raw, &decoded))
	result := decoded["classification"].(map[string]interface{})
	require.Equal(t, scored, decoded["probabilities_available"], string(raw))
	require.Equal(t, scored, result["confidence_available"], string(raw))
	if scored {
		require.Equal(t, 0.75, result["confidence"])
		require.Equal(t, map[string]interface{}{category: 0.75}, decoded["probabilities"])
	} else {
		require.Nil(t, result["confidence"], string(raw))
		require.NotContains(t, decoded, "probabilities", string(raw))
	}
}

func TestIntentUnrelatedSignalErrorPreservesDecisionScore(t *testing.T) {
	service := &ClassificationService{}
	signals := &classification.SignalResults{SignalErrors: map[string]string{"pii:email": "pii_evaluation_failed"}}
	result := &decision.DecisionResult{Decision: &config.Decision{Name: "math-route"}, Confidence: 0.75, ConfidenceScored: true}
	category := resolveIntentCategory(context.Background(), nil, result, signals, "")
	response := service.buildIntentResponseFromSignals(signals, result, category, IntentRequest{Options: &IntentOptions{ReturnProbabilities: true}}, nil, nil)
	assertIntentScore(t, response, "math-route", true)
	require.Equal(t, signals.SignalErrors, response.SignalErrors)
}
