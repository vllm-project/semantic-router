package looper

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestReMoMDistributeRoundRobin(t *testing.T) {
	calls := distributeRoundRobin(5, []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
		{Model: "model-c"},
	})

	require.Len(t, calls, 5)
	assert.Equal(t, []string{"model-a", "model-b", "model-c", "model-a", "model-b"}, modelNames(calls))
}

func TestReMoMDistributeByWeightUsesModelRefWeights(t *testing.T) {
	calls := distributeByWeight(6, []config.ModelRef{
		{Model: "model-a", Weight: 2},
		{Model: "model-b", Weight: 1},
	}, 42)

	require.Len(t, calls, 6)
	assert.Equal(t, map[string]int{"model-a": 4, "model-b": 2}, modelCounts(calls))
}

func TestReMoMDistributeByWeightFallsBackToEqualWhenUnset(t *testing.T) {
	calls := distributeByWeight(5, []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
	}, 42)

	require.Len(t, calls, 5)
	assert.Equal(t, map[string]int{"model-a": 3, "model-b": 2}, modelCounts(calls))
}

func TestReMoMDistributeCallsToModelsAcceptsRoundRobinStrategy(t *testing.T) {
	looper := NewReMoMLooper(&config.LooperConfig{})
	cfg := &config.ReMoMAlgorithmConfig{
		ModelDistribution: remomDistributionRoundRobin,
	}

	calls := looper.distributeCallsToModels(cfg, 4, []config.ModelRef{
		{Model: "model-a", LoRAName: "lora-a"},
		{Model: "model-b"},
	})

	require.Len(t, calls, 4)
	assert.Equal(t, []string{"model-a", "model-b", "model-a", "model-b"}, modelNames(calls))
	assert.Equal(t, "lora-a", calls[0].LoRAName)
	assert.Equal(t, "lora-a", calls[2].LoRAName)
}

func TestReMoMCollectParallelResultsReturnsAfterQuorum(t *testing.T) {
	results := make(chan remomParallelResult, 3)
	results <- remomParallelResult{resp: &ModelResponse{Content: "a", Model: "model-a"}}
	results <- remomParallelResult{resp: &ModelResponse{Content: "b", Model: "model-b"}}

	responses, err := collectRemomParallelResults(
		context.Background(),
		3,
		2,
		results,
		&config.ReMoMAlgorithmConfig{OnError: config.ReMoMOnErrorSkip},
	)

	require.NoError(t, err)
	require.Len(t, responses, 2)
	assert.Equal(t, []string{"model-a", "model-b"}, responseModelNames(responses))
}

func TestReMoMCollectParallelResultsReturnsPartialOnTimeoutWhenSkippingErrors(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()

	results := make(chan remomParallelResult, 3)
	results <- remomParallelResult{resp: &ModelResponse{Content: "a", Model: "model-a"}}

	responses, err := collectRemomParallelResults(
		ctx,
		3,
		2,
		results,
		&config.ReMoMAlgorithmConfig{OnError: config.ReMoMOnErrorSkip},
	)

	require.Error(t, err)
	require.Len(t, responses, 1)
	assert.Equal(t, "model-a", responses[0].Model)
}

func TestReMoMParallelMinSuccessfulDefaultsToAllCalls(t *testing.T) {
	assert.Equal(t, 3, remomParallelMinSuccessful(3, 0))
	assert.Equal(t, 3, remomParallelMinSuccessful(3, 4))
	assert.Equal(t, 2, remomParallelMinSuccessful(3, 2))
}

func TestReMoMFinalRoundModelCallsUsesSynthesisModel(t *testing.T) {
	defaultCalls := []ModelCall{{Model: "model-a"}}
	calls := remomFinalRoundModelCalls(
		&config.ReMoMAlgorithmConfig{SynthesisModel: "model-b"},
		defaultCalls,
		[]config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b", LoRAName: "synthesis-lora"},
		},
	)

	require.Len(t, calls, 1)
	assert.Equal(t, "model-b", calls[0].Model)
	assert.Equal(t, "synthesis-lora", calls[0].LoRAName)
}

func TestReMoMFinalRoundModelCallsFallsBackToDistribution(t *testing.T) {
	defaultCalls := []ModelCall{{Model: "model-a"}}
	calls := remomFinalRoundModelCalls(&config.ReMoMAlgorithmConfig{}, defaultCalls, nil)

	assert.Equal(t, defaultCalls, calls)
}

func TestCanFallbackToPreviousReMoMRoundRequiresSkipAndPriorResponse(t *testing.T) {
	cfg := &config.ReMoMAlgorithmConfig{OnError: config.ReMoMOnErrorSkip}
	rounds := []RoundResponse{{
		Round: 1,
		Responses: []IntermediateResp{{
			Model:   "model-a",
			Content: "ANSWER: A",
		}},
	}}

	assert.True(t, canFallbackToPreviousReMoMRound(cfg, rounds))
	assert.False(t, canFallbackToPreviousReMoMRound(&config.ReMoMAlgorithmConfig{OnError: config.ReMoMOnErrorFail}, rounds))
	assert.False(t, canFallbackToPreviousReMoMRound(cfg, nil))
	assert.False(t, canFallbackToPreviousReMoMRound(cfg, []RoundResponse{{Round: 1}}))
}

func modelNames(calls []ModelCall) []string {
	names := make([]string, 0, len(calls))
	for _, call := range calls {
		names = append(names, call.Model)
	}
	return names
}

func responseModelNames(responses []*ModelResponse) []string {
	names := make([]string, 0, len(responses))
	for _, response := range responses {
		names = append(names, response.Model)
	}
	return names
}

func modelCounts(calls []ModelCall) map[string]int {
	counts := make(map[string]int, len(calls))
	for _, call := range calls {
		counts[call.Model]++
	}
	return counts
}
