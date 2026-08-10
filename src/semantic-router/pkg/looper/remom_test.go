package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go"
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

func TestReMoMShuffleModelCallsIsDeterministicForSeed(t *testing.T) {
	models := []string{"m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"}
	build := func() []ModelCall {
		calls := make([]ModelCall, len(models))
		for i, m := range models {
			calls[i] = ModelCall{Model: m}
		}
		return calls
	}

	a, b := build(), build()
	shuffleModelCalls(a, 42)
	shuffleModelCalls(b, 42)

	// The seeded shuffle must stay reproducible: the same seed always yields
	// the same permutation (this is the contract the ReMoM callers rely on).
	assert.Equal(t, modelNames(a), modelNames(b), "same seed must produce the same permutation")
	// And it must remain a permutation — no calls dropped or duplicated.
	assert.ElementsMatch(t, models, modelNames(a), "shuffle must preserve the multiset of calls")
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

func TestReMoMCollectParallelResultsCountsOnlyUsableReplies(t *testing.T) {
	results := make(chan remomParallelResult, 3)
	results <- remomParallelResult{resp: &ModelResponse{
		Model: "empty",
		Usage: TokenUsage{TotalTokens: 3},
	}}
	results <- remomParallelResult{resp: &ModelResponse{
		Model:            "reasoning",
		ReasoningContent: "internal evidence",
		Usage:            TokenUsage{TotalTokens: 5},
	}}
	results <- remomParallelResult{resp: &ModelResponse{
		Model:   "answer",
		Content: "public answer",
		Usage:   TokenUsage{TotalTokens: 7},
	}}

	responses, err := collectRemomParallelResults(
		context.Background(),
		3,
		2,
		results,
		&config.ReMoMAlgorithmConfig{OnError: config.ReMoMOnErrorSkip},
	)

	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Len(t, usableReMoMResponses(responses), 2)
	assert.Empty(t, responses[1].Content)
	assert.Equal(t, "internal evidence", responses[1].ReasoningContent)
	assert.Equal(t, int64(15), SumUsage(responses...).TotalTokens)
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

func TestReMoMInternalCallsAreNonStreamingAndUseCompletionLimit(t *testing.T) {
	var mu sync.Mutex
	var requests []remomTestRequestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		mu.Lock()
		requests = append(requests, payload)
		mu.Unlock()
		writeReMoMTestCompletion(t, w, payload.Model, "complete answer")
	}))
	defer server.Close()

	completionLimit := 640
	request := newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
		BreadthSchedule:     []int{1},
		ModelDistribution:   remomDistributionFirstOnly,
		MaxCompletionTokens: &completionLimit,
		OnError:             config.ReMoMOnErrorSkip,
	}, true, []config.ModelRef{{Model: "model-a"}})
	request.OriginalRequest.MaxTokens = openai.Int(128)
	request.OriginalRequest.MaxCompletionTokens = openai.Int(256)
	var toolRequest openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{
		"tools": [{
			"type": "function",
			"function": {
				"name": "search",
				"description": "Search",
				"parameters": {"type": "object"}
			}
		}],
		"tool_choice": "auto"
	}`), &toolRequest))
	request.OriginalRequest.Tools = toolRequest.Tools
	request.OriginalRequest.ToolChoice = toolRequest.ToolChoice

	response, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		request,
	)

	require.NoError(t, err)
	require.NotNil(t, response)
	assert.Equal(t, "text/event-stream", response.ContentType)
	require.Len(t, requests, 2)
	for _, request := range requests {
		require.NotNil(t, request.Stream)
		assert.False(t, *request.Stream)
		require.NotNil(t, request.MaxCompletionTokens)
		assert.Equal(t, int64(640), *request.MaxCompletionTokens)
		assert.Nil(t, request.MaxTokens, "ReMoM max_completion_tokens must override legacy max_tokens")
		assert.Empty(t, request.Tools, "ReMoM private rounds must not expose callable schemas")
		assert.Nil(t, request.ToolChoice, "ReMoM private rounds must not expose tool_choice")
	}
}

func TestReMoMWithoutCompletionLimitPreservesRequestTokenField(t *testing.T) {
	var requests []remomTestRequestPayload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, decodeReMoMTestRequest(t, r))
		writeReMoMTestCompletion(t, w, "model-a", "complete answer")
	}))
	defer server.Close()

	request := newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
		BreadthSchedule:   []int{1},
		ModelDistribution: remomDistributionFirstOnly,
		OnError:           config.ReMoMOnErrorSkip,
	}, false, []config.ModelRef{{Model: "model-a"}})
	request.OriginalRequest.MaxTokens = openai.Int(320)

	_, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		request,
	)

	require.NoError(t, err)
	require.Len(t, requests, 2)
	for _, upstreamRequest := range requests {
		require.NotNil(t, upstreamRequest.MaxTokens)
		assert.Equal(t, int64(320), *upstreamRequest.MaxTokens)
		assert.Nil(t, upstreamRequest.MaxCompletionTokens)
	}
}

func TestReMoMUsesReasoningOnlyReplyForSynthesisWithoutPublishingIt(t *testing.T) {
	const privateReasoning = "private chain of thought"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		if payload.Model == "model-reasoning" {
			writeReMoMTestCompletionWithReasoning(t, w, payload.Model, "", privateReasoning, TokenUsage{})
			return
		}
		require.Contains(t, lastReMoMTestMessageContent(payload), privateReasoning)
		writeReMoMTestCompletion(t, w, payload.Model, "public synthesized answer")
	}))
	defer server.Close()

	response, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
			BreadthSchedule:              []int{1},
			ModelDistribution:            remomDistributionRoundRobin,
			SynthesisModel:               "model-final",
			IncludeIntermediateResponses: false,
			OnError:                      config.ReMoMOnErrorSkip,
		}, false, []config.ModelRef{
			{Model: "model-reasoning"},
			{Model: "model-final"},
		}),
	)

	require.NoError(t, err)
	assert.Contains(t, string(response.Body), "public synthesized answer")
	assert.NotContains(t, string(response.Body), privateReasoning)
}

func TestReMoMUsageIncludesUnusableAttemptedReply(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		switch payload.Model {
		case "model-empty":
			writeReMoMTestCompletionWithReasoning(t, w, payload.Model, "", "", TokenUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3})
		case "model-answer":
			time.Sleep(20 * time.Millisecond)
			writeReMoMTestCompletionWithReasoning(t, w, payload.Model, "candidate", "", TokenUsage{PromptTokens: 4, CompletionTokens: 1, TotalTokens: 5})
		case "model-final":
			writeReMoMTestCompletionWithReasoning(t, w, payload.Model, "final answer", "", TokenUsage{PromptTokens: 6, CompletionTokens: 1, TotalTokens: 7})
		default:
			t.Fatalf("unexpected model %q", payload.Model)
		}
	}))
	defer server.Close()

	response, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
			BreadthSchedule:              []int{2},
			ModelDistribution:            remomDistributionRoundRobin,
			SynthesisModel:               "model-final",
			MinSuccessfulResponses:       1,
			IncludeIntermediateResponses: false,
			OnError:                      config.ReMoMOnErrorSkip,
		}, false, []config.ModelRef{
			{Model: "model-empty"},
			{Model: "model-answer"},
			{Model: "model-final"},
		}),
	)

	require.NoError(t, err)
	assert.Equal(t, TokenUsage{PromptTokens: 12, CompletionTokens: 3, TotalTokens: 15}, response.Usage)
}

func TestReMoMEmptyFinalFallsBackToBestPreviousOutput(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		contentByModel := map[string]string{
			"model-short":    "brief",
			"model-detailed": "more complete candidate answer",
			"model-final":    "",
		}
		writeReMoMTestCompletion(t, w, payload.Model, contentByModel[payload.Model])
	}))
	defer server.Close()

	response, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
			BreadthSchedule:   []int{2},
			ModelDistribution: remomDistributionRoundRobin,
			SynthesisModel:    "model-final",
			OnError:           config.ReMoMOnErrorSkip,
		}, false, []config.ModelRef{
			{Model: "model-short"},
			{Model: "model-detailed"},
			{Model: "model-final"},
		}),
	)

	require.NoError(t, err)
	var completion struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(response.Body, &completion))
	require.Len(t, completion.Choices, 1)
	assert.Equal(t, "model-detailed", completion.Model)
	assert.Equal(t, "more complete candidate answer", completion.Choices[0].Message.Content)
}

func TestReMoMAllEmptyOutputsReturnError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		writeReMoMTestCompletion(t, w, payload.Model, "")
	}))
	defer server.Close()

	response, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
			BreadthSchedule:   []int{1},
			ModelDistribution: remomDistributionFirstOnly,
			OnError:           config.ReMoMOnErrorSkip,
		}, false, []config.ModelRef{{Model: "model-a"}}),
	)

	require.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "no usable content or reasoning")
}

func TestReMoMReasoningOnlyOutputsReturnExplicitError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		writeReMoMTestCompletionWithReasoning(t, w, payload.Model, "", "private reasoning", TokenUsage{})
	}))
	defer server.Close()

	response, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
			BreadthSchedule:   []int{1},
			ModelDistribution: remomDistributionFirstOnly,
			OnError:           config.ReMoMOnErrorSkip,
		}, false, []config.ModelRef{{Model: "model-a"}}),
	)

	require.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "no assistant content")
}

type remomTestRequestPayload struct {
	Model               string                   `json:"model"`
	Stream              *bool                    `json:"stream"`
	MaxTokens           *int64                   `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int64                   `json:"max_completion_tokens,omitempty"`
	Messages            []map[string]interface{} `json:"messages"`
	Tools               []json.RawMessage        `json:"tools,omitempty"`
	ToolChoice          interface{}              `json:"tool_choice,omitempty"`
}

func decodeReMoMTestRequest(t *testing.T, r *http.Request) remomTestRequestPayload {
	t.Helper()
	var payload remomTestRequestPayload
	require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
	return payload
}

func writeReMoMTestCompletion(t *testing.T, w http.ResponseWriter, model, content string) {
	t.Helper()
	writeReMoMTestCompletionWithReasoning(t, w, model, content, "", TokenUsage{})
}

func writeReMoMTestCompletionWithReasoning(
	t *testing.T,
	w http.ResponseWriter,
	model string,
	content string,
	reasoning string,
	usage TokenUsage,
) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	message := map[string]interface{}{
		"role":    "assistant",
		"content": content,
	}
	if reasoning != "" {
		message["reasoning_content"] = reasoning
	}
	require.NoError(t, json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "chatcmpl-generic",
		"object":  "chat.completion",
		"created": 0,
		"model":   model,
		"choices": []map[string]interface{}{{
			"index":         0,
			"message":       message,
			"finish_reason": "stop",
		}},
		"usage": usage.Map(),
	}))
}

func lastReMoMTestMessageContent(payload remomTestRequestPayload) string {
	if len(payload.Messages) == 0 {
		return ""
	}
	content, _ := payload.Messages[len(payload.Messages)-1]["content"].(string)
	return content
}

func newReMoMTestRequest(cfg *config.ReMoMAlgorithmConfig, streaming bool, modelRefs []config.ModelRef) *Request {
	return &Request{
		OriginalRequest: &openai.ChatCompletionNewParams{
			Model:    "routing-alias",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("solve the task")},
		},
		ModelRefs:    modelRefs,
		Algorithm:    &config.AlgorithmConfig{Type: "remom", ReMoM: cfg},
		IsStreaming:  streaming,
		DecisionName: "generic-remom",
	}
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
