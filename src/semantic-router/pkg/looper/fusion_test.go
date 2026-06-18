package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFusionLooperExecutesPanelJudgeAndFinal(t *testing.T) {
	var seenModels []string
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		seenModels = append(seenModels, model)
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b":
			return "panel b answer", http.StatusOK
		case "judge":
			if strings.Contains(prompt, "return only valid JSON") {
				return `{"consensus":["both answer"],"contradictions":[],"partial_coverage":[],"unique_insights":["b"],"blind_spots":[]}`, http.StatusOK
			}
			assert.Contains(t, prompt, "Structured analysis:")
			return "final fusion answer", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	require.Equal(t, "application/json", resp.ContentType)
	assert.Equal(t, "fusion", resp.AlgorithmType)
	assert.Equal(t, "judge", resp.Model)
	assert.Equal(t, []string{"panel-a", "panel-b", "judge"}, resp.ModelsUsed)
	assert.Equal(t, 4, resp.Iterations)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b", "judge", "judge"}, seenModels)
	expectedUsage := TokenUsage{PromptTokens: 90, CompletionTokens: 13, TotalTokens: 103}
	assert.Equal(t, expectedUsage, resp.Usage)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	assert.Equal(t, "judge", body["model"])
	var usageBody struct {
		Usage TokenUsage `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(resp.Body, &usageBody))
	assert.Equal(t, expectedUsage, usageBody.Usage)
	fusionTrace := body["fusion"].(map[string]interface{})
	assert.Len(t, fusionTrace["responses"], 2)
	assert.Equal(t, "judge", fusionTrace["judge_model"])
}

func TestFusionLooperRecordsPartialPanelFailures(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b":
			return "failed", http.StatusBadGateway
		case "judge":
			if strings.Contains(prompt, "return only valid JSON") {
				return `{"consensus":["a"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":["b failed"]}`, http.StatusOK
			}
			return "final from partial panel", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, []string{"panel-a", "panel-b", "judge"}, resp.ModelsUsed)
	assert.Equal(t, 4, resp.Iterations)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	fusionTrace := body["fusion"].(map[string]interface{})
	require.Len(t, fusionTrace["failed_models"], 1)
	failed := fusionTrace["failed_models"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, "panel-b", failed["model"])
}

type fusionToolCallObservation struct {
	model         string
	prompt        string
	hasTools      bool
	toolChoice    string
	hasToolResult bool
}

func TestFusionLooperAllowsFinalJudgeToolCallsOnly(t *testing.T) {
	var (
		mu           sync.Mutex
		observations []fusionToolCallObservation
	)
	server := newFusionToolCallServer(t, func(observation fusionToolCallObservation) {
		mu.Lock()
		observations = append(observations, observation)
		mu.Unlock()
	})
	defer server.Close()

	var params openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{
		"model":"vllm-sr/fusion",
		"messages":[
			{"role":"user","content":"search before answering"},
			{"role":"assistant","content":null,"tool_calls":[{
				"id":"call_existing",
				"type":"function",
				"function":{"name":"search","arguments":"{\"query\":\"existing\"}"}
			}]},
			{"role":"tool","tool_call_id":"call_existing","content":"existing search result"}
		],
		"tools":[{
			"type":"function",
			"function":{
				"name":"search",
				"description":"Search the web",
				"parameters":{"type":"object","properties":{"query":{"type":"string"}}}
			}
		}],
		"tool_choice":"auto"
	}`), &params))
	req := &Request{
		OriginalRequest: &params,
		DecisionName:    "fusion-test",
		Algorithm: &config.AlgorithmConfig{
			Type: "fusion",
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
			},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "application/json", resp.ContentType)
	assert.Equal(t, "fusion", resp.AlgorithmType)

	mu.Lock()
	defer mu.Unlock()
	assertFusionToolCallObservations(t, observations, 4)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	choices := body["choices"].([]interface{})
	message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	require.Len(t, message["tool_calls"], 1)
	assert.Equal(t, "tool_calls", choices[0].(map[string]interface{})["finish_reason"])
	assert.Contains(t, body, "fusion")

	var usageBody struct {
		Usage TokenUsage `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(resp.Body, &usageBody))
	assert.Equal(t, TokenUsage{PromptTokens: 90, CompletionTokens: 13, TotalTokens: 103}, usageBody.Usage)
}

func TestFusionLooperStreamsFinalJudgeToolCalls(t *testing.T) {
	server := newFusionToolCallServer(t, nil)
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a"},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "text/event-stream", resp.ContentType)
	body := string(resp.Body)
	assert.Contains(t, body, `"tool_calls"`)
	assert.Contains(t, body, `"finish_reason":"tool_calls"`)
	assert.Contains(t, body, `"fusion"`)
	assert.Contains(t, body, "data: [DONE]")
}

func newFusionToolCallServer(
	t *testing.T,
	observe func(fusionToolCallObservation),
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "1", r.Header.Get("x-vsr-fusion-depth"))
		payload := decodeFusionToolCallPayload(t, r)
		observation := fusionToolCallObservationFromPayload(payload)
		if observe != nil {
			observe(observation)
		}
		writeFusionToolCallFixtureResponse(t, w, observation.model, observation.prompt)
	}))
}

func decodeFusionToolCallPayload(t *testing.T, r *http.Request) map[string]interface{} {
	t.Helper()
	var payload map[string]interface{}
	require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
	return payload
}

func fusionToolCallObservationFromPayload(payload map[string]interface{}) fusionToolCallObservation {
	model, _ := payload["model"].(string)
	tools, _ := payload["tools"].([]interface{})
	toolChoice, _ := payload["tool_choice"].(string)
	observation := fusionToolCallObservation{
		model:      model,
		hasTools:   len(tools) > 0,
		toolChoice: toolChoice,
	}
	messages, ok := payload["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		return observation
	}
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		content, _ := message["content"].(string)
		if role == "tool" && strings.Contains(content, "existing search result") {
			observation.hasToolResult = true
		}
	}
	if message, ok := messages[len(messages)-1].(map[string]interface{}); ok {
		observation.prompt, _ = message["content"].(string)
	}
	return observation
}

func writeFusionToolCallFixtureResponse(t *testing.T, w http.ResponseWriter, model string, prompt string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if model == "judge" && strings.Contains(prompt, "Final answer:") {
		_ = json.NewEncoder(w).Encode(fusionToolCallCompletion(model))
		return
	}

	content := "panel answer"
	if model == "judge" && strings.Contains(prompt, "return only valid JSON") {
		content = `{"consensus":["panel"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`
	}
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 1,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": content},
				"finish_reason": "stop",
			},
		},
		"usage": fusionTestUsage(model),
	})
}

func fusionToolCallCompletion(model string) map[string]interface{} {
	return map[string]interface{}{
		"id":      "chatcmpl-tool-call",
		"object":  "chat.completion",
		"created": 1,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": nil,
					"tool_calls": []map[string]interface{}{
						{
							"id":   "call_search",
							"type": "function",
							"function": map[string]interface{}{
								"name":      "search",
								"arguments": `{"query":"fusion"}`,
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
		"usage": fusionTestUsage(model),
	}
}

func assertFusionToolCallObservations(t *testing.T, observations []fusionToolCallObservation, expected int) {
	t.Helper()
	require.Len(t, observations, expected)
	finalJudgeWithTools := false
	for _, got := range observations {
		assert.True(t, got.hasToolResult, "%s should preserve prior tool results", got.model)
		if got.model == "judge" && strings.Contains(got.prompt, "Final answer:") {
			assert.True(t, got.hasTools)
			assert.Equal(t, "auto", got.toolChoice)
			finalJudgeWithTools = true
			continue
		}
		assert.False(t, got.hasTools, "%s should not receive tools for prompt %q", got.model, got.prompt)
		assert.Empty(t, got.toolChoice)
	}
	assert.True(t, finalJudgeWithTools)
}

func TestFusionLooperAllPanelFailuresReturnError(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "failed", http.StatusBadGateway
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "all 2 analysis models failed")
}

func TestFusionLooperUsesDecisionModelRefs(t *testing.T) {
	looper := NewFusionLooper(&config.LooperConfig{Endpoint: "http://looper"})
	req := newFusionTestRequest()
	req.ModelRefs = []config.ModelRef{
		{Model: "panel-a"},
		{Model: "panel-b"},
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model: "judge",
		},
	}

	cfg := looper.resolveFusionExecutionConfig(req)

	assert.Equal(t, "judge", cfg.Model)
	assert.Equal(t, []string{"panel-a", "panel-b"}, cfg.AnalysisModels)
}

func TestFusionLooperRejectsInvalidOnError(t *testing.T) {
	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a"},
			OnError:        "ignore",
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: "http://looper"}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fusion on_error must be")
}

func TestFusionAnalysisPromptRequestsCompactJSON(t *testing.T) {
	prompt := buildFusionAnalysisPrompt(fusionExecutionConfig{}, "question", []*ModelResponse{
		{Model: "panel-a", Content: "answer"},
	})

	assert.Contains(t, prompt, "return only valid JSON")
	assert.Contains(t, prompt, "Return compact JSON only")
	assert.Contains(t, prompt, "no markdown")
	assert.Contains(t, prompt, "no code fences")
	assert.Contains(t, prompt, "at most two concise strings")
}

func TestParseFusionAnalysisAcceptsFencedJSON(t *testing.T) {
	analysis, err := parseFusionAnalysis("```json\n{\"consensus\":[\"agree\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]}\n```")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"agree"}, analysis.Consensus)
	assert.False(t, analysis.ParseFailed)
}

func newFusionTestRequest() *Request {
	params := openai.ChatCompletionNewParams{
		Model: "vllm-sr/fusion",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("compare the options"),
		},
	}
	return &Request{
		OriginalRequest: &params,
		DecisionName:    "fusion-test",
	}
}

func newFusionStubServer(
	t *testing.T,
	respond func(model string, prompt string) (content string, status int),
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "1", r.Header.Get("x-vsr-fusion-depth"))
		var payload struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		prompt := ""
		if len(payload.Messages) > 0 {
			prompt = payload.Messages[len(payload.Messages)-1].Content
		}
		content, status := respond(payload.Model, prompt)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1,
			"model":   payload.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": content,
					},
					"finish_reason": "stop",
				},
			},
			"usage": fusionTestUsage(payload.Model),
		})
	}))
}

func fusionTestUsage(model string) map[string]int64 {
	switch model {
	case "panel-a":
		return map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
	case "panel-b":
		return map[string]int64{"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}
	case "judge":
		return map[string]int64{"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34}
	default:
		return map[string]int64{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
	}
}
