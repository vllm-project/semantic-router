package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRatingsStreamingPreservesUTF8OverHTTP(t *testing.T) {
	answers := map[string]string{
		"chinese": strings.Repeat("中文回答", 31),
		"mixed":   strings.Repeat("a", 49) + "🚀你好🌍" + strings.Repeat("🧑🏽‍💻 café ", 14),
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(modelStreamFixture(answers[request.Model], true)))
	}))
	defer server.Close()
	l := NewRatingsLooper(&config.LooperConfig{Endpoint: server.URL})
	defer l.Close()
	req := usageBackendRequest(config.ModelRef{Model: "chinese"}, config.ModelRef{Model: "mixed"})
	req.IsStreaming = true
	out, err := l.Execute(context.Background(), req)
	require.NoError(t, err)
	content := make(map[int]string)
	for _, chunk := range streamJSONChunks(t, out.Body) {
		for _, value := range chunk["choices"].([]interface{}) {
			choice := value.(map[string]interface{})
			delta := choice["delta"].(map[string]interface{})
			if text, ok := delta["content"].(string); ok {
				content[int(choice["index"].(float64))] += text
			}
		}
	}
	require.Equal(t, answers["chinese"], content[0])
	require.Equal(t, answers["mixed"], content[1])
	require.NotContains(t, string(out.Body), "\\ufffd")
}

func TestRatingsStreamingHonorsErrorPolicyOverHTTP(t *testing.T) {
	for _, failure := range []string{"error_event", "incomplete", "malformed", "http_error"} {
		for _, onError := range []string{"fail", "skip"} {
			t.Run(failure+"_"+onError, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var request struct {
						Model string `json:"model"`
					}
					require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
					w.Header().Set("Content-Type", "text/event-stream")
					body := modelStreamFixture("complete answer", true)
					if request.Model == "broken" {
						if failure == "http_error" {
							http.Error(w, "backend unavailable", http.StatusServiceUnavailable)
							return
						}
						body = modelStreamFixture("partial answer", false)
						switch failure {
						case "error_event":
							body += "event: error\ndata: {\"error\":{\"type\":\"server_error\",\"message\":\"generation failed\"}}\n\n"
						case "malformed":
							body += "data: {\n\n"
						}
					}
					_, _ = w.Write([]byte(body))
				}))
				defer server.Close()
				l := NewRatingsLooper(&config.LooperConfig{Endpoint: server.URL})
				defer l.Close()
				req := usageBackendRequest(config.ModelRef{Model: "healthy"}, config.ModelRef{Model: "broken"})
				req.IsStreaming = true
				req.Algorithm = &config.AlgorithmConfig{Type: "ratings", Ratings: &config.RatingsAlgorithmConfig{OnError: onError}}
				out, err := l.Execute(context.Background(), req)
				if onError == "fail" {
					require.Error(t, err)
					require.Nil(t, out)
					return
				}
				require.NoError(t, err)
				require.Equal(t, []string{"healthy"}, out.ModelsUsed)
				require.Contains(t, string(out.Body), "complete answer")
				require.NotContains(t, string(out.Body), "partial answer")
			})
		}
	}
}

func TestConfidenceStreamingPreservesParallelToolsOverHTTP(t *testing.T) {
	calls := []map[string]interface{}{
		{"id": "call_weather", "type": "function", "function": map[string]interface{}{"name": "weather", "arguments": `{"city":"北京"}`}},
		{"id": "call_time", "type": "function", "function": map[string]interface{}{"name": "time", "arguments": `{"zone":"Asia/Shanghai"}`}},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Stream   bool `json:"stream"`
			Logprobs bool `json:"logprobs"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		require.False(t, request.Stream, "logprob assessment must receive a completion")
		require.True(t, request.Logprobs)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "chat-tools", "model": "small", "object": "chat.completion",
			"choices": []interface{}{map[string]interface{}{
				"index": 0, "finish_reason": "tool_calls",
				"message":  map[string]interface{}{"role": "assistant", "content": "查询中", "tool_calls": calls},
				"logprobs": map[string]interface{}{"content": []interface{}{map[string]interface{}{"token": "tool", "logprob": -0.01}}},
			}},
		})
	}))
	defer server.Close()
	l := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL})
	defer l.Close()
	req := confidenceLogprobRequest("fail")
	req.IsStreaming = true
	out, err := l.Execute(context.Background(), req)
	require.NoError(t, err)
	var streamed []interface{}
	var content, finish string
	for _, chunk := range streamJSONChunks(t, out.Body) {
		for _, value := range chunk["choices"].([]interface{}) {
			choice := value.(map[string]interface{})
			delta := choice["delta"].(map[string]interface{})
			if text, ok := delta["content"].(string); ok {
				content += text
			}
			if tools, ok := delta["tool_calls"].([]interface{}); ok {
				streamed = append(streamed, tools...)
			}
			if reason, ok := choice["finish_reason"].(string); ok {
				finish = reason
			}
		}
	}
	require.Equal(t, "查询中", content)
	require.Equal(t, "tool_calls", finish)
	require.Len(t, streamed, len(calls))
	for i, value := range streamed {
		call := value.(map[string]interface{})
		require.Equal(t, float64(i), call["index"])
		delete(call, "index")
		require.Equal(t, calls[i], call)
	}
}

func TestClientRejectsNonChatResponseShape(t *testing.T) {
	for _, body := range []string{`{}`, `{"choices":null}`, `{"object":"response","choices":[]}`, `{"error":{"message":"failed"}}`, `{"type":"message","content":[{"type":"text","text":"native"}]}`} {
		_, err := (&Client{}).parseNonStreamingResponse([]byte(body), "model")
		require.ErrorContains(t, err, "chat completion response")
	}
}

func modelStreamFixture(content string, complete bool) string {
	text, _ := json.Marshal(content)
	body := fmt.Sprintf("data: {\"id\":\"chat-fixture\",\"model\":\"backend\",\"choices\":[{\"index\":0,\"delta\":{\"content\":%s}}]}\n\n", text)
	if complete {
		body += "data: {\"id\":\"chat-fixture\",\"model\":\"backend\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	}
	return body
}

func streamJSONChunks(t *testing.T, body []byte) []map[string]interface{} {
	t.Helper()
	var chunks []map[string]interface{}
	for _, line := range strings.Split(string(body), "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(payload), &chunk))
		chunks = append(chunks, chunk)
	}
	return chunks
}

func TestBaseStreamingKeepsTaggedToolCompatibility(t *testing.T) {
	l := NewBaseLooper(&config.LooperConfig{})
	defer l.Close()
	text := `<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call>`
	response, err := l.formatStreamingResponse(&AggregatedResponse{
		Responses:       []*ModelResponse{{Raw: []byte(`{"choices":[{"message":{"content":"tagged tool"}}]}`)}},
		CombinedContent: text, FinalModel: "model",
	}, []string{"model"}, 1)
	require.NoError(t, err)
	require.Contains(t, string(response.Body), `"name":"weather"`)
	require.NotContains(t, string(response.Body), `"content"`)
}

func TestStreamingToolDefaultsDoNotMutateCandidateHistory(t *testing.T) {
	raw := []byte(`{"choices":[{"message":{"tool_calls":[{"function":{"name":"weather","arguments":""}},{"function":{"name":"time","arguments":"{}"}}]}}]}`)
	original := string(raw)
	agg := &AggregatedResponse{Responses: []*ModelResponse{{Raw: raw}}}
	calls, tagged := resolveToolCallsForStreaming(agg)
	require.False(t, tagged)
	require.Len(t, calls, 2)
	for index, call := range calls {
		require.Equal(t, index, call["index"])
		require.NotEmpty(t, call["id"])
		require.Equal(t, "function", call["type"])
		require.Equal(t, "{}", call["function"].(map[string]interface{})["arguments"])
	}
	require.NotEqual(t, calls[0]["id"], calls[1]["id"])
	require.Equal(t, original, string(agg.Responses[0].Raw))
}

func TestClientPreservesEmptyChatAccounting(t *testing.T) {
	for _, object := range []string{"", "chat.completion"} {
		body, err := json.Marshal(map[string]interface{}{
			"object": object, "choices": []interface{}{},
			"usage": map[string]int{"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
		})
		require.NoError(t, err)
		if object == "" {
			body = []byte(`{"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}`)
		}
		response, err := (&Client{}).parseNonStreamingResponse(body, "empty-model")
		require.NoError(t, err)
		require.Empty(t, response.Content)
		require.NotNil(t, response.Parsed)
		require.Equal(t, TokenUsage{PromptTokens: 20, CompletionTokens: 2, TotalTokens: 22}, response.Usage)
		require.Equal(t, body, response.Raw)
	}
}
