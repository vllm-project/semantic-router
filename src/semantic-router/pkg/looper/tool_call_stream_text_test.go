package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const textBesideToolCall = `{"id":"c","object":"chat.completion","created":0,"model":"%s","choices":[{"index":0,"message":{"role":"assistant",` +
	`"content":"I will check the forecast for Paris first.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"flow\"}"}}]},` +
	`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// A streamed answer that calls a tool keeps the text the model wrote beside
// the call, as the JSON answer does.
func TestStreamedToolCallAnswersKeepTheirText(t *testing.T) {
	var tools openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`), &tools))
	for name, run := range map[string]func(url string) *Response{
		"fusion": func(url string) *Response {
			req := newFusionTestRequest()
			req.OriginalRequest.Tools = tools.Tools
			req.IsStreaming = true
			req.Algorithm = &config.AlgorithmConfig{Type: "fusion", Fusion: &config.FusionAlgorithmConfig{
				Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeNone,
			}}
			resp, err := fusionHopsTo(url).Execute(context.Background(), req)
			require.NoError(t, err)
			return resp
		},
		"flow": func(url string) *Response {
			req := workflowToolLooperRequest(workflowToolTestRequest())
			req.IsStreaming = true
			resp, err := NewWorkflowsLooper(workflowToolLooperConfig(url, t.TempDir())).Execute(context.Background(), req)
			require.NoError(t, err)
			return resp
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload struct {
					Model string `json:"model"`
				}
				require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
				w.Header().Set("Content-Type", "application/json")
				if payload.Model == "judge" || payload.Model == "worker-model" {
					_, _ = w.Write([]byte(strings.ReplaceAll(textBesideToolCall, "%s", payload.Model)))
					return
				}
				_, _ = w.Write(workflowChatCompletion(payload.Model, "a panel answer"))
			}))
			defer server.Close()

			body := string(run(server.URL).ProtocolBody())
			require.Contains(t, body, `"tool_calls"`)
			require.Contains(t, body, "I will check the forecast for Paris first.")
		})
	}
}
