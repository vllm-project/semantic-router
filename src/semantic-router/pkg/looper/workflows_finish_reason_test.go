package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkflowsPreservesFinalBackendFinishReason(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		mode := "json"
		if streaming {
			mode = "sse"
		}
		for _, reason := range []string{"stop", "length", "content_filter"} {
			t.Run(mode+"/"+reason, func(t *testing.T) {
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					payload := decodeWorkflowRequestPayload(t, r)
					calls.Add(1)
					finish, content := "stop", "worker evidence"
					if payload.Model == "final-model" {
						finish, content = reason, "final response prefix"
					}
					w.Header().Set("Content-Type", "application/json")
					if err := json.NewEncoder(w).Encode(map[string]interface{}{
						"id": "completion", "object": "chat.completion", "model": payload.Model,
						"choices": []map[string]interface{}{{
							"index": 0, "finish_reason": finish,
							"message": map[string]interface{}{"role": "assistant", "content": content},
						}},
					}); err != nil {
						t.Errorf("write backend response: %v", err)
					}
				}))
				defer server.Close()
				loop := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL})
				defer loop.Close()
				response, err := loop.Execute(context.Background(), &Request{
					OriginalRequest: workflowTestRequest(),
					ModelRefs:       []config.ModelRef{{Model: "worker-model"}, {Model: "final-model"}},
					IsStreaming:     streaming,
					Algorithm: &config.AlgorithmConfig{Type: "workflows", Workflows: &config.WorkflowsAlgorithmConfig{
						Mode:  config.WorkflowModeStatic,
						Roles: []config.WorkflowRoleConfig{{Name: "worker", Models: []string{"worker-model"}}},
						Final: config.WorkflowFinalConfig{Model: "final-model"},
					}},
				})
				if err != nil {
					t.Fatalf("Execute: %v", err)
				}
				if calls.Load() != 2 {
					t.Fatalf("backend calls = %d, want one worker and one final call", calls.Load())
				}
				if !streaming {
					if got := workflowChoiceFinishReason(t, response.Body); got != reason {
						t.Fatalf("finish_reason = %v, want backend %s", got, reason)
					}
					return
				}
				var terminal []string
				for _, line := range strings.Split(string(response.Body), "\n") {
					if !strings.HasPrefix(line, "data: {") {
						continue
					}
					var chunk struct {
						Choices []struct {
							FinishReason *string `json:"finish_reason"`
						} `json:"choices"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
						t.Fatal(err)
					}
					for _, choice := range chunk.Choices {
						if choice.FinishReason != nil {
							terminal = append(terminal, *choice.FinishReason)
						}
					}
				}
				if len(terminal) != 1 || terminal[0] != reason {
					t.Fatalf("SSE terminal reasons = %v, want [%s]", terminal, reason)
				}
				if !strings.HasSuffix(string(response.Body), "data: [DONE]\n\n") {
					t.Fatal("SSE completion is missing its terminal marker")
				}
			})
		}
	}
}
