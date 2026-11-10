package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Each executor serves the backend finish_reason of the answer it publishes,
// so a reply cut by max_tokens or a content filter does not reach the client
// as a complete one.
func TestLooperExecutorsPreserveTheServedFinishReason(t *testing.T) {
	for _, executor := range []struct {
		name    string
		request func() *Request
		cut     string   // the model whose answers end with the tested reason
		served  []string // the models whose answers are served, by choice
	}{
		{"escalate", func() *Request {
			request := escalateRequest(config.EscalateOnErrorLastAttempt, config.ConfidenceMethodAverageLogprob)
			// Neither answer is accepted, so the last one is served.
			request.Algorithm.Escalate.Accept.Confidence.GTE = gte(0.99)
			return request
		}, "large", []string{"large"}},
		{"fusion", func() *Request {
			request := newFusionTestRequest()
			request.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
				Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeNone,
			}}
			return request
		}, "judge", []string{"judge"}},
		{"remom", func() *Request {
			return newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
				BreadthSchedule: []int{1}, ModelDistribution: remomDistributionFirstOnly,
			}, false, []config.ModelRef{{Model: "model-a"}})
		}, "model-a", []string{"model-a"}},
		{"ratings", func() *Request {
			return &Request{
				OriginalRequest: workflowTestRequest(),
				ModelRefs:       []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
				Algorithm:       &config.AlgorithmConfig{Type: config.DecisionAlgorithmRatings, Ratings: &config.RatingsAlgorithmConfig{}},
			}
		}, "model-b", []string{"model-a", "model-b"}},
	} {
		for _, streaming := range []bool{false, true} {
			// Ratings streams its candidates, which carry no parsed completion.
			if streaming && executor.name == "ratings" {
				continue
			}
			for _, reason := range []string{"length", "content_filter"} {
				t.Run(executor.name+"/"+map[bool]string{false: "json", true: "sse"}[streaming]+"/"+reason, func(t *testing.T) {
					server := newFinishReasonBackend(t, executor.cut, reason)
					defer server.Close()
					request := executor.request()
					request.IsStreaming = streaming
					loop, err := Factory(&config.LooperConfig{Endpoint: server.URL}, request.Algorithm.Type)
					if err != nil {
						t.Fatalf("Factory() error = %v", err)
					}
					defer loop.Close()
					response, err := loop.Execute(context.Background(), request)
					if err != nil {
						t.Fatalf("Execute() error = %v", err)
					}
					var want []string
					for _, model := range executor.served {
						want = append(want, map[bool]string{false: "stop", true: reason}[model == executor.cut])
					}
					if got := servedFinishReasons(t, response.ProtocolBody(), streaming); !slices.Equal(got, want) {
						t.Fatalf("finish reasons = %v, want %v", got, want)
					}
				})
			}
		}
	}
}

// newFinishReasonBackend answers "<model> answer" with one token logprob,
// ending the answers of the cut model with reason and every other with stop.
func newFinishReasonBackend(t *testing.T, cut, reason string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		finish := "stop"
		if request.Model == cut {
			finish = reason
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "chatcmpl-finish", "object": "chat.completion", "created": 0, "model": request.Model,
			"choices": []map[string]interface{}{{
				"index": 0, "finish_reason": finish,
				"message": map[string]interface{}{"role": "assistant", "content": request.Model + " answer"},
				"logprobs": map[string]interface{}{"content": []map[string]interface{}{{
					"token": "answer", "logprob": -0.1, "bytes": []int{97},
					"top_logprobs": []map[string]interface{}{{"token": "answer", "logprob": -0.1, "bytes": []int{97}}},
				}}},
			}},
			"usage": map[string]int{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
		})
	}))
}

// servedFinishReasons returns each choice's finish_reason: from the completion,
// or from the terminal chunks of a stream.
func servedFinishReasons(t *testing.T, body []byte, streaming bool) []string {
	t.Helper()
	type choice struct {
		FinishReason *string `json:"finish_reason"`
	}
	var chunks [][]choice
	if !streaming {
		var completion struct{ Choices []choice }
		if err := json.Unmarshal(body, &completion); err != nil {
			t.Fatalf("decode completion: %v", err)
		}
		chunks = append(chunks, completion.Choices)
	}
	for _, line := range strings.Split(string(body), "\n") {
		if !streaming || !strings.HasPrefix(line, "data: {") {
			continue
		}
		var chunk struct{ Choices []choice }
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("decode chunk: %v", err)
		}
		chunks = append(chunks, chunk.Choices)
	}
	var reasons []string
	for _, choices := range chunks {
		for _, choice := range choices {
			if choice.FinishReason != nil {
				reasons = append(reasons, *choice.FinishReason)
			}
		}
	}
	return reasons
}
