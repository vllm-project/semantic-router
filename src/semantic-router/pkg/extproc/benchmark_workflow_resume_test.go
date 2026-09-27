package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestBenchmarkWorkflowResumePreservesBindingAndFailsClosedForUntracedUsage(t *testing.T) {
	for _, tc := range []struct {
		name      string
		streaming bool
		cached    int64
	}{
		{name: "buffered_read_write", cached: 30},
		{name: "buffered_write_only"},
		{name: "streaming_read_write", streaming: true, cached: 30},
		{name: "streaming_write_only", streaming: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			streaming := tc.streaming
			hash := strings.Repeat("b", 64)
			var mu sync.Mutex
			var dispatched []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get(headers.SRBenchExpectedConfigHash); got != hash {
					t.Errorf("model dispatch config binding = %q, want %q", got, hash)
				}
				var payload struct {
					Model    string                   `json:"model"`
					Messages []map[string]interface{} `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Error(err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				mu.Lock()
				dispatched = append(dispatched, payload.Model)
				mu.Unlock()
				body := workflowJSONChatCompletion(payload.Model, "completed")
				if payload.Model == "worker-model" && !workflowPayloadHasToolMessage(payload.Messages) {
					body = workflowJSONToolCallCompletion(payload.Model)
				}
				var completion map[string]interface{}
				if err := json.Unmarshal(body, &completion); err != nil {
					t.Error(err)
					return
				}
				completion["usage"] = map[string]interface{}{
					"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
					"prompt_tokens_details": map[string]int64{"cached_tokens": tc.cached, "created_cache_tokens": 20},
				}
				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(completion); err != nil {
					t.Error(err)
				}
			}))
			defer server.Close()
			router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, config.WorkflowStateBackendMemory))
			router.Config.DocumentHash = hash
			decision := workflowTestDecision()
			decision.ModelRefs = append(decision.ModelRefs, config.ModelRef{Model: "prepare-model"})
			decision.Algorithm.Workflows.Roles = append([]config.WorkflowRoleConfig{{
				Name: "prepare", Models: []string{"prepare-model"}, Prompt: "Prepare the request.",
			}}, decision.Algorithm.Workflows.Roles...)
			route := func(body []byte) *ext_proc.ProcessingResponse {
				t.Helper()
				request, err := decodeWorkflowChatRequest(body)
				if err != nil {
					t.Fatal(err)
				}
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				reqCtx := &RequestContext{
					TraceContext: ctx, SourceFormat: llmprotocol.OpenAIChatV1,
					Headers:             map[string]string{headers.SRBenchExpectedConfigHash: hash},
					VSRSelectedDecision: &decision, ExpectStreamingResponse: streaming,
				}
				if precondition := router.benchmarkConfigPrecondition(reqCtx); precondition != nil {
					t.Fatal("valid config binding rejected")
				}
				resp, err := router.handleLooperExecution(ctx, request, &decision, reqCtx)
				if err != nil {
					t.Fatal(err)
				}
				if immediateStatus(resp) != http.StatusOK {
					t.Fatalf("workflow request failed: %s", immediateBody(resp))
				}
				router.bindBenchmarkConfigResponse(resp, reqCtx)
				return resp
			}
			pause := route(workflowChatBody(streaming))
			assertWorkflowClientTrace(t, immediateBody(pause), true)
			assertBenchmarkWorkflowReceipt(t, pause, hash, "worker-model")
			resume := route(workflowResumeChatBodyWithStream(t, immediateBody(pause), streaming))
			assertWorkflowClientTrace(t, immediateBody(resume), true)
			// Workflows do not currently emit attempt traces. Neither an initial nor
			// a resumed aggregate is proof of complete request-local accounting.
			assertBenchmarkWorkflowReceipt(t, resume, hash, "verifier-model")
			if !streaming {
				assertWorkflowBufferedUsage(t, pause, 2, tc.cached)
				assertWorkflowBufferedUsage(t, resume, 3, tc.cached)
			}
			mu.Lock()
			defer mu.Unlock()
			wantDispatch := []string{"prepare-model", "worker-model", "worker-model", "verifier-model"}
			if !reflect.DeepEqual(dispatched, wantDispatch) {
				t.Fatalf("dispatches = %v, want %v (no replayed preparation)", dispatched, wantDispatch)
			}
		})
	}
}

func assertWorkflowBufferedUsage(t *testing.T, resp *ext_proc.ProcessingResponse, cumulativeCalls, cached int64) {
	t.Helper()
	var body struct {
		Usage looper.TokenUsage `json:"usage"`
	}
	if err := json.Unmarshal(immediateBody(resp), &body); err != nil {
		t.Fatal(err)
	}
	usage := body.Usage
	if usage.PromptTokens != 100*cumulativeCalls || usage.CompletionTokens != 10*cumulativeCalls ||
		usage.TotalTokens != 110*cumulativeCalls || usage.CachedInputTokens != cached*cumulativeCalls ||
		usage.CacheWriteTokens != 20*cumulativeCalls {
		t.Fatalf("workflow aggregate lost cache buckets: %+v", usage)
	}
}

func TestBenchmarkUsageWorkflowHistoricalUsageFailsClosed(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{VSRSelectedDecision: &config.Decision{Algorithm: &config.AlgorithmConfig{Type: "workflows"}}}
	usage := looper.TokenUsage{PromptTokens: 100, CompletionTokens: 10, TotalTokens: 110, CachedInputTokens: 30, CacheWriteTokens: 20}
	resp := &looper.Response{
		Usage: usage.Add(&looper.ModelResponse{Usage: usage}),
		ExecutionTrace: looper.ExecutionTrace{Version: 1, Attempts: []looper.AttemptTrace{{
			Model: "worker-model", Status: looper.AttemptStatusSucceeded, Usage: usage,
		}}},
	}
	var receipt benchmarkUsageReceipt
	if err := json.Unmarshal([]byte(router.benchmarkLooperUsage(resp, ctx)), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Complete || len(receipt.Calls) != 1 || receipt.Calls[0].Usage != (benchmarkUsageBuckets{100, 10, 110, 30, 20}) {
		t.Fatalf("historical usage was counted as a new complete bill: %+v", receipt)
	}
}

func assertBenchmarkWorkflowReceipt(t *testing.T, resp *ext_proc.ProcessingResponse, hash, selected string) {
	t.Helper()
	values := make(map[string]string)
	for _, option := range resp.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		values[option.GetHeader().GetKey()] = string(option.GetHeader().GetRawValue())
	}
	if values[headers.VSRConfigHash] != hash || values[headers.VSRSelectedModel] != selected {
		t.Fatalf("incorrect generation/model identity: %v", values)
	}
	var receipt benchmarkUsageReceipt
	if err := json.Unmarshal([]byte(values[headers.VSRModelUsage]), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Version != 1 || receipt.Complete || len(receipt.Calls) != 0 {
		t.Fatalf("untraced workflow incorrectly claims complete accounting: %+v", receipt)
	}
}
