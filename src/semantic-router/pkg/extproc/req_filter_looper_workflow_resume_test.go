package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/alicebob/miniredis/v2"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestHandleLooperExecution_TwoIndependentRequestsPauseResume(t *testing.T) {
	t.Parallel()

	for _, backend := range []string{
		config.WorkflowStateBackendMemory,
		config.WorkflowStateBackendFile,
		config.WorkflowStateBackendRedis,
	} {
		backend := backend
		t.Run(backend, func(t *testing.T) {
			t.Parallel()
			server, tracker := newWorkflowPauseResumeServer(t)
			router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, backend))

			pauseResp := routeWorkflowRequest(t, router, workflowPauseChatBody(t))
			if got := immediateStatus(pauseResp); got != 200 {
				t.Fatalf("pause status = %d, body %s", got, immediateBody(pauseResp))
			}
			pauseBody := immediateBody(pauseResp)
			if strings.Contains(string(pauseBody), `"flow"`) {
				t.Fatalf("client pause body leaked looper flow extension: %s", pauseBody)
			}
			resumeResp := routeWorkflowRequest(t, router, workflowResumeChatBody(t, pauseBody))
			if got := immediateStatus(resumeResp); got != 200 {
				t.Fatalf("resume status = %d, body %s", got, immediateBody(resumeResp))
			}
			if !tracker.sawToolResult() {
				t.Fatal("resume request did not include the tool result")
			}
			if !tracker.sawFinal() {
				t.Fatal("final synthesis was not called after resume")
			}
			if !strings.Contains(string(immediateBody(resumeResp)), "final answer after tool") {
				t.Fatalf("resume body missing final answer: %s", immediateBody(resumeResp))
			}
		})
	}
}

func TestHandleLooperExecution_WorkflowConcurrentTakeExactlyOnce(t *testing.T) {
	t.Parallel()

	server, _ := newWorkflowPauseResumeServer(t)
	router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, config.WorkflowStateBackendMemory))
	pauseResp := routeWorkflowRequest(t, router, workflowPauseChatBody(t))
	resumeBody := workflowResumeChatBody(t, immediateBody(pauseResp))

	results := make([]int, 2)
	errs := make([]error, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	for i := 0; i < 2; i++ {
		i := i
		go func() {
			defer wg.Done()
			resp, err := routeWorkflowRequestErr(router, resumeBody)
			errs[i] = err
			if err == nil {
				results[i] = immediateStatus(resp)
			}
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("resume %d: %v", i, err)
		}
	}

	ok, failed := 0, 0
	for _, status := range results {
		switch status {
		case 200:
			ok++
		case 500:
			failed++
		default:
			t.Fatalf("unexpected resume status %d", status)
		}
	}
	if ok != 1 || failed != 1 {
		t.Fatalf("concurrent take statuses = %v, want one 200 and one 500", results)
	}
}

type workflowResumeTracker struct {
	mu                  sync.Mutex
	workerSawToolResult bool
	finalCalled         bool
}

func (tr *workflowResumeTracker) sawToolResult() bool {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return tr.workerSawToolResult
}

func (tr *workflowResumeTracker) sawFinal() bool {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return tr.finalCalled
}

func newWorkflowPauseResumeServer(t *testing.T) (*httptest.Server, *workflowResumeTracker) {
	t.Helper()
	tracker := &workflowResumeTracker{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model    string                   `json:"model"`
			Messages []map[string]interface{} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "worker-model":
			if workflowPayloadHasToolMessage(payload.Messages) {
				tracker.mu.Lock()
				tracker.workerSawToolResult = true
				tracker.mu.Unlock()
				_, _ = w.Write(workflowJSONChatCompletion("worker-model", "worker completed with tool result"))
				return
			}
			_, _ = w.Write(workflowJSONToolCallCompletion("worker-model"))
		case "verifier-model":
			tracker.mu.Lock()
			tracker.finalCalled = true
			tracker.mu.Unlock()
			_, _ = w.Write(workflowJSONChatCompletion("verifier-model", "final answer after tool"))
		default:
			http.Error(w, "unexpected model "+payload.Model, http.StatusInternalServerError)
		}
	}))
	t.Cleanup(server.Close)
	return server, tracker
}

func newWorkflowLooperConfig(t *testing.T, endpoint, backend string) config.LooperConfig {
	t.Helper()
	cfg := config.LooperConfig{
		Endpoint: endpoint,
		Flow: config.FlowRuntimeConfig{
			State: config.WorkflowStateRuntimeConfig{
				StoreBackend: backend,
				TTLSeconds:   60,
			},
		},
	}
	switch backend {
	case config.WorkflowStateBackendFile:
		cfg.Flow.State.File.Directory = t.TempDir()
	case config.WorkflowStateBackendRedis:
		mr := miniredis.RunT(t)
		cfg.Flow.State.Redis.Address = mr.Addr()
		cfg.Flow.State.Redis.KeyPrefix = "extproc-wf:"
		cfg.Flow.State.Redis.PoolSize = 4
	}
	return cfg
}

func newWorkflowRouter(t *testing.T, looperCfg config.LooperConfig) *OpenAIRouter {
	t.Helper()
	cfg := &config.RouterConfig{
		Looper: looperCfg,
		Memory: config.MemoryConfig{AutoStore: false},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{workflowTestDecision()},
		},
	}
	svc := looper.NewWorkflowStateService(&cfg.Looper)
	resources := newResourceScope()
	resources.add(svc.Close)
	router := &OpenAIRouter{
		Config:               cfg,
		WorkflowStateService: svc,
		ProtocolCodecs:       protocolcodec.NewBuiltinRegistry(),
		resources:            resources,
	}
	t.Cleanup(func() { _ = router.Close() })
	return router
}

func routeWorkflowRequest(t *testing.T, router *OpenAIRouter, body []byte) *ext_proc.ProcessingResponse {
	t.Helper()
	resp, err := routeWorkflowRequestErr(router, body)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func routeWorkflowRequestErr(router *OpenAIRouter, body []byte) (*ext_proc.ProcessingResponse, error) {
	request, err := decodeWorkflowChatRequest(body)
	if err != nil {
		return nil, err
	}
	decision := workflowTestDecision()
	ctx := &RequestContext{
		RequestID:           "workflow-independent-turn",
		TraceContext:        context.Background(),
		SourceFormat:        llmprotocol.OpenAIChatV1,
		VSRSelectedDecision: &decision,
	}
	resp, err := router.handleLooperExecution(context.Background(), request, &decision, ctx)
	if err != nil {
		return nil, err
	}
	if resp == nil || resp.GetImmediateResponse() == nil {
		return nil, fmt.Errorf("expected ImmediateResponse, got %#v", resp)
	}
	return resp, nil
}

func decodeWorkflowChatRequest(body []byte) (*llmprotocol.Request, error) {
	engine, err := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		return nil, err
	}
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		return nil, err
	}
	return &request, nil
}

func workflowPauseChatBody(t *testing.T) []byte {
	t.Helper()
	return []byte(`{
		"model":"MoM",
		"messages":[{"role":"user","content":"Use the lookup tool, then answer."}],
		"tools":[{
			"type":"function",
			"function":{
				"name":"lookup",
				"description":"Look up a value.",
				"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
			}
		}],
		"tool_choice":"auto"
	}`)
}

func workflowResumeChatBody(t *testing.T, pauseBody []byte) []byte {
	t.Helper()
	assistant, toolCallID := assistantToolMessageFromImmediateBody(t, pauseBody)
	body := map[string]interface{}{
		"model": "MoM",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "Use the lookup tool, then answer."},
			assistant,
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": toolCallID,
				"content":      `{"value":"42"}`,
			},
		},
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "lookup",
					"description": "Look up a value.",
					"parameters": map[string]interface{}{
						"type":       "object",
						"properties": map[string]interface{}{"query": map[string]interface{}{"type": "string"}},
						"required":   []interface{}{"query"},
					},
				},
			},
		},
		"tool_choice": "auto",
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal resume request: %v", err)
	}
	return data
}

func assistantToolMessageFromImmediateBody(t *testing.T, body []byte) (map[string]interface{}, string) {
	t.Helper()
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("parse pause body: %v", err)
	}
	choices, _ := parsed["choices"].([]interface{})
	if len(choices) == 0 {
		t.Fatalf("pause body missing choices: %s", body)
	}
	message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	rawToolCalls, _ := message["tool_calls"].([]interface{})
	if len(rawToolCalls) == 0 {
		t.Fatalf("pause body missing tool_calls: %s", body)
	}
	toolCallID, _ := rawToolCalls[0].(map[string]interface{})["id"].(string)
	if toolCallID == "" {
		t.Fatalf("pause body missing tool_call id: %s", body)
	}
	if !strings.HasPrefix(toolCallID, "flowtool_") {
		t.Fatalf("pause tool_call id %q missing workflow state prefix", toolCallID)
	}
	return message, toolCallID
}

func immediateBody(resp *ext_proc.ProcessingResponse) []byte {
	return resp.GetImmediateResponse().GetBody()
}

func immediateStatus(resp *ext_proc.ProcessingResponse) int {
	return int(resp.GetImmediateResponse().GetStatus().GetCode())
}

func workflowPayloadHasToolMessage(messages []map[string]interface{}) bool {
	for _, message := range messages {
		if role, _ := message["role"].(string); role == "tool" {
			return true
		}
	}
	return false
}

func workflowJSONChatCompletion(model, content string) []byte {
	body := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 0,
		"model":   model,
		"choices": []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": content,
			},
			"finish_reason": "stop",
		}},
	}
	data, _ := json.Marshal(body)
	return data
}

func workflowJSONToolCallCompletion(model string) []byte {
	return []byte(`{
		"id":"chatcmpl-tool-worker",
		"object":"chat.completion",
		"created":0,
		"model":"` + model + `",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":null,
				"tool_calls":[{
					"id":"call_lookup",
					"type":"function",
					"function":{"name":"lookup","arguments":"{\"query\":\"flow\"}"}
				}]
			},
			"finish_reason":"tool_calls"
		}]
	}`)
}
