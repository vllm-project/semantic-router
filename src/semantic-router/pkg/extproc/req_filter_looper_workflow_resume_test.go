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

func TestHandleLooperExecution_TwoRecipesSameDecisionDoNotShareState(t *testing.T) {
	t.Parallel()

	server, tracker := newWorkflowPauseResumeServer(t)
	router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, config.WorkflowStateBackendMemory))

	pauseResp := routeWorkflowRequestWith(t, router, workflowPauseChatBody(t), workflowRouteOptions{recipe: "recipe-a"})
	if got := immediateStatus(pauseResp); got != 200 {
		t.Fatalf("pause status = %d, body %s", got, immediateBody(pauseResp))
	}
	pauseBody := immediateBody(pauseResp)
	resumeBody := workflowResumeChatBody(t, pauseBody)

	cross := routeWorkflowRequestWith(t, router, resumeBody, workflowRouteOptions{recipe: "recipe-b"})
	if got := immediateStatus(cross); got != 500 {
		t.Fatalf("recipe B resume status = %d, want 500, body %s", got, immediateBody(cross))
	}
	if !strings.Contains(string(immediateBody(cross)), "not found or expired") &&
		!strings.Contains(string(immediateBody(cross)), "belongs to recipe") {
		t.Fatalf("recipe B resume body = %s", immediateBody(cross))
	}
	if tracker.sawToolResult() || tracker.sawFinal() {
		t.Fatal("recipe B consumed recipe A's workflow state")
	}

	resumeResp := routeWorkflowRequestWith(t, router, resumeBody, workflowRouteOptions{recipe: "recipe-a"})
	if got := immediateStatus(resumeResp); got != 200 {
		t.Fatalf("recipe A resume status = %d, body %s", got, immediateBody(resumeResp))
	}
	if !tracker.sawToolResult() || !tracker.sawFinal() {
		t.Fatal("recipe A resume did not finish the workflow")
	}
}

func TestHandleLooperExecution_WorkflowTraceAtClientBoundary(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		include bool
		stream  bool
	}{
		{name: "buffered_enabled", include: true, stream: false},
		{name: "buffered_disabled", include: false, stream: false},
		{name: "sse_enabled", include: true, stream: true},
		{name: "sse_disabled", include: false, stream: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			server, tracker := newWorkflowPauseResumeServer(t)
			router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, config.WorkflowStateBackendMemory))
			include := tc.include
			opts := workflowRouteOptions{streaming: tc.stream, includeIntermediate: &include}

			pauseBody := workflowPauseChatBody(t)
			resumeFn := workflowResumeChatBody
			if tc.stream {
				pauseBody = workflowStreamingPauseChatBody(t)
				resumeFn = workflowStreamingResumeChatBody
			}

			pauseResp := routeWorkflowRequestWith(t, router, pauseBody, opts)
			if got := immediateStatus(pauseResp); got != 200 {
				t.Fatalf("pause status = %d, body %s", got, immediateBody(pauseResp))
			}
			assertWorkflowClientTrace(t, immediateBody(pauseResp), tc.include)

			resumeResp := routeWorkflowRequestWith(t, router, resumeFn(t, immediateBody(pauseResp)), opts)
			if got := immediateStatus(resumeResp); got != 200 {
				t.Fatalf("resume status = %d, body %s", got, immediateBody(resumeResp))
			}
			assertWorkflowClientTrace(t, immediateBody(resumeResp), tc.include)
			if !tracker.sawToolResult() || !tracker.sawFinal() {
				t.Fatal("resume did not finish the workflow")
			}
		})
	}
}

func TestIsolateAndRestoreLooperWorkflowTrace(t *testing.T) {
	t.Parallel()

	buffered := []byte(`{"id":"c","object":"chat.completion","flow":{"steps":[{"responses":[{"agent_id":"worker:0:worker-model","content":"x"}]}]},"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`)
	sse := []byte("data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"flow\":{\"steps\":[{\"responses\":[{\"agent_id\":\"worker:0:worker-model\"}]}]},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: [DONE]\n")

	for _, tc := range []struct {
		name    string
		body    []byte
		include bool
		sse     bool
	}{
		{name: "buffered_enabled", body: buffered, include: true},
		{name: "buffered_disabled", body: buffered, include: false},
		{name: "sse_enabled", body: sse, include: true, sse: true},
		{name: "sse_disabled", body: sse, include: false, sse: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			stripped, flow := isolateLooperWorkflowFlow(tc.body)
			if strings.Contains(string(stripped), `"flow"`) {
				t.Fatalf("isolated body still contains flow: %s", stripped)
			}
			if len(flow) == 0 {
				t.Fatal("expected isolated flow payload")
			}
			if tc.sse && !strings.Contains(string(stripped), `"delta"`) {
				t.Fatalf("SSE lost completion payload: %s", stripped)
			}
			decision := workflowTestDecisionWithIntermediateResponses(tc.include)
			restored := restoreLooperWorkflowTrace(stripped, flow, &RequestContext{VSRSelectedDecision: &decision})
			assertWorkflowClientTrace(t, restored, tc.include)
		})
	}
}

func TestLooperClientResponseBody_StripsFlowFromSSE(t *testing.T) {
	t.Parallel()
	body := []byte("data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"flow\":{\"pending_tool_call\":{\"state_id\":\"abc\"}},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: [DONE]\n")
	stripped := looperClientResponseBody(body)
	if strings.Contains(string(stripped), `"flow"`) {
		t.Fatalf("SSE still contains flow: %s", stripped)
	}
	if !strings.Contains(string(stripped), `"delta"`) {
		t.Fatalf("SSE lost completion payload: %s", stripped)
	}
}

func TestHandleLooperExecution_TwoIndependentRequestsPauseResume(t *testing.T) {
	t.Parallel()

	for _, backend := range []string{
		config.WorkflowStateBackendMemory,
		config.WorkflowStateBackendFile,
		config.WorkflowStateBackendRedis,
	} {
		t.Run(backend, func(t *testing.T) {
			t.Parallel()
			server, tracker := newWorkflowPauseResumeServer(t)
			router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, backend))

			pauseResp := routeWorkflowRequest(t, router, workflowPauseChatBody(t))
			if got := immediateStatus(pauseResp); got != 200 {
				t.Fatalf("pause status = %d, body %s", got, immediateBody(pauseResp))
			}
			pauseBody := immediateBody(pauseResp)
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

type workflowRouteOptions struct {
	recipe              config.RecipeName
	streaming           bool
	includeIntermediate *bool
}

func routeWorkflowRequestWith(t *testing.T, router *OpenAIRouter, body []byte, opts workflowRouteOptions) *ext_proc.ProcessingResponse {
	t.Helper()
	resp, err := routeWorkflowRequestWithErr(router, body, opts)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func routeWorkflowRequestErr(router *OpenAIRouter, body []byte) (*ext_proc.ProcessingResponse, error) {
	return routeWorkflowRequestWithErr(router, body, workflowRouteOptions{})
}

func routeWorkflowRequestWithErr(router *OpenAIRouter, body []byte, opts workflowRouteOptions) (*ext_proc.ProcessingResponse, error) {
	request, err := decodeWorkflowChatRequest(body)
	if err != nil {
		return nil, err
	}
	include := true
	if opts.includeIntermediate != nil {
		include = *opts.includeIntermediate
	}
	decision := workflowTestDecisionWithIntermediateResponses(include)
	ctx := &RequestContext{
		RequestID:               "workflow-independent-turn",
		TraceContext:            context.Background(),
		SourceFormat:            llmprotocol.OpenAIChatV1,
		VSRSelectedDecision:     &decision,
		ExpectStreamingResponse: opts.streaming,
	}
	if opts.recipe != "" {
		ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: opts.recipe})
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
	return workflowChatBody(false)
}

func workflowStreamingPauseChatBody(t *testing.T) []byte {
	t.Helper()
	return workflowChatBody(true)
}

func workflowChatBody(stream bool) []byte {
	body := `{
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
		"tool_choice":"auto"`
	if stream {
		body += `,
		"stream":true`
	}
	return []byte(body + "\n\t}")
}

func workflowResumeChatBody(t *testing.T, pauseBody []byte) []byte {
	t.Helper()
	return workflowResumeChatBodyWithStream(t, pauseBody, false)
}

func workflowStreamingResumeChatBody(t *testing.T, pauseBody []byte) []byte {
	t.Helper()
	return workflowResumeChatBodyWithStream(t, pauseBody, true)
}

func workflowResumeChatBodyWithStream(t *testing.T, pauseBody []byte, stream bool) []byte {
	t.Helper()
	assistant, toolCallID := assistantToolMessageFromClientBody(t, pauseBody)
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
	if stream {
		body["stream"] = true
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal resume request: %v", err)
	}
	return data
}

func assistantToolMessageFromClientBody(t *testing.T, body []byte) (map[string]interface{}, string) {
	t.Helper()
	trimmed := strings.TrimSpace(string(body))
	if strings.HasPrefix(trimmed, "data:") || strings.Contains(string(body), "\ndata:") {
		return assistantToolMessageFromSSEBody(t, body)
	}
	return assistantToolMessageFromImmediateBody(t, body)
}

func assistantToolMessageFromSSEBody(t *testing.T, body []byte) (map[string]interface{}, string) {
	t.Helper()
	var toolCalls []interface{}
	for _, line := range strings.Split(string(body), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("parse SSE chunk %q: %v", payload, err)
		}
		choices, _ := chunk["choices"].([]interface{})
		if len(choices) == 0 {
			continue
		}
		choice, _ := choices[0].(map[string]interface{})
		delta, _ := choice["delta"].(map[string]interface{})
		if delta == nil {
			continue
		}
		if raw, ok := delta["tool_calls"].([]interface{}); ok && len(raw) > 0 {
			toolCalls = raw
		}
	}
	if len(toolCalls) == 0 {
		t.Fatalf("streaming pause missing tool_calls: %s", body)
	}
	toolCall, _ := toolCalls[0].(map[string]interface{})
	toolCallID, _ := toolCall["id"].(string)
	if toolCallID == "" {
		t.Fatalf("streaming pause missing tool_call id: %s", body)
	}
	if !strings.HasPrefix(toolCallID, "flowtool_") {
		t.Fatalf("pause tool_call id %q missing workflow state prefix", toolCallID)
	}
	delete(toolCall, "index")
	return map[string]interface{}{
		"role":       "assistant",
		"content":    nil,
		"tool_calls": toolCalls,
	}, toolCallID
}

func assertWorkflowClientTrace(t *testing.T, body []byte, wantPresent bool) {
	t.Helper()
	flow, ok := workflowClientFlow(body)
	if !wantPresent {
		if ok {
			t.Fatalf("client body should omit flow: %s", body)
		}
		return
	}
	if !ok {
		t.Fatalf("client body missing documented flow trace: %s", body)
	}
	if !workflowFlowHasAgentID(flow) {
		t.Fatalf("flow.steps[].responses[].agent_id missing: %#v", flow)
	}
}

func workflowClientFlow(body []byte) (map[string]interface{}, bool) {
	if isLooperSSEBody(body) {
		for _, line := range strings.Split(string(body), "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "" || payload == "[DONE]" {
				continue
			}
			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}
			flow, ok := chunk["flow"].(map[string]interface{})
			if ok {
				return flow, true
			}
		}
		return nil, false
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, false
	}
	flow, ok := parsed["flow"].(map[string]interface{})
	return flow, ok
}

func workflowFlowHasAgentID(flow map[string]interface{}) bool {
	steps, _ := flow["steps"].([]interface{})
	for _, step := range steps {
		stepMap, _ := step.(map[string]interface{})
		responses, _ := stepMap["responses"].([]interface{})
		for _, response := range responses {
			respMap, _ := response.(map[string]interface{})
			if id, _ := respMap["agent_id"].(string); strings.TrimSpace(id) != "" {
				return true
			}
		}
	}
	return false
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
