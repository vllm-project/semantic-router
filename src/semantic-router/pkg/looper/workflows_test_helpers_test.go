package looper

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func workflowTestRequest() *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Model: "vllm-sr/flow",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("solve this task"),
		},
	}
}

func workflowToolTestRequest() *openai.ChatCompletionNewParams {
	raw := []byte(`{
		"model":"vllm-sr/flow",
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
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(raw, &req); err != nil {
		panic(err)
	}
	return &req
}

func workflowMixedToolSchemasTestRequest() *openai.ChatCompletionNewParams {
	raw := []byte(`{
		"model":"vllm-sr/flow",
		"messages":[{"role":"user","content":"Use callable context when needed."}],
		"tools":[{
			"type":"function",
			"function":{
				"name":"lookup",
				"description":"Look up a value.",
				"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
			}
		}],
		"tool_choice":"auto",
		"functions":[{
			"name":"legacy_lookup",
			"description":"Legacy lookup.",
			"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
		}],
		"function_call":"auto"
	}`)
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(raw, &req); err != nil {
		panic(err)
	}
	return &req
}

func workflowLegacyFunctionTestRequest() *openai.ChatCompletionNewParams {
	raw := []byte(`{
		"model":"vllm-sr/flow",
		"messages":[{"role":"user","content":"Use the legacy function, then answer."}],
		"functions":[{
			"name":"legacy_lookup",
			"description":"Legacy lookup.",
			"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}
		}],
		"function_call":"auto"
	}`)
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(raw, &req); err != nil {
		panic(err)
	}
	return &req
}

func workflowToolResumeRequest(t *testing.T, assistant map[string]interface{}, toolCallID string) *openai.ChatCompletionNewParams {
	t.Helper()
	return workflowToolResumeRequestWithIDs(t, assistant, toolCallID)
}

func workflowToolResumeRequestWithIDs(t *testing.T, assistant map[string]interface{}, toolCallIDs ...string) *openai.ChatCompletionNewParams {
	t.Helper()
	if len(toolCallIDs) == 0 {
		t.Fatal("workflowToolResumeRequestWithIDs requires at least one tool_call_id")
	}
	toolMessages := make([]interface{}, 0, len(toolCallIDs))
	for _, toolCallID := range toolCallIDs {
		toolMessages = append(toolMessages, map[string]interface{}{
			"role":         "tool",
			"tool_call_id": toolCallID,
			"content":      `{"value":"42"}`,
		})
	}
	messages := []interface{}{
		map[string]interface{}{"role": "user", "content": "Use the lookup tool, then answer."},
		assistant,
	}
	messages = append(messages, toolMessages...)
	body := map[string]interface{}{
		"model":    "vllm-sr/flow",
		"messages": messages,
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
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &req); err != nil {
		t.Fatalf("parse resume request: %v", err)
	}
	return &req
}

func workflowToolLooperRequest(req *openai.ChatCompletionNewParams) *Request {
	includeTrace := true
	return &Request{
		OriginalRequest: req,
		ModelRefs: []config.ModelRef{
			{Model: "worker-model"},
			{Model: "verifier-model"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeStatic,
				Roles: []config.WorkflowRoleConfig{
					{Name: "worker", Models: []string{"worker-model"}, Prompt: "Use tools when needed, then solve."},
				},
				Final:                        config.WorkflowFinalConfig{Model: "verifier-model"},
				MaxSteps:                     2,
				MaxParallel:                  1,
				IncludeIntermediateResponses: &includeTrace,
			},
		},
		DecisionName: "tool-flow-test",
	}
}

func workflowParallelToolLooperRequest(req *openai.ChatCompletionNewParams) *Request {
	looperReq := workflowToolLooperRequest(req)
	looperReq.ModelRefs = []config.ModelRef{
		{Model: "worker-a"},
		{Model: "worker-b"},
		{Model: "verifier-model"},
	}
	looperReq.Algorithm.Workflows.Roles = []config.WorkflowRoleConfig{
		{Name: "workers", Models: []string{"worker-a", "worker-b"}, Prompt: "Use tools only if needed, then solve."},
	}
	looperReq.Algorithm.Workflows.MaxParallel = 2
	return looperReq
}

func workflowTwoStepToolLooperRequest(req *openai.ChatCompletionNewParams) *Request {
	includeTrace := true
	return &Request{
		OriginalRequest: req,
		ModelRefs: []config.ModelRef{
			{Model: "thinker-model"},
			{Model: "worker-model"},
			{Model: "verifier-model"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeStatic,
				Roles: []config.WorkflowRoleConfig{
					{Name: "thinker", Models: []string{"thinker-model"}, Prompt: "Prepare evidence."},
					{Name: "worker", Models: []string{"worker-model"}, Prompt: "Use tools when needed, then solve.", AccessList: []string{"thinker"}},
				},
				Final:                        config.WorkflowFinalConfig{Model: "verifier-model"},
				MaxSteps:                     3,
				MaxParallel:                  1,
				IncludeIntermediateResponses: &includeTrace,
			},
		},
		DecisionName: "two-step-tool-flow-test",
	}
}

func workflowAccessListToolLooperRequest(req *openai.ChatCompletionNewParams) *Request {
	includeTrace := true
	return &Request{
		OriginalRequest: req,
		ModelRefs: []config.ModelRef{
			{Model: "worker-a"},
			{Model: "worker-b"},
			{Model: "verifier-model"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeStatic,
				Roles: []config.WorkflowRoleConfig{
					{Name: "lookup-agent", Models: []string{"worker-a"}, Prompt: "Use tools when needed, then provide only the useful result."},
					{Name: "consumer", Models: []string{"worker-b"}, Prompt: "Use only accessible prior outputs, not private tool traces.", AccessList: []string{"lookup-agent"}},
				},
				Final:                        config.WorkflowFinalConfig{Model: "verifier-model"},
				MaxSteps:                     3,
				MaxParallel:                  1,
				IncludeIntermediateResponses: &includeTrace,
			},
		},
		DecisionName: "access-list-tool-flow-test",
	}
}

func workflowToolLooperConfig(endpoint string, stateDir string) *config.LooperConfig {
	return &config.LooperConfig{
		Endpoint: endpoint,
		Flow: config.FlowRuntimeConfig{
			State: config.WorkflowStateRuntimeConfig{
				StoreBackend: config.WorkflowStateBackendFile,
				TTLSeconds:   60,
				File: config.WorkflowStateFileConfig{
					Directory: stateDir,
				},
			},
		},
	}
}

func workflowToolCallCompletion(model string, toolCallID string) []byte {
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
					"id":"` + toolCallID + `",
					"type":"function",
					"function":{"name":"lookup","arguments":"{\"query\":\"flow\"}"}
				}]
			},
			"finish_reason":"tool_calls"
		}]
	}`)
}

func workflowMultipleToolCallCompletion(model string, toolCallIDs ...string) []byte {
	toolCalls := make([]map[string]interface{}, 0, len(toolCallIDs))
	for _, toolCallID := range toolCallIDs {
		toolCalls = append(toolCalls, map[string]interface{}{
			"id":   toolCallID,
			"type": "function",
			"function": map[string]interface{}{
				"name":      "lookup",
				"arguments": `{"query":"flow"}`,
			},
		})
	}
	body := map[string]interface{}{
		"id":      "chatcmpl-tool-worker",
		"object":  "chat.completion",
		"created": 0,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":       "assistant",
					"content":    nil,
					"tool_calls": toolCalls,
				},
				"finish_reason": "tool_calls",
			},
		},
	}
	data, _ := json.Marshal(body)
	return data
}

func workflowLegacyFunctionCallCompletion(model string) []byte {
	return []byte(`{
		"id":"chatcmpl-function-worker",
		"object":"chat.completion",
		"created":0,
		"model":"` + model + `",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":null,
				"function_call":{"name":"legacy_lookup","arguments":"{\"query\":\"flow\"}"}
			},
			"finish_reason":"function_call"
		}]
	}`)
}

func workflowChatCompletion(model string, content string) []byte {
	body := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 0,
		"model":   model,
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
	}
	data, _ := json.Marshal(body)
	return data
}

func assistantToolMessageFromResponse(t *testing.T, body []byte) (map[string]interface{}, string) {
	t.Helper()
	message, ids := assistantToolMessageIDsFromResponse(t, body)
	if len(ids) == 0 {
		t.Fatalf("response missing tool_calls: %s", string(body))
	}
	return message, ids[0]
}

func assistantToolMessageIDsFromResponse(t *testing.T, body []byte) (map[string]interface{}, []string) {
	t.Helper()
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("parse tool response: %v", err)
	}
	message := parsed["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})
	rawToolCalls := message["tool_calls"].([]interface{})
	ids := make([]string, 0, len(rawToolCalls))
	for _, rawToolCall := range rawToolCalls {
		ids = append(ids, rawToolCall.(map[string]interface{})["id"].(string))
	}
	return message, ids
}

func workflowStoredPendingState(t *testing.T, stateDir string, toolCallID string) workflowPendingToolState {
	t.Helper()
	stateID, ok := parseWorkflowToolStateID(toolCallID)
	if !ok {
		t.Fatalf("tool_call_id %q does not contain workflow state id", toolCallID)
	}
	data, err := os.ReadFile(filepath.Join(stateDir, stateID+".json"))
	if err != nil {
		t.Fatalf("read workflow state for %q: %v", stateID, err)
	}
	var state workflowPendingToolState
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatalf("parse workflow state for %q: %v", stateID, err)
	}
	return state
}

func assertWorkflowInitialPendingToolState(t *testing.T, state workflowPendingToolState) {
	t.Helper()
	if state.AgentID != "worker:0:worker-model" {
		t.Fatalf("first state agent_id = %q", state.AgentID)
	}
	if state.ToolCallSeq != 1 {
		t.Fatalf("first state tool_call_seq = %d, want 1", state.ToolCallSeq)
	}
	if len(state.ToolTrajectory) != 0 {
		t.Fatalf("first state trajectory = %d turns, want 0", len(state.ToolTrajectory))
	}
}

func assertWorkflowSecondPendingToolState(t *testing.T, state workflowPendingToolState, firstToolCallID string) {
	t.Helper()
	if state.ToolCallSeq != 2 {
		t.Fatalf("second state tool_call_seq = %d, want 2", state.ToolCallSeq)
	}
	if len(state.ToolTrajectory) != 1 {
		t.Fatalf("second state trajectory = %d turns, want 1", len(state.ToolTrajectory))
	}
	assertWorkflowToolTrajectoryTurn(t, state.ToolTrajectory[0], firstToolCallID)
}

func assertWorkflowToolTrajectoryTurn(t *testing.T, turn workflowAgentToolTurn, firstToolCallID string) {
	t.Helper()
	assertWorkflowToolTrajectoryTurnForAgent(t, turn, firstToolCallID, "worker:0:worker-model")
}

func assertWorkflowToolTrajectoryTurnForAgent(
	t *testing.T,
	turn workflowAgentToolTurn,
	firstToolCallID string,
	agentID string,
) {
	t.Helper()
	if turn.AgentID != agentID {
		t.Fatalf("trajectory agent_id = %q, want %q", turn.AgentID, agentID)
	}
	if !workflowContainsString(turn.ToolCallIDs, firstToolCallID) {
		t.Fatalf("trajectory tool ids = %v, want %q", turn.ToolCallIDs, firstToolCallID)
	}
	if len(turn.ToolMessages) != 1 {
		t.Fatalf("trajectory tool messages = %d, want 1", len(turn.ToolMessages))
	}
	if turn.ToolMessages[0]["tool_call_id"] != firstToolCallID {
		t.Fatalf("trajectory tool_call_id = %v, want %q", turn.ToolMessages[0]["tool_call_id"], firstToolCallID)
	}
}

func assertWorkflowTraceToolTrajectory(t *testing.T, body []byte, agentID string, toolCallID string) {
	t.Helper()
	response := workflowTraceResponse(t, body, agentID)
	rawTrajectory, ok := response["tool_trajectory"].([]interface{})
	if !ok || len(rawTrajectory) == 0 {
		t.Fatalf("trace response for %s missing tool_trajectory: %#v", agentID, response)
	}
	for _, rawTurn := range rawTrajectory {
		turn, ok := rawTurn.(map[string]interface{})
		if !ok {
			continue
		}
		rawIDs, ok := turn["tool_call_ids"].([]interface{})
		if !ok {
			continue
		}
		for _, rawID := range rawIDs {
			if rawID == toolCallID {
				return
			}
		}
	}
	t.Fatalf("trace response for %s missing tool_call_id %q: %#v", agentID, toolCallID, rawTrajectory)
}

func workflowTraceResponse(t *testing.T, body []byte, agentID string) map[string]interface{} {
	t.Helper()
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("parse workflow response: %v", err)
	}
	flow, ok := parsed["flow"].(map[string]interface{})
	if !ok {
		t.Fatalf("response missing flow trace: %s", string(body))
	}
	steps, ok := flow["steps"].([]interface{})
	if !ok {
		t.Fatalf("flow trace missing steps: %#v", flow)
	}
	for _, rawStep := range steps {
		step, ok := rawStep.(map[string]interface{})
		if !ok {
			continue
		}
		responses, ok := step["responses"].([]interface{})
		if !ok {
			continue
		}
		for _, rawResponse := range responses {
			response, ok := rawResponse.(map[string]interface{})
			if ok && response["agent_id"] == agentID {
				return response
			}
		}
	}
	t.Fatalf("flow trace missing response for agent %q: %#v", agentID, flow)
	return nil
}

func workflowChoiceFinishReason(t *testing.T, body []byte) interface{} {
	t.Helper()
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("parse workflow response: %v", err)
	}
	return parsed["choices"].([]interface{})[0].(map[string]interface{})["finish_reason"]
}

func payloadHasToolMessage(messages []map[string]interface{}) bool {
	for _, message := range messages {
		if message["role"] == "tool" {
			return true
		}
	}
	return false
}

func payloadToolMessageIDContains(messages []map[string]interface{}, needle string) bool {
	for _, message := range messages {
		if message["role"] != "tool" {
			continue
		}
		if strings.Contains(fmt.Sprint(message["tool_call_id"]), needle) {
			return true
		}
	}
	return false
}

func payloadHasAssistantToolCalls(messages []map[string]interface{}) bool {
	for _, message := range messages {
		if message["role"] == "assistant" {
			if toolCalls, ok := message["tool_calls"].([]interface{}); ok && len(toolCalls) > 0 {
				return true
			}
		}
	}
	return false
}

func payloadHasTopLevelField(payload map[string]interface{}, field string) bool {
	value, ok := payload[field]
	if !ok || value == nil {
		return false
	}
	switch typed := value.(type) {
	case []interface{}:
		return len(typed) > 0
	case map[string]interface{}:
		return len(typed) > 0
	case string:
		return typed != ""
	default:
		return true
	}
}

func payloadMessagesContain(messages []map[string]interface{}, needle string) bool {
	for _, message := range messages {
		if strings.Contains(fmt.Sprint(message["content"]), needle) {
			return true
		}
	}
	return false
}

func newWorkflowTestServer(t *testing.T, responses map[string]string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		content := responses[payload.Model]
		if payload.Model == "qwen-coordinator" && strings.Contains(r.Header.Get("x-vsr-looper-iteration"), "4") {
			content = "final synthesized answer"
		}
		if payload.Model == "qwen-coordinator" && !json.Valid([]byte(content)) {
			content = "final synthesized answer"
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 0,
			"model":   payload.Model,
			"choices": []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": content,
				},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{
				"prompt_tokens":     1,
				"completion_tokens": 1,
				"total_tokens":      2,
			},
		})
	}))
}

func workflowContainsString(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}
