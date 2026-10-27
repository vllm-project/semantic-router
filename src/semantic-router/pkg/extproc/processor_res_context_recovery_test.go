package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type recoveryStoreStub struct {
	entries map[string]contextcompression.RecoveryEntry
}

func (store *recoveryStoreStub) Put(
	_ context.Context,
	entry contextcompression.RecoveryEntry,
	_ time.Duration,
) error {
	if store.entries == nil {
		store.entries = make(map[string]contextcompression.RecoveryEntry)
	}
	store.entries[entry.Scope+":"+entry.Key] = entry
	return nil
}

func (store *recoveryStoreStub) Get(
	_ context.Context,
	scope string,
	key string,
) (contextcompression.RecoveryEntry, error) {
	entry, ok := store.entries[scope+":"+key]
	if !ok {
		return contextcompression.RecoveryEntry{}, contextcompression.ErrRecoveryNotFound
	}
	return entry, nil
}

func (store *recoveryStoreStub) Health(context.Context) error { return nil }
func (store *recoveryStoreStub) Close() error                 { return nil }
func (store *recoveryStoreStub) InvalidateScope(
	_ context.Context,
	scope string,
) (int64, error) {
	var count int64
	for key, entry := range store.entries {
		if entry.Scope == scope {
			delete(store.entries, key)
			count++
		}
	}
	return count, nil
}

func TestHandleContextRecoveryFollowupCallsLooper(t *testing.T) {
	var received map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		defer request.Body.Close()
		_ = json.NewDecoder(request.Body).Decode(&received)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{
			"id":"followup",
			"model":"model",
			"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25},
			"choices":[{"index":0,"message":{"role":"assistant","content":"final answer"},"finish_reason":"stop"}]
		}`))
	}))
	defer server.Close()

	decision := &config.Decision{
		Name: "recoverable",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"recovery": map[string]interface{}{
					"enabled": true,
					"store":   "redis",
				},
			}),
		}},
	}
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: server.URL},
		},
		CompressionRecovery: &recoveryStoreStub{},
	}
	ctx := &RequestContext{
		RequestID:                      "request-1",
		RequestModel:                   "model",
		VSRSelectedModel:               "model",
		VSRSelectedDecisionName:        decision.Name,
		VSRSelectedDecision:            decision,
		ContextCompressionRecoveryKeys: []string{"key-1"},
		Headers:                        map[string]string{"x-authz-user-id": "user-1"},
		SourceFormat:                   llmprotocol.OpenAIChatV1,
		SemanticRequest:                testNeutralRequest("model", "question"),
	}
	scope := router.contextCompressionScope(ctx)
	router.CompressionRecovery.(*recoveryStoreStub).entries = map[string]contextcompression.RecoveryEntry{
		scope + ":key-1": {
			Key:     "key-1",
			Scope:   scope,
			Content: "full original context",
		},
	}
	response := []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{
			"id":"retrieve-1",
			"type":"function",
			"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"key-1\"}"}
		}]}}]
	}`)
	got, err := router.handleContextRecoveryFollowup(
		context.Background(),
		response,
		ctx,
	)
	if err != nil {
		t.Fatalf("handleContextRecoveryFollowup() error = %v", err)
	}
	if !json.Valid(got) || received == nil {
		t.Fatalf("followup response/request invalid: %s %#v", got, received)
	}
	var final map[string]interface{}
	_ = json.Unmarshal(got, &final)
	usage := final["usage"].(map[string]interface{})
	if usage["total_tokens"] != float64(38) {
		t.Fatalf("merged usage = %#v", usage)
	}
	messages := received["messages"].([]interface{})
	tool := messages[len(messages)-1].(map[string]interface{})
	if tool["content"] != "full original context" {
		t.Fatalf("recovered content = %#v", tool["content"])
	}
}

func TestParseContextRecoveryCallsPreservesAssistantContentAndClientTools(t *testing.T) {
	calls, assistant, err := parseContextRecoveryCalls(llmprotocol.Response{Output: []llmprotocol.OutputItem{{
		Role: llmprotocol.RoleAssistant,
		Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "looking that up"},
			{Kind: llmprotocol.ContentReasoning, Text: "need compressed context"},
			{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
				ID: "client-1", Name: "get_weather", Arguments: `{"city":"sf"}`,
			}},
			{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
				ID: "retrieve-1", Name: contextcompression.RetrieveToolName, Arguments: `{"key":"key-1"}`,
			}},
		},
	}}})
	if err != nil {
		t.Fatalf("parseContextRecoveryCalls() error = %v", err)
	}
	if len(calls) != 1 || calls[0].ID != "retrieve-1" || calls[0].Key != "key-1" {
		t.Fatalf("calls = %#v", calls)
	}
	if len(assistant) != 1 || len(assistant[0].Content) != 4 {
		t.Fatalf("assistant = %#v", assistant)
	}
	if assistant[0].Content[0].Kind != llmprotocol.ContentText ||
		assistant[0].Content[0].Text != "looking that up" {
		t.Fatalf("assistant text dropped: %#v", assistant[0].Content)
	}
	if assistant[0].Content[1].Kind != llmprotocol.ContentReasoning {
		t.Fatalf("reasoning dropped: %#v", assistant[0].Content)
	}
	if assistant[0].Content[2].ToolCall == nil || assistant[0].Content[2].ToolCall.Name != "get_weather" {
		t.Fatalf("client tool dropped: %#v", assistant[0].Content)
	}
	if assistant[0].Content[3].ToolCall == nil ||
		assistant[0].Content[3].ToolCall.Name != contextcompression.RetrieveToolName {
		t.Fatalf("recovery tool dropped: %#v", assistant[0].Content)
	}
}

func TestHandleContextRecoveryFollowupSkipsMixedClientToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(
		http.ResponseWriter,
		*http.Request,
	) {
		t.Error("mixed client and recovery tool calls dispatched a follow-up")
	}))
	defer server.Close()

	decision := &config.Decision{
		Name: "recoverable",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"recovery": map[string]interface{}{
					"enabled": true,
					"store":   "redis",
				},
			}),
		}},
	}
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: server.URL},
		},
		CompressionRecovery: &recoveryStoreStub{},
	}
	ctx := &RequestContext{
		RequestID:                      "request-1",
		RequestModel:                   "model",
		VSRSelectedModel:               "model",
		VSRSelectedDecisionName:        decision.Name,
		VSRSelectedDecision:            decision,
		ContextCompressionRecoveryKeys: []string{"key-1"},
		Headers:                        map[string]string{"x-authz-user-id": "user-1"},
		SourceFormat:                   llmprotocol.OpenAIChatV1,
		SemanticRequest:                testNeutralRequest("model", "question"),
	}
	scope := router.contextCompressionScope(ctx)
	router.CompressionRecovery.(*recoveryStoreStub).entries = map[string]contextcompression.RecoveryEntry{
		scope + ":key-1": {
			Key:     "key-1",
			Scope:   scope,
			Content: "full original context",
		},
	}
	response := []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"looking up","tool_calls":[
			{"id":"client","type":"function","function":{"name":"get_weather","arguments":"{}"}},
			{"id":"internal","type":"function","function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"key-1\"}"}}
		]}}]
	}`)
	got, err := router.handleContextRecoveryFollowup(
		context.Background(),
		response,
		ctx,
	)
	if err != nil {
		t.Fatalf("handleContextRecoveryFollowup() error = %v", err)
	}
	if string(got) != string(response) {
		t.Fatalf("mixed-call follow-up mutated the response: %s", got)
	}
	redacted := router.redactContextRecoveryToolCalls(got, ctx)
	if strings.Contains(string(redacted), "vsr_context_retrieve") {
		t.Fatalf("mixed-call redaction leaked recovery tool: %s", redacted)
	}
	if !strings.Contains(string(redacted), "get_weather") {
		t.Fatalf("mixed-call redaction dropped client tool: %s", redacted)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(redacted, &decoded); err != nil {
		t.Fatalf("redacted mixed-call response: %v", err)
	}
	choice := decoded["choices"].([]interface{})[0].(map[string]interface{})
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %#v, want tool_calls", choice["finish_reason"])
	}
}

func TestParseContextRecoveryCallsRejectsMalformedArguments(t *testing.T) {
	_, _, err := parseContextRecoveryCalls(llmprotocol.Response{Output: []llmprotocol.OutputItem{{
		Role: llmprotocol.RoleAssistant,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
			ID: "call", Name: contextcompression.RetrieveToolName, Arguments: "not-json",
		}}},
	}}})
	if err == nil {
		t.Fatal("parseContextRecoveryCalls() accepted malformed arguments")
	}
}

func TestHandleContextRecoveryFollowupRejectsForgedKey(t *testing.T) {
	decision := &config.Decision{
		Name: "recoverable",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"recovery": map[string]interface{}{
					"enabled": true,
					"store":   "redis",
				},
			}),
		}},
	}
	router := &OpenAIRouter{
		Config:              &config.RouterConfig{},
		CompressionRecovery: &recoveryStoreStub{},
	}
	ctx := &RequestContext{
		RequestID:                      "request",
		VSRSelectedDecision:            decision,
		ContextCompressionRecoveryKeys: []string{"issued-key"},
		Headers:                        map[string]string{"x-authz-user-id": "user"},
		SourceFormat:                   llmprotocol.OpenAIChatV1,
		SemanticRequest:                testNeutralRequest("model", "question"),
	}
	response := []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{
			"id":"retrieve",
			"type":"function",
			"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"forged-key\"}"}
		}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	if _, err := router.handleContextRecoveryFollowup(
		context.Background(),
		response,
		ctx,
	); err == nil {
		t.Fatal("forged recovery key was accepted")
	}
}

func TestHandleContextRecoveryFollowupRedactsResidualRetrieve(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		defer request.Body.Close()
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{
			"id":"followup",
			"model":"model",
			"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25},
			"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{
				"id":"retrieve-2",
				"type":"function",
				"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"secret-key\"}"}
			}]}}]
		}`))
	}))
	defer server.Close()

	decision := &config.Decision{
		Name: "recoverable",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"recovery": map[string]interface{}{
					"enabled": true,
					"store":   "redis",
				},
			}),
		}},
	}
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: server.URL},
		},
		CompressionRecovery: &recoveryStoreStub{},
	}
	ctx := &RequestContext{
		RequestID:                      "request-1",
		RequestModel:                   "model",
		VSRSelectedModel:               "model",
		VSRSelectedDecisionName:        decision.Name,
		VSRSelectedDecision:            decision,
		ContextCompressionRecoveryKeys: []string{"key-1"},
		Headers:                        map[string]string{"x-authz-user-id": "user-1"},
		SourceFormat:                   llmprotocol.OpenAIChatV1,
		SemanticRequest:                testNeutralRequest("model", "question"),
	}
	scope := router.contextCompressionScope(ctx)
	router.CompressionRecovery.(*recoveryStoreStub).entries = map[string]contextcompression.RecoveryEntry{
		scope + ":key-1": {
			Key:     "key-1",
			Scope:   scope,
			Content: "full original context",
		},
	}
	response := []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{
			"id":"retrieve-1",
			"type":"function",
			"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"key-1\"}"}
		}]}}]
	}`)
	got, err := router.handleContextRecoveryFollowup(
		context.Background(),
		response,
		ctx,
	)
	if err != nil {
		t.Fatalf("handleContextRecoveryFollowup() error = %v", err)
	}
	if !strings.Contains(string(got), "vsr_context_retrieve") {
		t.Fatalf("successful followup omitted residual retrieve: %s", got)
	}
	redacted := router.redactContextRecoveryToolCalls(got, ctx)
	if strings.Contains(string(redacted), "vsr_context_retrieve") ||
		strings.Contains(string(redacted), "secret-key") {
		t.Fatalf("residual retrieve leaked after redact: %s", redacted)
	}
}

func TestRedactContextRecoveryToolCallsDoesNotExposeKey(t *testing.T) {
	body := []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"choices":[{
			"index":0,
			"finish_reason":"tool_calls",
			"message":{"role":"assistant","content":null,"tool_calls":[{
				"id":"retrieve",
				"type":"function",
				"function":{
					"name":"vsr_context_retrieve",
					"arguments":"{\"key\":\"secret-key\"}"
				}
			}]}
		}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	for _, name := range []string{"vsr_context_retrieve", `vsr_context_\u0072etrieve`} {
		t.Run(name, func(t *testing.T) {
			encoded := []byte(strings.ReplaceAll(string(body), "vsr_context_retrieve", name))
			redacted := (&OpenAIRouter{}).redactContextRecoveryToolCalls(encoded, &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1})
			if strings.Contains(string(redacted), "secret-key") || strings.Contains(string(redacted), name) {
				t.Fatalf("redacted response leaked recovery data: %s", redacted)
			}
			if !strings.Contains(string(redacted), "could not be retrieved") {
				t.Fatalf("redacted response omitted fallback message: %s", redacted)
			}
		})
	}
}

func TestRedactContextRecoveryToolCallsPreservesOrdinaryResponse(t *testing.T) {
	for _, content := range []string{`ordinary answer`, `ordinary\nanswer`} {
		body := []byte(`{"id":"answer","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"` + content + `"},"finish_reason":"stop"}]}`)
		got := (&OpenAIRouter{}).redactContextRecoveryToolCalls(body, &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1})
		if string(got) != string(body) {
			t.Fatalf("ordinary response changed: %s", got)
		}
	}
}
