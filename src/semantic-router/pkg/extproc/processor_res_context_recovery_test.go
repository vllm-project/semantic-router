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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
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
	semanticRequest := testNeutralRequest("model", "question")
	semanticRequest.Tools = []llmprotocol.Tool{
		makeTool("lookup"),
		makeTool(contextcompression.RetrieveToolName),
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
		SemanticRequest:                semanticRequest,
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
	if followupToolsContainRetrieve(received["tools"]) {
		t.Fatalf("followup still advertised vsr_context_retrieve: %#v", received["tools"])
	}
}

func TestHandleContextRecoveryFollowupSkipsCompressionOnSessionCaller(t *testing.T) {
	recovered := strings.Repeat("irrelevant inventory ", 300) +
		"authentication validator failed " +
		strings.Repeat("irrelevant billing ", 300)
	decision := contextRecoveryFollowupCompressionDecision("recoverable")
	router := &OpenAIRouter{
		Cache: &spyCache{},
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{*decision},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model": {PreferredEndpoints: []string{"panel-backend"}},
				},
				VLLMEndpoints: []config.VLLMEndpoint{{
					Name:    "panel-backend",
					Address: "127.0.0.1",
					Port:    8000,
					Type:    "vllm",
					Weight:  1,
				}},
			},
		},
		CompressionRecovery: &recoveryStoreStub{},
	}
	router.CredentialResolver = newTestCredentialResolver(router.Config)
	upstream := &capturingHopUpstream{responseBody: []byte(`{
		"id":"followup","object":"chat.completion","created":1,"model":"model",
		"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25},
		"choices":[{"index":0,"message":{"role":"assistant","content":"final answer"},"finish_reason":"stop"}]
	}`)}
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	router.hopCaller = &graph.SessionCaller{
		Engine:   routing.NewEngine(router, opts),
		Upstream: upstream,
	}

	semanticRequest := testNeutralRequest("model", "question")
	semanticRequest.Tools = []llmprotocol.Tool{
		makeTool("lookup"),
		makeTool(contextcompression.RetrieveToolName),
	}
	ctx := &RequestContext{
		RequestID:                      "request-1",
		RequestModel:                   "model",
		VSRSelectedModel:               "model",
		VSRSelectedDecisionName:        decision.Name,
		VSRSelectedDecision:            &router.Config.Decisions[0],
		ContextCompressionRecoveryKeys: []string{"key-1"},
		Headers:                        map[string]string{"x-authz-user-id": "user-1"},
		SourceFormat:                   llmprotocol.OpenAIChatV1,
		SemanticRequest:                semanticRequest,
	}
	scope := router.contextCompressionScope(ctx)
	router.CompressionRecovery.(*recoveryStoreStub).entries = map[string]contextcompression.RecoveryEntry{
		scope + ":key-1": {
			Key:     "key-1",
			Scope:   scope,
			Content: recovered,
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
	got, err := router.handleContextRecoveryFollowup(context.Background(), response, ctx)
	if err != nil {
		t.Fatalf("handleContextRecoveryFollowup() error = %v", err)
	}
	if !json.Valid(got) || len(upstream.body) == 0 {
		t.Fatalf("followup response/request invalid: %s %s", got, upstream.body)
	}
	if !strings.Contains(string(upstream.body), strings.Repeat("irrelevant inventory ", 30)) {
		t.Fatalf("SessionCaller hop re-compressed recovered payload: %s", upstream.body)
	}
	if followupToolsContainRetrieve(decodeJSONObject(t, upstream.body)["tools"]) {
		t.Fatalf("SessionCaller hop re-injected vsr_context_retrieve: %s", upstream.body)
	}
	if !strings.Contains(string(got), "final answer") {
		t.Fatalf("followup response = %s", got)
	}
}

func TestIsContextRecoveryFollowupRequestRequiresMatchingToolResult(t *testing.T) {
	recovered := "full original context"
	if !isContextRecoveryFollowupRequest(contextRecoveryFollowupSemanticRequest("model", recovered)) {
		t.Fatal("assistant retrieve call with matching tool result was not detected")
	}
	ordinary := semanticCompressionRequest(recovered)
	if isContextRecoveryFollowupRequest(ordinary) {
		t.Fatal("ordinary Looper tool hop was treated as a recovery follow-up")
	}
	callOnly := testNeutralRequest("model", "question")
	callOnly.Messages = append(callOnly.Messages, llmprotocol.Message{
		Role: llmprotocol.RoleAssistant,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{
				ID: "retrieve-1", Name: contextcompression.RetrieveToolName, Arguments: `{"key":"key-1"}`,
			},
		}},
	})
	if isContextRecoveryFollowupRequest(callOnly) {
		t.Fatal("retrieve call without a tool result was treated as a follow-up")
	}
}

func TestIsContextRecoveryFollowupRequestOnlyMatchesCurrentExchange(t *testing.T) {
	for _, nextRole := range []llmprotocol.Role{llmprotocol.RoleUser, llmprotocol.RoleAssistant} {
		request := contextRecoveryFollowupSemanticRequest("model", "recovered context")
		request.Messages = append(request.Messages, llmprotocol.Message{
			Role: nextRole, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "next turn"}},
		})
		if isContextRecoveryFollowupRequest(request) {
			t.Fatalf("historical recovery matched after a %s message", nextRole)
		}
	}
	request := contextRecoveryFollowupSemanticRequest("model", "recovered context")
	ordinary := semanticCompressionRequest("ordinary tool result")
	request.Messages = append(request.Messages, ordinary.Messages[:3]...)
	if isContextRecoveryFollowupRequest(request) {
		t.Fatal("historical recovery matched a later ordinary tool exchange")
	}
	request = contextRecoveryFollowupSemanticRequest("model", "recovered context")
	request.Messages[len(request.Messages)-1].Content[0].ToolResult.CallID = "other-call"
	if isContextRecoveryFollowupRequest(request) {
		t.Fatal("unmatched trailing result was treated as recovery")
	}
}

func TestIsContextRecoveryFollowupRequestMatchesMultipleResults(t *testing.T) {
	request := contextRecoveryFollowupSemanticRequest("model", "first recovered context")
	assistant := &request.Messages[len(request.Messages)-2]
	assistant.Content = append(assistant.Content, llmprotocol.Content{
		Kind:     llmprotocol.ContentToolCall,
		ToolCall: &llmprotocol.ToolCall{ID: "retrieve-2", Name: contextcompression.RetrieveToolName, Arguments: `{"key":"key-2"}`},
	})
	request.Messages = append(request.Messages, llmprotocol.Message{
		Role: llmprotocol.RoleTool,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
			CallID: "retrieve-2", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "second recovered context"}},
		}}},
	})
	if !isContextRecoveryFollowupRequest(request) {
		t.Fatal("multiple matching recovery results were not detected")
	}
}

type capturingHopUpstream struct {
	body         []byte
	responseBody []byte
}

func (u *capturingHopUpstream) Send(_ context.Context, call *routing.Call) (*graph.Delivery, error) {
	u.body = append([]byte(nil), call.Request.Body...)
	return &graph.Delivery{
		Response: &routing.UpstreamResponse{
			Status: http.StatusOK,
			Header: routing.Header{{Name: "content-type", Value: "application/json"}},
			Body:   strings.NewReader(string(u.responseBody)),
		},
	}, nil
}

func contextRecoveryFollowupCompressionDecision(name string) *config.Decision {
	return &config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":      true,
				"failure_mode": "fail_closed",
				"targets": map[string]interface{}{
					"tool_outputs": map[string]interface{}{
						"mode": "recoverable", "min_tokens": 10, "target_tokens": 30,
					},
				},
				"recovery": map[string]interface{}{
					"enabled":               true,
					"store":                 "redis",
					"max_bytes_per_request": 1024 * 1024,
				},
			}),
		}},
		ModelRefs: []config.ModelRef{{Model: "model"}},
	}
}

func followupToolsContainRetrieve(raw interface{}) bool {
	tools, _ := raw.([]interface{})
	for _, tool := range tools {
		object, _ := tool.(map[string]interface{})
		function, _ := object["function"].(map[string]interface{})
		if function["name"] == contextcompression.RetrieveToolName {
			return true
		}
		if object["name"] == contextcompression.RetrieveToolName {
			return true
		}
	}
	return false
}

func decodeJSONObject(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var object map[string]interface{}
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatalf("decode hop body: %v", err)
	}
	return object
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
	redacted := (&OpenAIRouter{}).redactContextRecoveryToolCalls(body, &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1})
	if strings.Contains(string(redacted), "secret-key") ||
		strings.Contains(string(redacted), "vsr_context_retrieve") {
		t.Fatalf("redacted response leaked recovery data: %s", redacted)
	}
	if !strings.Contains(string(redacted), "could not be retrieved") {
		t.Fatalf("redacted response omitted fallback message: %s", redacted)
	}
}
