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
	body := contextRecoveryChatRetrieveBody("secret-key")
	redacted := (&OpenAIRouter{}).redactContextRecoveryToolCalls(body, &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1})
	assertContextRecoveryRedacted(t, redacted, "secret-key")
}

func TestHandleContextRecoveryFollowupDecodesBackendChatFormat(t *testing.T) {
	for _, client := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIResponsesV1,
	} {
		t.Run(string(client), func(t *testing.T) {
			router, ctx, received := newContextRecoveryFollowup(t, client, llmprotocol.OpenAIChatV1, "")
			got, err := router.handleContextRecoveryFollowup(
				context.Background(),
				contextRecoveryChatRetrieveBody("key-1"),
				ctx,
			)
			if err != nil {
				t.Fatalf("handleContextRecoveryFollowup() error = %v", err)
			}
			assertContextRecoveryFollowup(t, got, *received)
		})
	}
}

func TestHandleContextRecoveryFollowupAcceptsResponseVendor(t *testing.T) {
	router, ctx, received := newContextRecoveryFollowup(
		t,
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.ResponseVendorAzure,
	)
	got, err := router.handleContextRecoveryFollowup(
		context.Background(),
		azureContextRecoveryChatRetrieveBody("key-1"),
		ctx,
	)
	if err != nil {
		t.Fatalf("handleContextRecoveryFollowup() error = %v", err)
	}
	assertContextRecoveryFollowup(t, got, *received)
}

func TestRedactContextRecoveryToolCallsUsesBackendFormatAndVendor(t *testing.T) {
	tests := []struct {
		name   string
		source llmprotocol.WireFormat
		target llmprotocol.WireFormat
		vendor llmprotocol.ResponseVendor
		body   []byte
	}{
		{
			name:   "anthropic_client_chat_backend",
			source: llmprotocol.AnthropicMessagesV1,
			target: llmprotocol.OpenAIChatV1,
			body:   contextRecoveryChatRetrieveBody("secret-key"),
		},
		{
			name:   "responses_client_chat_backend",
			source: llmprotocol.OpenAIResponsesV1,
			target: llmprotocol.OpenAIChatV1,
			body:   contextRecoveryChatRetrieveBody("secret-key"),
		},
		{
			name:   "azure_vendor",
			source: llmprotocol.OpenAIChatV1,
			target: llmprotocol.OpenAIChatV1,
			vendor: llmprotocol.ResponseVendorAzure,
			body:   azureContextRecoveryChatRetrieveBody("secret-key"),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			redacted := (&OpenAIRouter{}).redactContextRecoveryToolCalls(test.body, &RequestContext{
				SourceFormat:   test.source,
				TargetFormat:   test.target,
				ResponseVendor: test.vendor,
			})
			assertContextRecoveryRedacted(t, redacted, "secret-key")
		})
	}
}

func newContextRecoveryFollowup(
	t *testing.T,
	source llmprotocol.WireFormat,
	target llmprotocol.WireFormat,
	vendor llmprotocol.ResponseVendor,
) (*OpenAIRouter, *RequestContext, *map[string]interface{}) {
	t.Helper()
	var received map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		defer request.Body.Close()
		_ = json.NewDecoder(request.Body).Decode(&received)
		writer.Header().Set("Content-Type", "application/json")
		if vendor == llmprotocol.ResponseVendorAzure {
			_, _ = writer.Write(azureContextRecoveryChatAnswerBody())
			return
		}
		_, _ = writer.Write([]byte(`{
			"id":"followup",
			"model":"model",
			"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25},
			"choices":[{"index":0,"message":{"role":"assistant","content":"final answer"},"finish_reason":"stop"}]
		}`))
	}))
	t.Cleanup(server.Close)

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
		SourceFormat:                   source,
		TargetFormat:                   target,
		ResponseVendor:                 vendor,
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
	return router, ctx, &received
}

func contextRecoveryChatRetrieveBody(key string) []byte {
	return []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{
			"id":"retrieve-1",
			"type":"function",
			"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"` + key + `\"}"}
		}]}}]
	}`)
}

func azureContextRecoveryChatRetrieveBody(key string) []byte {
	return []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"prompt_filter_results":[{"prompt_index":0,"content_filter_results":{}}],
		"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"latency_checkpoint":{"time_to_first_token_ms":12}},
		"choices":[{"index":0,"finish_reason":"tool_calls",
			"content_filter_results":{"hate":{"filtered":false,"severity":"safe"}},
			"message":{"role":"assistant","content":null,"tool_calls":[{
				"id":"retrieve-1",
				"type":"function",
				"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"` + key + `\"}"}
			}]}}]
	}`)
}

func azureContextRecoveryChatAnswerBody() []byte {
	return []byte(`{
		"id":"followup","object":"chat.completion","created":1,"model":"model",
		"prompt_filter_results":[{"prompt_index":0,"content_filter_results":{}}],
		"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25,"latency_checkpoint":{"time_to_first_token_ms":12}},
		"choices":[{"index":0,"finish_reason":"stop",
			"content_filter_results":{"hate":{"filtered":false,"severity":"safe"}},
			"message":{"role":"assistant","content":"final answer"}}]
	}`)
}

func assertContextRecoveryFollowup(t *testing.T, got []byte, received map[string]interface{}) {
	t.Helper()
	if !json.Valid(got) || received == nil {
		t.Fatalf("followup response/request invalid: %s %#v", got, received)
	}
	if strings.Contains(string(got), "vsr_context_retrieve") {
		t.Fatalf("followup leaked recovery tool call: %s", got)
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

func assertContextRecoveryRedacted(t *testing.T, redacted []byte, key string) {
	t.Helper()
	if strings.Contains(string(redacted), key) ||
		strings.Contains(string(redacted), "vsr_context_retrieve") {
		t.Fatalf("redacted response leaked recovery data: %s", redacted)
	}
	if !strings.Contains(string(redacted), "could not be retrieved") {
		t.Fatalf("redacted response omitted fallback message: %s", redacted)
	}
}
