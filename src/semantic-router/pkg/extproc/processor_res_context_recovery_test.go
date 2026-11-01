package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

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

func TestHandleResponseBodySameFormatChatRecoverySetsBodyMutation(t *testing.T) {
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
		TargetFormat:                   llmprotocol.OpenAIChatV1,
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
	upstream := []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13},
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{
			"id":"retrieve-1",
			"type":"function",
			"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"key-1\"}"}
		}]}}]
	}`)
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: upstream, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("handleResponseBody() error = %v", err)
	}
	assertSameFormatChatBodyMutation(t, response, upstream, "final answer", "vsr_context_retrieve")
}

func TestHandleResponseBodySameFormatChatRedactionSetsBodyMutation(t *testing.T) {
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
		TargetFormat:                   llmprotocol.OpenAIChatV1,
		SemanticRequest:                testNeutralRequest("model", "question"),
	}
	upstream := []byte(`{
		"id":"initial","object":"chat.completion","created":1,"model":"model",
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{
			"id":"retrieve",
			"type":"function",
			"function":{"name":"vsr_context_retrieve","arguments":"{\"key\":\"secret-key\"}"}
		}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: upstream, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("handleResponseBody() error = %v", err)
	}
	assertSameFormatChatBodyMutation(t, response, upstream, "could not be retrieved", "secret-key", "vsr_context_retrieve")
}

func assertSameFormatChatBodyMutation(
	t *testing.T,
	response *ext_proc.ProcessingResponse,
	upstream []byte,
	wantContains string,
	wantAbsent ...string,
) {
	t.Helper()
	if response.GetImmediateResponse() != nil {
		t.Fatalf("response was replaced with an error: %+v", response.GetImmediateResponse())
	}
	common := response.GetResponseBody().GetResponse()
	got := common.GetBodyMutation().GetBody()
	if len(got) == 0 {
		t.Fatal("same-format Chat did not set BodyMutation")
	}
	if string(got) == string(upstream) {
		t.Fatalf("BodyMutation retained the upstream tool-call body: %s", got)
	}
	if !strings.Contains(string(got), wantContains) {
		t.Fatalf("BodyMutation omitted %q: %s", wantContains, got)
	}
	for _, absent := range wantAbsent {
		if strings.Contains(string(got), absent) {
			t.Fatalf("BodyMutation leaked %q: %s", absent, got)
		}
	}
	if !containsStringForTest(common.GetHeaderMutation().GetRemoveHeaders(), "content-length") {
		t.Fatalf("content-length was not removed: %#v", common.GetHeaderMutation())
	}
}
