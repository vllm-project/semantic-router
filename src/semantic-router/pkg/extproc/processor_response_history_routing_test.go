package extproc

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestRetainedResponseHistoryPrecedesSignalsAndToolsPolicy(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	} {
		for _, strip := range []bool{false, true} {
			name := string(format) + "/keep_history"
			if strip {
				name = string(format) + "/strip_history"
			}
			t.Run(name, func(t *testing.T) {
				router, model, request, ctx := retainedToolRoutingRequest(t, format)
				currentOnly := extractSemanticRequestSignals(request)
				ingress, err := json.Marshal(ctx.ResponseObjectState.Input)
				if err != nil {
					t.Fatal(err)
				}
				snapshot, err := router.extractRequestSignalSnapshot(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if snapshot.UserMessageCount != 2 || !snapshot.HasAssistantReply ||
					snapshot.AssistantToolCallCount != 1 || snapshot.ToolResultCount != 1 {
					t.Fatalf("routing missed retained conversation facts: %+v", snapshot)
				}
				if snapshot.ContextTokenFloor <= currentOnly.ContextTokenFloor {
					t.Fatal("retained history did not contribute to the routing context budget")
				}
				generation := request.Generation
				again, err := router.extractRequestSignalSnapshot(ctx)
				if err != nil || !reflect.DeepEqual(snapshot, again) || request.Generation != generation {
					t.Fatalf("repeated extraction changed the complete request: err=%v", err)
				}

				payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
					Enabled: true, Mode: config.ToolsPluginModeNone, StripToolHistory: strip,
				})
				if err != nil {
					t.Fatal(err)
				}
				ctx.VSRSelectedDecision = &config.Decision{Name: "private", Plugins: []config.DecisionPlugin{{
					Type: config.DecisionPluginTools, Configuration: payload,
				}}}
				response, err := router.handleModelRouting(request, model, "private", entropy.ReasoningDecision{}, model, ctx)
				if err != nil || response.GetRequestBody() == nil {
					t.Fatalf("dispatch failed: response=%+v err=%v", response, err)
				}
				body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
				decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, body)
				if err != nil {
					t.Fatalf("final provider request has invalid tool linkage: %v", err)
				}
				facts := extractSemanticRequestSignals(&decoded)
				wantTools := 1
				if strip {
					wantTools = 0
				}
				if facts.AssistantToolCallCount != wantTools || facts.ToolResultCount != wantTools || len(decoded.Tools) != 0 {
					t.Fatalf("tool policy did not survive final encoding: %+v", facts)
				}
				if facts.UserMessageCount != 2 || !facts.HasAssistantReply {
					t.Fatal("tool policy discarded ordinary conversation text")
				}
				after, err := json.Marshal(ctx.ResponseObjectState.Input)
				if err != nil || string(after) != string(ingress) {
					t.Fatal("routing mutated the immutable object-store ingress snapshot")
				}
			})
		}
	}
}

func TestMaterializedHistoryIsNotRepeatedBySessionOrMemoryReaders(t *testing.T) {
	for _, authenticated := range []bool{false, true} {
		router, _, request, ctx := retainedToolRoutingRequest(t, llmprotocol.OpenAIChatV1)
		if authenticated {
			ctx.Headers["x-authz-user-id"] = "synthetic-user"
		}
		if _, err := router.extractRequestSignalSnapshot(ctx); err != nil {
			t.Fatal(err)
		}
		_, _, history := router.extractSessionContext(ctx)
		if strings.Count(strings.Join(history, "\n"), "prior request") != 1 ||
			strings.Count(strings.Join(history, "\n"), "prior answer") != 1 {
			t.Fatalf("session context repeated retained history: %v", history)
		}
		_, _, messages, err := extractMemoryInfo(ctx)
		if (err == nil) != authenticated || len(messages) != len(request.Messages) {
			t.Fatalf("memory history duplicated or lost the conversation: got=%d want=%d err=%v", len(messages), len(request.Messages), err)
		}
	}
}

func retainedToolRoutingRequest(t *testing.T, format llmprotocol.WireFormat) (*OpenAIRouter, string, *llmprotocol.Request, *RequestContext) {
	t.Helper()
	router, model := routingTestRouterForFormat(format)
	store := NewMockResponseStore()
	if err := store.StoreResponse(t.Context(), &responseapi.StoredResponse{
		ID: "resp_previous",
		Input: []responseapi.InputItem{{
			Type: responseapi.ItemTypeMessage, Role: responseapi.RoleUser,
			Content: json.RawMessage(`"prior request"`),
		}},
		Output: []responseapi.OutputItem{
			{
				Type: responseapi.ItemTypeMessage, Role: responseapi.RoleAssistant,
				Content: []responseapi.ContentPart{{Type: responseapi.ContentTypeOutputText, Text: "prior answer"}},
			},
			{Type: responseapi.ItemTypeFunctionCall, CallID: "call_lookup", Name: "lookup", Arguments: `{}`},
		},
	}); err != nil {
		t.Fatal(err)
	}
	router.ResponseAPIFilter = NewResponseAPIFilter(store)
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: t.Context(), Headers: map[string]string{}}
	request, immediate := router.prepareProtocolRequest([]byte(`{
		"model":"public-model", "previous_response_id":"resp_previous", "store":false,
		"input":[{"type":"function_call_output","call_id":"call_lookup","output":"result"},
		{"role":"user","content":"current request"}],
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]
	}`), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("prepare retained request failed: %+v", immediate)
	}
	return router, model, request, ctx
}
