package extproc

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func guardProvenanceText(text string) []llmprotocol.Content {
	return []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}
}

func TestGuardProvenanceNeutralSnapshot(t *testing.T) {
	request := &llmprotocol.Request{
		Instructions: []llmprotocol.InstructionBlock{
			{Role: llmprotocol.RoleSystem, Content: guardProvenanceText("system-instruction")},
			{Role: llmprotocol.RoleDeveloper, Content: guardProvenanceText("developer-instruction")},
		},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: guardProvenanceText("prior-user")},
			{Role: llmprotocol.RoleAssistant, Content: guardProvenanceText("assistant-answer")},
			{Role: llmprotocol.RoleTool, Content: guardProvenanceText("prior-tool")},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "current-user"},
				{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call-1", Content: guardProvenanceText("current-tool-result")}},
			}},
		},
	}
	before, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := extractSemanticRequestSignals(request)
	want := &classification.JailbreakInput{
		Current: []classification.JailbreakContent{{Role: "tool", Text: "prior-tool"}, {Role: "user", Text: "current-user"}, {Role: "tool", Text: "current-tool-result"}},
		History: []classification.JailbreakContent{{Role: "user", Text: "prior-user"}},
	}
	if !reflect.DeepEqual(snapshot.JailbreakInput, want) {
		t.Fatalf("Guard projection = %#v, want %#v", snapshot.JailbreakInput, want)
	}
	if snapshot.UserContent != "current-user" || !reflect.DeepEqual(snapshot.NonUserMessages,
		[]string{"system-instruction", "developer-instruction", "assistant-answer"}) {
		t.Fatal("Guard projection changed other signals' role/text inputs")
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	input := router.prepareSignalEvaluationInput(signalConversationHistoryFromSnapshot(snapshot))
	if !reflect.DeepEqual(input.requestFacts.JailbreakInput, want) {
		t.Fatal("Guard provenance was lost between snapshot and signal dispatch")
	}
	after, err := json.Marshal(request)
	if err != nil || string(before) != string(after) {
		t.Fatal("Guard extraction rewrote the provider request")
	}
}

func TestGuardProvenanceNoTrustedFallbackOrStaleCurrentUser(t *testing.T) {
	for _, role := range []llmprotocol.Role{llmprotocol.RoleSystem, llmprotocol.RoleDeveloper, llmprotocol.RoleAssistant} {
		t.Run(string(role), func(t *testing.T) {
			request := &llmprotocol.Request{Messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleUser, Content: guardProvenanceText("older-user")},
				{Role: role, Content: guardProvenanceText("trusted-final-message")},
			}}
			snapshot := extractSemanticRequestSignals(request)
			if snapshot.JailbreakInput == nil || len(snapshot.JailbreakInput.Current) != 0 {
				t.Fatal("trusted final message or stale user turn became current Guard evidence")
			}
			want := []classification.JailbreakContent{{Role: "user", Text: "older-user"}}
			if !reflect.DeepEqual(snapshot.JailbreakInput.History, want) {
				t.Fatal("untrusted prior user was not retained for include_history")
			}
		})
	}
	request := &llmprotocol.Request{Instructions: []llmprotocol.InstructionBlock{{
		Role: llmprotocol.RoleSystem, Content: guardProvenanceText("instruction-only"),
	}}}
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	input := router.prepareSignalEvaluationInput(extractSignalConversationHistory(request))
	if input.evaluationText != "instruction-only" || input.requestFacts.JailbreakInput == nil ||
		len(input.requestFacts.JailbreakInput.Current)+len(input.requestFacts.JailbreakInput.History) != 0 {
		t.Fatal("general fallback must not become a Guard fallback")
	}
}

func TestGuardProvenanceCurrentToolAndNonText(t *testing.T) {
	for _, role := range []llmprotocol.Role{llmprotocol.RoleTool, llmprotocol.RoleUser} {
		request := &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: role, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call-2", Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "tool-visible-text"},
				{Kind: llmprotocol.ContentImage, Data: "media-is-not-text", MediaType: "image/png"},
			}},
		}}}}}
		want := []classification.JailbreakContent{{Role: "tool", Text: "tool-visible-text"}}
		if got := extractJailbreakInput(request); !reflect.DeepEqual(got.Current, want) {
			t.Fatalf("role %s current tool projection = %#v", role, got)
		}
	}
}

func TestGuardProvenanceDecodedToolResults(t *testing.T) {
	for _, test := range []struct {
		name  string
		codec protocolcodec.MessageCodec
		body  string
	}{
		{"chat", protocolcodec.OpenAIChatCodec{}, `{"model":"test-model","messages":[{"role":"system","content":"system-marker"},{"role":"user","content":"user-marker"},{"role":"assistant","content":"assistant-marker","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-1","content":"tool-marker"}]}`},
		{"responses", protocolcodec.OpenAIResponsesCodec{}, `{"model":"test-model","instructions":"system-marker","input":[{"role":"user","content":"user-marker"},{"type":"function_call","call_id":"call-1","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"call-1","output":"tool-marker"}]}`},
		{"anthropic", protocolcodec.AnthropicMessagesCodec{}, `{"model":"test-model","max_tokens":32,"system":"system-marker","messages":[{"role":"user","content":"user-marker"},{"role":"assistant","content":[{"type":"text","text":"assistant-marker"},{"type":"tool_use","id":"call-1","name":"lookup","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","content":"tool-marker"}]}]}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := test.codec.DecodeRequest([]byte(test.body), llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatal(err)
			}
			want := &classification.JailbreakInput{
				Current: []classification.JailbreakContent{{Role: "tool", Text: "tool-marker"}},
				History: []classification.JailbreakContent{{Role: "user", Text: "user-marker"}},
			}
			if got := extractSemanticRequestSignals(&request).JailbreakInput; !reflect.DeepEqual(got, want) {
				t.Fatalf("decoded Guard projection = %#v, want %#v", got, want)
			}
		})
	}
}

func TestGuardProvenanceDecodedSiblingToolResults(t *testing.T) {
	body := []byte(`{"model":"test-model","max_tokens":32,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"lookup","input":{}},{"type":"tool_use","id":"call-2","name":"lookup","input":{}}]},{"role":"user","content":[{"type":"text","text":"current-user"},{"type":"tool_result","tool_use_id":"call-1","content":"first-result"},{"type":"tool_result","tool_use_id":"call-2","content":"second-result"},{"type":"text","text":"current-suffix"}]}]}`)
	request, _, _, err := (protocolcodec.AnthropicMessagesCodec{}).DecodeRequest(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	want := &classification.JailbreakInput{Current: []classification.JailbreakContent{
		{Role: "user", Text: "current-user"},
		{Role: "tool", Text: "first-result"},
		{Role: "tool", Text: "second-result"},
		{Role: "user", Text: "current-suffix"},
	}}
	if got := extractSemanticRequestSignals(&request).JailbreakInput; !reflect.DeepEqual(got, want) {
		t.Fatalf("sibling results were split across current and history: %#v", got)
	}
}
