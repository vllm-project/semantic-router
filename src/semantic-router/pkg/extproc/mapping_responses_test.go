package extproc

import (
	"encoding/json"
	"testing"
)

func TestMapResponsesRequestToChatCompletions_TextInput(t *testing.T) {
	in := []byte(`{"model":"gpt-test","input":"Hello world","temperature":0.2,"top_p":0.9,"max_output_tokens":128}`)
	out, err := mapResponsesRequestToChatCompletions(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal mapped: %v", err)
	}
	if m["model"].(string) != "gpt-test" {
		t.Fatalf("model not mapped")
	}
	if _, ok := m["messages"].([]interface{}); !ok {
		t.Fatalf("messages missing")
	}
}

func TestMapChatCompletionToResponses_Minimal(t *testing.T) {
	in := []byte(`{
        "id":"chatcmpl-1","object":"chat.completion","created":123,"model":"gpt-test",
        "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],
        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
    }`)
	out, err := mapChatCompletionToResponses(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal mapped: %v", err)
	}
	if m["object"].(string) != "response" {
		t.Fatalf("object not 'response'")
	}
	if m["stop_reason"].(string) == "" {
		t.Fatalf("stop_reason missing")
	}
}

func TestTranslateSSEChunkToResponses(t *testing.T) {
	chunk := []byte(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`)
	evs, ok := translateSSEChunkToResponses(chunk)
	if !ok || len(evs) == 0 {
		t.Fatalf("expected events")
	}
}

func TestMapResponsesRequestToChatCompletions_ToolsPassThrough(t *testing.T) {
	in := []byte(`{
        "model":"gpt-test",
        "input":"call a tool",
        "tools":[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{}}}}],
        "tool_choice":"auto"
    }`)
	out, err := mapResponsesRequestToChatCompletions(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal mapped: %v", err)
	}
	if _, ok := m["tools"]; !ok {
		t.Fatalf("tools not passed through")
	}
	if v, ok := m["tool_choice"]; !ok || v == nil {
		t.Fatalf("tool_choice not passed through")
	}
}

func TestMapChatCompletionToResponses_ToolCallsModern(t *testing.T) {
	in := []byte(`{
        "id":"x","object":"chat.completion","created":2,"model":"m",
        "choices":[{"index":0,"finish_reason":"stop","message":{
            "role":"assistant",
            "content":"",
            "tool_calls":[{"type":"function","function":{"name":"get_time","arguments":"{\\"tz\\":\\"UTC\\"}"}}]
        }}],
        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
    }`)
	out, err := mapChatCompletionToResponses(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	outs, _ := m["output"].([]interface{})
	if len(outs) == 0 {
		t.Fatalf("expected output entries")
	}
	var hasTool bool
	for _, o := range outs {
		om := o.(map[string]interface{})
		if om["type"] == "tool_call" {
			hasTool = true
		}
	}
	if !hasTool {
		t.Fatalf("expected tool_call in output")
	}
}

func TestMapChatCompletionToResponses_FunctionCallLegacy(t *testing.T) {
	in := []byte(`{
        "id":"x","object":"chat.completion","created":2,"model":"m",
        "choices":[{"index":0,"finish_reason":"stop","message":{
            "role":"assistant",
            "content":"",
            "function_call":{"name":"get_time","arguments":"{\\"tz\\":\\"UTC\\"}"}
        }}],
        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
    }`)
	out, err := mapChatCompletionToResponses(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	outs, _ := m["output"].([]interface{})
	var hasTool bool
	for _, o := range outs {
		if om, ok := o.(map[string]interface{}); ok && om["type"] == "tool_call" {
			hasTool = true
		}
	}
	if !hasTool {
		t.Fatalf("expected legacy function tool_call in output")
	}
}

func TestTranslateSSEChunkToResponses_ToolCallsDelta(t *testing.T) {
	chunk := []byte(`{
        "id":"c1","object":"chat.completion.chunk","created":1,
        "model":"m",
        "choices":[{"index":0,
          "delta":{
            "tool_calls":[{"index":0,"function":{"name":"get_time","arguments":"{\\"tz\\":\\"UTC\\"}"}}]
          },
          "finish_reason":null
        }]
    }`)
	evs, ok := translateSSEChunkToResponses(chunk)
	if !ok || len(evs) == 0 {
		t.Fatalf("expected events for tool_calls delta")
	}
	var hasToolDelta bool
	for _, ev := range evs {
		var m map[string]interface{}
		_ = json.Unmarshal(ev, &m)
		if m["type"] == "response.tool_calls.delta" {
			hasToolDelta = true
		}
	}
	if !hasToolDelta {
		t.Fatalf("expected response.tool_calls.delta event")
	}
}
