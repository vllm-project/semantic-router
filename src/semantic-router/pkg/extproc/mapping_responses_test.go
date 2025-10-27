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
