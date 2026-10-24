package http

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildStreamingCacheBodyReplaysReasoningToolsAndUsage(t *testing.T) {
	cached := []byte(`{
		"id":"cached",
		"model":"model-a",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"",
				"reasoning_content":"thinking",
				"tool_calls":[{
					"id":"call-1",
					"type":"function",
					"function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}
				}]
			},
			"finish_reason":"tool_calls"
		}],
		"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}
	}`)

	body := string(buildStreamingCacheBody(cached))

	for _, expected := range []string{
		`"reasoning_content":"thinking"`,
		`"tool_calls"`,
		`"finish_reason":"tool_calls"`,
		`"prompt_tokens":12`,
		"data: [DONE]",
	} {
		if !strings.Contains(body, expected) {
			t.Fatalf("streaming replay missing %q:\n%s", expected, body)
		}
	}
}

func TestBuildStreamingCacheBodyKeepsContentWhitespace(t *testing.T) {
	content := "Steps:\n1. a\n2.  b\n\n```go\nx := 1\n```"
	cached, err := json.Marshal(map[string]any{
		"model": "model-a",
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var replayed strings.Builder
	for _, line := range strings.Split(string(buildStreamingCacheBody(cached)), "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("invalid chunk %q: %v", payload, err)
		}
		for _, choice := range chunk.Choices {
			replayed.WriteString(choice.Delta.Content)
		}
	}

	if replayed.String() != content {
		t.Fatalf("streaming replay changed the content:\n got: %q\nwant: %q", replayed.String(), content)
	}
}

func TestBuildStreamingCacheBodyRejectsMultipleChoices(t *testing.T) {
	cached := []byte(`{
		"model":"model-a",
		"choices":[
			{"index":0,"message":{"role":"assistant","content":"a"},"finish_reason":"stop"},
			{"index":1,"message":{"role":"assistant","content":"b"},"finish_reason":"stop"}
		]
	}`)

	body := string(buildStreamingCacheBody(cached))

	if !strings.Contains(body, "multiple choices") ||
		!strings.Contains(body, "data: [DONE]") {
		t.Fatalf("multi-choice replay must fail explicitly:\n%s", body)
	}
}
