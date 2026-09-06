package http

import (
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
