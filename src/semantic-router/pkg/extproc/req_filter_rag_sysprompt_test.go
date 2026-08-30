package extproc

import (
	"encoding/json"
	"strings"
	"testing"
)

// RAG system_prompt injection must preserve the existing system message's
// instructions even when its content is structured (an array of content parts,
// which is valid OpenAI input). Otherwise the user's original system prompt is
// silently dropped — same class as #2127, but on the RAG injection path.
func TestInjectAsSystemPromptPreservesStructuredSystemContent(t *testing.T) {
	r := &OpenAIRouter{}
	messages := []interface{}{
		map[string]interface{}{
			"role": "system",
			"content": []interface{}{
				map[string]interface{}{"type": "text", "text": "ORIGINAL_SYSTEM_INSTRUCTIONS"},
			},
		},
		map[string]interface{}{"role": "user", "content": "hi"},
	}
	requestMap := map[string]interface{}{"model": "m", "messages": messages}
	ctx := &RequestContext{}

	if err := r.injectAsSystemPrompt(messages, "RAG_CONTEXT_XYZ", requestMap, ctx); err != nil {
		t.Fatalf("injectAsSystemPrompt error: %v", err)
	}

	var parsed struct {
		Messages []map[string]interface{} `json:"messages"`
	}
	if err := json.Unmarshal(ctx.OriginalRequestBody, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	sys := parsed.Messages[0]["content"]
	text := systemContentToString(sys)

	if !strings.Contains(text, "RAG_CONTEXT_XYZ") {
		t.Fatalf("RAG context missing from system message: %v", sys)
	}
	if !strings.Contains(text, "ORIGINAL_SYSTEM_INSTRUCTIONS") {
		t.Fatalf("original structured system content was dropped: %v", sys)
	}
}

// systemContentToString flattens a message content that may be a plain string
// or a structured array of {type,text} parts, for assertion purposes.
func systemContentToString(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var b strings.Builder
		for _, part := range v {
			if pm, ok := part.(map[string]interface{}); ok {
				if t, ok := pm["text"].(string); ok {
					b.WriteString(t)
				}
			}
		}
		return b.String()
	default:
		return ""
	}
}
