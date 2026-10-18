package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Preserve-mode message fields must survive the replay clone: a fallback
// attempt re-encodes from the clone and would otherwise silently drop every
// captured field (the JSON round-trip skips json:"-" members).
func TestCloneSemanticRequestPreservesMessagePassthrough(t *testing.T) {
	request := &llmprotocol.Request{
		Generation: 1,
		Model:      "public/target",
		Instructions: []llmprotocol.InstructionBlock{{
			Role: llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "sys"},
			},
			Unknown: map[string]json.RawMessage{"x_sys": []byte(`"s1"`)},
		}},
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "hi"},
			},
			Unknown: map[string]json.RawMessage{"x_trace": []byte(`"u1"`)},
		}},
	}
	cloned, err := cloneSemanticRequestForReplay(request)
	if err != nil {
		t.Fatal(err)
	}
	if got := cloned.Messages[0].Unknown["x_trace"]; string(got) != `"u1"` {
		t.Fatalf("message passthrough lost in replay clone: %v", cloned.Messages[0].Unknown)
	}
	if got := cloned.Instructions[0].Unknown["x_sys"]; string(got) != `"s1"` {
		t.Fatalf("instruction passthrough lost in replay clone: %v", cloned.Instructions[0].Unknown)
	}
	// The clone must not alias the original maps: mutating one side cannot
	// leak into a retained snapshot.
	cloned.Messages[0].Unknown["x_trace"] = []byte(`"mutated"`)
	if got := request.Messages[0].Unknown["x_trace"]; string(got) != `"u1"` {
		t.Fatal("clone aliases the original passthrough map")
	}
}
