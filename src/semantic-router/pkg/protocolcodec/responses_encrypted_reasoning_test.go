package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A store:false client that asks for provider-encrypted reasoning keeps the
// item the Router returns and sends it back on the next turn. Accepting the
// include and then rejecting the item leaves that client with no continuation.
func TestResponsesEncryptedReasoningReplayIsAccepted(t *testing.T) {
	engine := NewBuiltinEngine()

	request := []byte(`{"model":"m","input":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"gAAAA"}]}`)
	_, _, diagnostics, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, request)
	if err != nil {
		t.Fatalf("replaying a reasoning item that carries encrypted_content: %v", err)
	}
	assertDroppedFields(t, diagnostics, "input[].encrypted_content")
}

// The reply's own items are what a client sends back on the next turn, so the
// reasoning item arrives with the status the provider stamped on it. Rejecting
// that status would break the continuation the include path promised.
func TestResponsesReplayedReasoningItemKeepsItsProviderStatus(t *testing.T) {
	engine := NewBuiltinEngine()

	input := `[{"type":"reasoning","id":"rs_1","status":"completed","summary":[{"type":"summary_text","text":"thought"}],"encrypted_content":"gAAAA"},` +
		`{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"answer"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"more"}]}]`
	request, _, diagnostics, err := engine.DecodeRequest(
		llmprotocol.OpenAIResponsesV1,
		[]byte(`{"model":"m","input":`+input+`}`),
	)
	if err != nil {
		t.Fatalf("replaying a reply's own items: %v", err)
	}
	assertDroppedFields(t, diagnostics, "input[].encrypted_content")

	kept := false
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentReasoning && content.Text == "thought" {
				kept = true
			}
		}
	}
	if !kept {
		t.Fatalf("the replayed reasoning summary is missing: %+v", request.Messages)
	}
}
