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
