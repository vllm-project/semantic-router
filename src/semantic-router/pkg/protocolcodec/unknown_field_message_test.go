package protocolcodec

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A client that sends a field the contract does not define learns which one,
// at the top level or inside a nested object, by its path from the top of the
// request.
func TestUnknownRequestFieldIsNamed(t *testing.T) {
	for body, path := range map[string]string{
		`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"new_field":true}`:                     "new_field",
		`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"adaptive","x":1}}`: "thinking.x",
		`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi","x":1}]}`:                                "messages[].x",
	} {
		_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(err, &protocolError) {
			t.Errorf("err = %v, want unknown_parameter naming %s", err, path)
			continue
		}
		if protocolError.Code != "unknown_parameter" || protocolError.Parameter != path ||
			protocolError.Message != "Unrecognized request argument supplied: "+path {
			t.Errorf("err = %v (param %q), want unknown_parameter naming %s", err, protocolError.Parameter, path)
		}
	}
}
