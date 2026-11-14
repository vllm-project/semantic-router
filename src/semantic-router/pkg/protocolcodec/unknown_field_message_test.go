package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A client that sends a field the contract does not define learns which one,
// at the top level or inside a nested object.
func TestUnknownRequestFieldIsNamed(t *testing.T) {
	for body, field := range map[string]string{
		`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"new_field":true}`:                     `unknown field "new_field"`,
		`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"adaptive","x":1}}`: `unknown field "x"`,
	} {
		_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(body))
		if err == nil || !strings.Contains(err.Error(), field) {
			t.Errorf("err = %v, want it to name %s", err, field)
		}
	}
}
