package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// "disabled" is the default of the Responses truncation field: nothing is
// dropped from the context. A Chat or Messages backend does that anyway, so only
// "auto" needs conversation state from the target.
func TestResponsesTruncationDisabledTranslatesToStatelessTargets(t *testing.T) {
	cases := []struct {
		name       string
		truncation string
		wantErr    bool
	}{
		{"omitted", "", false},
		{"disabled", `,"truncation":"disabled"`, false},
		{"auto", `,"truncation":"auto"`, true},
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		for _, tc := range cases {
			t.Run(string(target)+"/"+tc.name, func(t *testing.T) {
				body := []byte(`{"model":"m","input":"hi"` + tc.truncation + `}`)
				_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIResponsesV1, target, body, nil)
				if !tc.wantErr {
					if err != nil {
						t.Fatalf("translate: %v", err)
					}
					return
				}
				if err == nil || !strings.Contains(err.Error(), "conversation_state") {
					t.Fatalf("truncation auto must still need conversation state, got %v", err)
				}
			})
		}
	}
}

// An unknown truncation value must still fail instead of taking the stateless path.
func TestResponsesTruncationUnknownValueIsRejected(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			body := []byte(`{"model":"m","input":"hi","truncation":"disable"}`)
			_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIResponsesV1, target, body, nil)
			if err == nil || !strings.Contains(err.Error(), "invalid_truncation") {
				t.Fatalf("an unknown truncation value must be rejected, got %v", err)
			}
		})
	}
}
