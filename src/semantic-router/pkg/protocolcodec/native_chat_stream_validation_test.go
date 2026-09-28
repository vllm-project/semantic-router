package protocolcodec

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNativeChatStreamValidatesLeadingFrames(t *testing.T) {
	const answer = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"fixture\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"
	const usage = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"fixture\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"
	const providerError = "data: {\"error\":{\"message\":\"Temporarily unavailable\",\"type\":\"server_error\",\"code\":\"service_unavailable\"}}\n\n"
	const done = "data: [DONE]\n\n"
	for _, tc := range []struct {
		name, body string
		valid      bool
	}{
		{"answer_then_usage", answer + usage + done, true},
		{"usage_before_choice", usage + answer + done, false},
		{"error_before_choice", providerError + answer + done, false},
		{"error_after_choice", answer + providerError + done, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewBuiltinEngine().ValidateNativeChatStream([]byte(tc.body))
			if tc.valid {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
			}
		})
	}
}
