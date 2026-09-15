package looper

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDecodeModelStreamRequiresSuccessfulTerminal(t *testing.T) {
	prefix := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n"
	finish := "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	tests := []struct {
		name  string
		body  string
		valid bool
	}{
		{"complete", prefix + finish + "data: [DONE]\n\n", true},
		{"missing_sentinel", prefix + finish, false},
		{"missing_finish", prefix + "data: [DONE]\n\n", false},
		{"empty", "", false},
		{"error", prefix + "event: error\ndata: {\"error\":{\"type\":\"server_error\",\"message\":\"failed\"}}\n\n", false},
		{"malformed", prefix + "data: {\n\n", false},
		{"multiline_data", "data: {\"choices\":[{\"index\":0,\n" + "data: \"delta\":{\"content\":\"中文🚀\"}}]}\n\n" + finish + "data: [DONE]\n\n", true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			events, err := decodeModelStream([]byte(test.body), "model")
			if !test.valid {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, llmprotocol.EventResponseCompleted, events[len(events)-1].Type)
			if strings.Contains(test.name, "multiline") {
				var text string
				for _, event := range events {
					if event.Type == llmprotocol.EventOutputTextDelta {
						text += event.Delta
					}
				}
				require.Equal(t, "中文🚀", text)
			}
		})
	}
}
