package protocolcodec

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnswerSeparatorRule(t *testing.T) {
	key := streamContentKey{}
	for _, test := range []struct {
		name      string
		reasoning bool
		text      []string
		want      []string
		removed   int64
	}{
		{"newlines before the answer", true, []string{"\n\nRed"}, []string{"Red"}, 2},
		{"whitespace of every kind", true, []string{" \t\r\n\u00a0\u2003Red"}, []string{"Red"}, 6},
		{"runes, not bytes", true, []string{"\n\n日本語"}, []string{"日本語"}, 2},
		{"its own deltas", true, []string{"\n", "\n", "Red", " square"}, []string{"", "", "Red", " square"}, 2},
		{"whitespace inside the answer stays", true, []string{"\n\nRed\n\nsquare"}, []string{"Red\n\nsquare"}, 2},
		{"no reasoning keeps the text", false, []string{"\n\nRed"}, []string{"\n\nRed"}, 0},
		{"answer first keeps the later text", false, []string{"Hello", " there"}, []string{"Hello", " there"}, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			var separator answerSeparator
			if test.reasoning {
				separator.observeReasoning()
			}
			for index, text := range test.text {
				require.Equal(t, test.want[index], separator.trim(key, text))
			}
			require.Equal(t, test.removed, separator.removed[key])
		})
	}
	t.Run("an answer that began before the reasoning is left alone", func(t *testing.T) {
		var separator answerSeparator
		require.Equal(t, "Hello", separator.trim(key, "Hello"))
		separator.observeReasoning()
		require.Equal(t, " there", separator.trim(key, " there"))
	})
}

// A same-format Chat reply is patched where it stands, so its replay differs
// from the provider bytes. Any other reply is encoded again, since its provider
// bytes still hold the separator.
func TestAnswerSeparatorEnvelope(t *testing.T) {
	engine := NewBuiltinEngine()
	chat := func(reasoning, content string) ([]byte, llmprotocol.Envelope) {
		provider := separatorChatReply(t, reasoning, content, nil)
		result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, provider, nil)
		require.NoError(t, err)
		return provider, result.Envelope
	}
	provider, patched := chat(separatorThought, "\n\nRed")
	require.NotEqual(t, string(provider), string(patched.Response), "the replay holds the patched body")
	require.False(t, patched.ResponseReencodeRequired)
	provider, untouched := chat(separatorThought, "Red")
	require.Equal(t, string(provider), string(untouched.Response), "a reply without the separator is replayed as sent")
	require.False(t, untouched.ResponseReencodeRequired)

	messages := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` +
		`{"type":"thinking","thinking":"hmm","signature":"c2ln"},{"type":"text","text":"\n\nRed"}],` +
		`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":2}}`)
	result, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, messages, nil)
	require.NoError(t, err)
	require.True(t, result.Envelope.ResponseReencodeRequired)
	require.False(t, result.Envelope.CanReplay(llmprotocol.AnthropicMessagesV1, result.Response.Generation, llmprotocol.DefaultPolicy(), true))
}
