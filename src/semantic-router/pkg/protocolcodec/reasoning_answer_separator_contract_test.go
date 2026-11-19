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

// A same-format reply is patched where it stands, keeping any other member
// the provider sent, including a citation whose offsets the trim must shift.
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
	require.False(t, result.Envelope.ResponseReencodeRequired, "the signed thinking block is untouched and replays fine")
	require.Contains(t, string(result.Envelope.Response), `"signature":"c2ln"`)
	require.True(t, result.Envelope.CanReplay(llmprotocol.AnthropicMessagesV1, result.Response.Generation, llmprotocol.DefaultPolicy(), true))

	responsesWithCitation := []byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m","output":[` +
		`{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"hmm"}]},` +
		`{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"\n\nSee the site",` +
		`"annotations":[{"type":"url_citation","url":"https://example.com","title":"site","start_index":10,"end_index":14,"provider_note":"ranked #1"}]}]}],` +
		`"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`)
	result, err = engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, responsesWithCitation, nil)
	require.NoError(t, err)
	require.False(t, result.Envelope.ResponseReencodeRequired, "the citation is shifted and replays with the rest of the reply")
	require.True(t, result.Envelope.CanReplay(llmprotocol.OpenAIResponsesV1, result.Response.Generation, llmprotocol.DefaultPolicy(), true))
	require.Contains(t, string(result.Envelope.Response), `"start_index":8`)
	require.Contains(t, string(result.Envelope.Response), `"end_index":12`)
	require.Contains(t, string(result.Envelope.Response), `"provider_note":"ranked #1"`, "a member the annotation wire does not declare survives the shift")
	answer, _ := wireAnswerAndReasoning(t, llmprotocol.OpenAIResponsesV1, result.Body)
	require.Equal(t, "See the site", answer, string(result.Body))
}
