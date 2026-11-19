package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vLLM's qwen3 reasoning parser leaves the blank line the model writes after
// its closing think tag at the start of the answer (testdata/providers has the
// vLLM 0.33 captures). The Router drops it, and nothing else.
const separatorThought = "The image is one solid red square, so the answer is red."

var separatorTargets = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
}

func separatorChatReply(t *testing.T, reasoning, content string, decorate func(message, reply map[string]any)) []byte {
	t.Helper()
	message := map[string]any{"role": "assistant", "content": content}
	if reasoning != "" {
		message["reasoning"] = reasoning
	}
	reply := map[string]any{
		"id": "chatcmpl-separator", "object": "chat.completion", "created": 1794416301, "model": "m",
		"choices": []any{map[string]any{"index": 0, "message": message, "logprobs": nil, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
	}
	if decorate != nil {
		decorate(message, reply)
	}
	body, err := json.Marshal(reply)
	require.NoError(t, err)
	return body
}

// answerAndReasoning joins what a neutral reply says and thinks.
func answerAndReasoning(response llmprotocol.Response) (string, string) {
	var answer, reasoning string
	for _, item := range response.Output {
		for _, block := range item.Content {
			switch block.Kind {
			case llmprotocol.ContentText:
				answer += block.Text
			case llmprotocol.ContentReasoning:
				reasoning += block.Text
			}
		}
	}
	return answer, reasoning
}

func wireAnswerAndReasoning(t *testing.T, format llmprotocol.WireFormat, body []byte) (string, string) {
	t.Helper()
	var wire struct {
		Choices []struct {
			Message struct {
				Content          *string `json:"content"`
				Reasoning        string  `json:"reasoning"`
				ReasoningContent string  `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Content []struct {
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		} `json:"content"`
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(body, &wire), string(body))
	var answer, reasoning string
	switch format {
	case llmprotocol.AnthropicMessagesV1:
		for _, block := range wire.Content {
			answer += block.Text
			reasoning += block.Thinking
		}
	case llmprotocol.OpenAIResponsesV1:
		for _, item := range wire.Output {
			for _, part := range item.Content {
				if item.Type == "reasoning" {
					reasoning += part.Text
				} else {
					answer += part.Text
				}
			}
		}
	default:
		require.NotEmpty(t, wire.Choices, string(body))
		if wire.Choices[0].Message.Content != nil {
			answer = *wire.Choices[0].Message.Content
		}
		reasoning = wire.Choices[0].Message.Reasoning + wire.Choices[0].Message.ReasoningContent
	}
	return answer, reasoning
}

func TestTranslateResponseDropsTheAnswerSeparatorAfterReasoning(t *testing.T) {
	body := separatorChatReply(t, separatorThought, "\n\nRed", nil)
	for _, target := range separatorTargets {
		t.Run(string(target), func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
			require.NoError(t, err)
			answer, reasoning := answerAndReasoning(result.Response)
			require.Equal(t, "Red", answer)
			require.Equal(t, separatorThought, reasoning)
			answer, reasoning = wireAnswerAndReasoning(t, target, result.Body)
			require.Equal(t, "Red", answer, string(result.Body))
			require.Equal(t, separatorThought, reasoning, string(result.Body))
		})
	}
}

// A Chat client of a Chat backend loses the model server's own members and the
// separator, and keeps every other member the provider sent.
func TestChatReplayKeepsTheProviderBodyBesidesTheAnswer(t *testing.T) {
	body := separatorChatReply(t, separatorThought, "\n\nRed", func(message, reply map[string]any) {
		reply["system_fingerprint"] = "vllm-0.31.0-1c103593"
		reply["kv_transfer_params"] = map[string]any{"remote_engine_id": "engine-1"}
		reply["prompt_token_ids"] = []int{1, 2, 3}
		choice := reply["choices"].([]any)[0].(map[string]any)
		choice["token_ids"] = []int{271, 3731}
		choice["logprobs"] = map[string]any{"content": []any{map[string]any{"token": "\n\n", "logprob": -0.1, "top_logprobs": []any{}}}}
	})
	engine := NewBuiltinEngine()
	result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
	require.NoError(t, err)
	require.False(t, result.Envelope.ResponseReencodeRequired)
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, result.Response, result.Envelope)
	require.NoError(t, err)
	require.JSONEq(t, string(result.Body), string(encoded.Body))

	var original, forwarded map[string]any
	require.NoError(t, json.Unmarshal(body, &original))
	require.NoError(t, json.Unmarshal(result.Body, &forwarded))
	choice := func(object map[string]any) map[string]any {
		return object["choices"].([]any)[0].(map[string]any)
	}
	message := func(object map[string]any) map[string]any { return choice(object)["message"].(map[string]any) }
	require.Equal(t, "Red", message(forwarded)["content"])
	message(original)["content"], message(forwarded)["content"] = nil, nil
	for _, member := range []string{"system_fingerprint", "kv_transfer_params", "prompt_token_ids"} {
		delete(original, member)
	}
	delete(choice(original), "token_ids")
	require.Equal(t, original, forwarded, "only the answer text and the server's own members may change")
}

// A Messages client of a Messages backend keeps members its own wire contract
// does not carry through a full re-encode, such as the container a gateway
// attaches or extra usage accounting, because the patch never goes through
// that contract at all.
func TestMessagesReplayKeepsTheProviderBodyBesidesTheAnswer(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` +
		`{"type":"thinking","thinking":"` + separatorThought + `","signature":""},{"type":"text","text":"\n\nRed"}],` +
		`"stop_reason":"end_turn","stop_sequence":null,"container":{"id":"container_123"},` +
		`"usage":{"input_tokens":20,"output_tokens":12,"speed":"45.2","total_cost":0.0031}}`)
	engine := NewBuiltinEngine()
	result, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, body, nil)
	require.NoError(t, err)
	require.False(t, result.Envelope.ResponseReencodeRequired)
	encoded, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, result.Response, result.Envelope)
	require.NoError(t, err)
	require.JSONEq(t, string(result.Body), string(encoded.Body))

	var original, forwarded map[string]any
	require.NoError(t, json.Unmarshal(body, &original))
	require.NoError(t, json.Unmarshal(result.Body, &forwarded))
	require.Equal(t, "container_123", forwarded["container"].(map[string]any)["id"], "the gateway's container survives the patch")
	require.Equal(t, "45.2", forwarded["usage"].(map[string]any)["speed"])
	require.InDelta(t, 0.0031, forwarded["usage"].(map[string]any)["total_cost"], 0.0001)

	content := func(object map[string]any) []any { return object["content"].([]any) }
	text := func(object map[string]any) map[string]any { return content(object)[1].(map[string]any) }
	require.Equal(t, "Red", text(forwarded)["text"])
	text(original)["text"], text(forwarded)["text"] = nil, nil
	require.Equal(t, original, forwarded, "only the answer text may change")
}

// A Responses client of a Responses backend keeps request-scoped metadata its
// own wire contract cannot carry through a full re-encode, for the same reason.
func TestResponsesReplayKeepsTheProviderBodyBesidesTheAnswer(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m","output":[` +
		`{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"` + separatorThought + `"}]},` +
		`{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"\n\nRed","annotations":[]}]}],` +
		`"metadata":{"trace_id":"trace-123"},"usage":{"input_tokens":20,"output_tokens":12,"total_tokens":32}}`)
	engine := NewBuiltinEngine()
	result, err := engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, body, nil)
	require.NoError(t, err)
	require.False(t, result.Envelope.ResponseReencodeRequired)
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, result.Response, result.Envelope)
	require.NoError(t, err)
	require.JSONEq(t, string(result.Body), string(encoded.Body))

	var original, forwarded map[string]any
	require.NoError(t, json.Unmarshal(body, &original))
	require.NoError(t, json.Unmarshal(result.Body, &forwarded))
	require.Equal(t, "trace-123", forwarded["metadata"].(map[string]any)["trace_id"], "request-scoped metadata survives the patch")

	message := func(object map[string]any) map[string]any { return object["output"].([]any)[1].(map[string]any) }
	text := func(object map[string]any) map[string]any {
		return message(object)["content"].([]any)[0].(map[string]any)
	}
	require.Equal(t, "Red", text(forwarded)["text"])
	text(original)["text"], text(forwarded)["text"] = nil, nil
	require.Equal(t, original, forwarded, "only the answer text may change")
}

func TestTranslateResponseLeavesOtherRepliesAlone(t *testing.T) {
	for name, test := range map[string]struct{ reasoning, content string }{
		"deepseek style, clean answer after reasoning": {separatorThought, "Red"},
		"no reasoning, whitespace is part of the text": {"", "\n\nRed"},
		"no reasoning, indented answer":                {"", "    return x"},
		"reasoning only":                               {separatorThought, ""},
	} {
		t.Run(name, func(t *testing.T) {
			body := separatorChatReply(t, test.reasoning, test.content, nil)
			engine := NewBuiltinEngine()
			for _, target := range separatorTargets {
				result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
				require.NoError(t, err)
				answer, reasoning := answerAndReasoning(result.Response)
				require.Equal(t, test.content, answer)
				require.Equal(t, test.reasoning, reasoning)
				require.False(t, result.Envelope.ResponseReencodeRequired)
				require.EqualValues(t, 1, result.Response.Generation)
			}
			result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
			require.NoError(t, err)
			require.Equal(t, string(body), string(result.Body), "a same-format reply is replayed as sent")
		})
	}
}

func TestTranslateResponseTrimsEachChoice(t *testing.T) {
	body := separatorChatReply(t, separatorThought, "\n\nRed", func(_, reply map[string]any) {
		second := map[string]any{"index": 1, "message": map[string]any{"role": "assistant", "content": "\n\nBlue"}, "finish_reason": "stop"}
		third := map[string]any{"index": 2, "message": map[string]any{"role": "assistant", "content": "\n\nGreen", "reasoning": "green thought"}, "finish_reason": "stop"}
		reply["choices"] = append(reply["choices"].([]any), second, third)
	})
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1} {
		result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
		require.NoError(t, err)
		var reply struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		require.NoError(t, json.Unmarshal(result.Body, &reply))
		var answers []string
		for _, choice := range reply.Choices {
			answers = append(answers, choice.Message.Content)
		}
		require.Equal(t, []string{"Red", "\n\nBlue", "Green"}, answers, "only a choice with reasoning is trimmed")
	}
}

// A request can hide the reasoning from the client; the provider still split it
// from the answer, so the separator goes all the same.
func TestTranslateResponseTrimsBeforeRequestPolicyHidesTheReasoning(t *testing.T) {
	hideReasoning := func(response *llmprotocol.Response) error {
		for index := range response.Output {
			var kept []llmprotocol.Content
			for _, block := range response.Output[index].Content {
				if block.Kind != llmprotocol.ContentReasoning {
					kept = append(kept, block)
				}
			}
			response.Output[index].Content = kept
		}
		return nil
	}
	body := separatorChatReply(t, separatorThought, "\n\nRed", nil)
	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, hideReasoning)
	require.NoError(t, err)
	answer, _ := wireAnswerAndReasoning(t, llmprotocol.AnthropicMessagesV1, result.Body)
	require.Equal(t, "Red", answer, string(result.Body))
	require.NotContains(t, string(result.Body), `"type":"thinking"`)
}

func TestTranslateResponseKeepsToolCallsAfterTheSeparator(t *testing.T) {
	call := map[string]any{"id": "call-1", "type": "function", "function": map[string]any{"name": "lookup", "arguments": "{}"}}
	body := separatorChatReply(t, separatorThought, "\n\n", func(message, _ map[string]any) {
		message["tool_calls"] = []any{call}
	})
	engine := NewBuiltinEngine()
	for _, target := range separatorTargets {
		result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
		require.NoError(t, err)
		require.Contains(t, string(result.Body), "lookup")
		answer, reasoning := answerAndReasoning(result.Response)
		require.Empty(t, answer, "nothing is left of the separator")
		require.Equal(t, separatorThought, reasoning)
		if target == llmprotocol.AnthropicMessagesV1 {
			require.NotContains(t, string(result.Body), `"type":"text"`, "no empty text block")
		}
	}
}

// Offsets of URL citations index into the text as the provider sent it.
func TestTrimmedAnswerKeepsItsCitationsInRange(t *testing.T) {
	body := separatorChatReply(t, separatorThought, "\n\nSee the site", func(message, _ map[string]any) {
		message["annotations"] = []any{map[string]any{"type": "url_citation", "url_citation": map[string]any{
			"url": "https://example.com", "title": "site", "start_index": 10, "end_index": 14,
		}}}
	})
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
			require.NoError(t, err)
			answer, _ := answerAndReasoning(result.Response)
			require.Equal(t, "See the site", answer)
			citation := result.Response.Output[0].Content[1].Citations[0]
			require.Equal(t, []int64{8, 12}, []int64{citation.StartIndex, citation.EndIndex})
			text := []rune(answer)
			require.Equal(t, "site", string(text[citation.StartIndex:citation.EndIndex]))
		})
	}
	result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
	require.NoError(t, err)
	require.Contains(t, string(result.Body), `"start_index":8`)
	require.Contains(t, string(result.Body), `"end_index":12`)
}

func TestMessagesBackendReplyDropsTheAnswerSeparator(t *testing.T) {
	reply := func(signature string) []byte {
		return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` +
			`{"type":"thinking","thinking":"` + separatorThought + `","signature":"` + signature + `"},{"type":"text","text":"\n\nRed"}],` +
			`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":12}}`)
	}
	engine := NewBuiltinEngine()
	// A signed thinking block is left untouched by the in-place patch, so a
	// Messages client of a Messages backend keeps the signature and still
	// replays rather than paying a full re-encode.
	for target, signature := range map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: "c2ln", llmprotocol.OpenAIChatV1: "", llmprotocol.OpenAIResponsesV1: "",
	} {
		t.Run(string(target), func(t *testing.T) {
			result, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, reply(signature), nil)
			require.NoError(t, err)
			answer, reasoning := wireAnswerAndReasoning(t, target, result.Body)
			require.Equal(t, "Red", answer, string(result.Body))
			require.Equal(t, separatorThought, reasoning, string(result.Body))
			if target == llmprotocol.AnthropicMessagesV1 {
				require.False(t, result.Envelope.ResponseReencodeRequired, "the signed thinking block is untouched and replays fine")
				require.Contains(t, string(result.Body), `"signature":"c2ln"`)
			}
		})
	}
}

func TestResponsesBackendReplyDropsTheAnswerSeparator(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m","output":[` +
		`{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"` + separatorThought + `"}]},` +
		`{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"\n\nRed","annotations":[]}]}],` +
		`"usage":{"input_tokens":20,"output_tokens":12,"total_tokens":32}}`)
	engine := NewBuiltinEngine()
	for _, target := range separatorTargets {
		t.Run(string(target), func(t *testing.T) {
			result, err := engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, target, body, nil)
			require.NoError(t, err)
			answer, _ := wireAnswerAndReasoning(t, target, result.Body)
			require.Equal(t, "Red", answer, string(result.Body))
		})
	}
}

// DecodeResponse feeds a buffered provider answer into a synthesized stream.
func TestDecodeResponseDropsTheAnswerSeparator(t *testing.T) {
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, separatorChatReply(t, separatorThought, "\n\nRed", nil))
	require.NoError(t, err)
	answer, reasoning := answerAndReasoning(response)
	require.Equal(t, "Red", answer)
	require.Equal(t, separatorThought, reasoning)
}

// separatorChatStream is a vLLM Chat stream: the role delta, the reasoning
// deltas, the answer deltas and the finish.
func separatorChatStream(reasoning, content []string, extra ...string) string {
	deltas := []string{`{"role":"assistant","content":""}`}
	for _, piece := range reasoning {
		encoded, _ := json.Marshal(piece)
		deltas = append(deltas, `{"reasoning":`+string(encoded)+`}`)
	}
	for _, piece := range content {
		encoded, _ := json.Marshal(piece)
		deltas = append(deltas, `{"content":`+string(encoded)+`}`)
	}
	return chatStream(append(append(deltas, extra...), `{}`)...)
}

type separatorPlayback struct {
	wire      string
	answer    string
	reasoning string
}

// separatorStreamThrough pushes a stream through the engine and returns what
// the client receives and what the accepted neutral events say.
func separatorStreamThrough(
	t *testing.T,
	source, target llmprotocol.WireFormat,
	body string,
	mutation StreamEventMutation,
) separatorPlayback {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStreamWithMutation(source, target, llmprotocol.StreamContext{
		Source: source, Target: target, PublicModel: "m",
	}, mutation)
	require.NoError(t, err)
	var playback separatorPlayback
	var wire strings.Builder
	collect := func(frames [][]byte, events []llmprotocol.Event) {
		for _, frame := range frames {
			wire.Write(frame)
		}
		for _, event := range events {
			switch {
			case event.Type == llmprotocol.EventOutputTextDelta && (event.Content == nil || event.Content.Kind == llmprotocol.ContentText):
				playback.answer += event.Delta
			case event.Type == llmprotocol.EventReasoningDelta:
				playback.reasoning += event.Delta
			}
		}
	}
	frames, events, _, pushErr := stream.Push([]byte(body))
	collect(frames, events)
	frames, events, _, finalErr := stream.Finalize(pushErr)
	collect(frames, events)
	require.NoError(t, pushErr, wire.String())
	require.NoError(t, finalErr, wire.String())
	playback.wire = wire.String()
	require.NotContains(t, playback.wire, "event: error", playback.wire)
	return playback
}

// clientStreamText joins the answer and reasoning deltas a client sees.
func clientStreamText(t *testing.T, target llmprotocol.WireFormat, wire string) (string, string) {
	t.Helper()
	var answer, reasoning string
	for _, frame := range strings.Split(wire, "\n\n") {
		var data string
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		switch target {
		case llmprotocol.AnthropicMessagesV1:
			var event struct {
				Delta struct {
					Text     string `json:"text"`
					Thinking string `json:"thinking"`
				} `json:"delta"`
			}
			require.NoError(t, json.Unmarshal([]byte(data), &event), data)
			answer, reasoning = answer+event.Delta.Text, reasoning+event.Delta.Thinking
		case llmprotocol.OpenAIResponsesV1:
			var event struct {
				Type  string `json:"type"`
				Delta string `json:"delta"`
			}
			require.NoError(t, json.Unmarshal([]byte(data), &event), data)
			switch event.Type {
			case "response.output_text.delta":
				answer += event.Delta
			case "response.reasoning_text.delta":
				reasoning += event.Delta
			}
		default:
			var event struct {
				Choices []struct {
					Delta struct {
						Content          *string `json:"content"`
						Reasoning        string  `json:"reasoning"`
						ReasoningContent string  `json:"reasoning_content"`
					} `json:"delta"`
				} `json:"choices"`
			}
			require.NoError(t, json.Unmarshal([]byte(data), &event), data)
			for _, choice := range event.Choices {
				if choice.Delta.Content != nil {
					answer += *choice.Delta.Content
				}
				reasoning += choice.Delta.Reasoning + choice.Delta.ReasoningContent
			}
		}
	}
	return answer, reasoning
}

func TestStreamDropsTheAnswerSeparatorAfterReasoning(t *testing.T) {
	reasoning := []string{"The image is one solid red square,", " so the answer is red."}
	for name, content := range map[string][]string{
		"separator in its own delta":     {"\n\n", "Red"},
		"separator split over two":       {"\n", "\n", "Red"},
		"separator joined to the answer": {"\n\nRed", " square"},
		"separator split with the word":  {"\n", "\nRed", " square"},
	} {
		for _, target := range separatorTargets {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				playback := separatorStreamThrough(t, llmprotocol.OpenAIChatV1, target, separatorChatStream(reasoning, content), nil)
				want := strings.TrimLeft(strings.Join(content, ""), "\n")
				require.Equal(t, want, playback.answer, "accepted events")
				require.Equal(t, strings.Join(reasoning, ""), playback.reasoning)
				answer, thought := clientStreamText(t, target, playback.wire)
				require.Equal(t, want, answer, playback.wire)
				require.Equal(t, strings.Join(reasoning, ""), thought, playback.wire)
			})
		}
	}
}

func TestStreamLeavesOtherRepliesAlone(t *testing.T) {
	for name, test := range map[string]struct{ reasoning, content []string }{
		"deepseek style, clean answer after reasoning": {[]string{"thought"}, []string{"Re", "d"}},
		"no reasoning, whitespace is part of the text": {nil, []string{"\n\n", "Red"}},
		"no reasoning, indented answer":                {nil, []string{"    ", "return x"}},
	} {
		for _, target := range separatorTargets {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				playback := separatorStreamThrough(t, llmprotocol.OpenAIChatV1, target, separatorChatStream(test.reasoning, test.content), nil)
				require.Equal(t, strings.Join(test.content, ""), playback.answer)
				require.Equal(t, strings.Join(test.reasoning, ""), playback.reasoning)
				answer, thought := clientStreamText(t, target, playback.wire)
				require.Equal(t, strings.Join(test.content, ""), answer, playback.wire)
				require.Equal(t, strings.Join(test.reasoning, ""), thought, playback.wire)
			})
		}
	}
}

// Reasoning can come after the first words of the answer (Mistral). The
// answer has started, so the delta after it is not the start of the answer.
func TestStreamKeepsAnAnswerThatStartedBeforeItsReasoning(t *testing.T) {
	stream := chatStream(`{"role":"assistant","content":"Hello"}`, `{"reasoning":"Check"}`, `{"content":" there"}`, `{"content":" friend"}`)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			playback := separatorStreamThrough(t, llmprotocol.OpenAIChatV1, target, stream, nil)
			require.Equal(t, "Hello there friend", playback.answer)
		})
	}
}

func TestStreamWithOnlyTheSeparatorOpensNoAnswer(t *testing.T) {
	for _, target := range separatorTargets {
		t.Run(string(target), func(t *testing.T) {
			playback := separatorStreamThrough(t, llmprotocol.OpenAIChatV1, target, separatorChatStream([]string{"thought"}, []string{"\n\n"}), nil)
			require.Empty(t, playback.answer)
			answer, _ := clientStreamText(t, target, playback.wire)
			require.Empty(t, answer)
			switch target {
			case llmprotocol.AnthropicMessagesV1:
				require.NotContains(t, playback.wire, `"type":"text"`, "no empty text block")
			case llmprotocol.OpenAIResponsesV1:
				require.NotContains(t, playback.wire, `"type":"message"`, "no empty message item")
			}
		})
	}
}

func TestStreamKeepsToolCallsAfterTheSeparator(t *testing.T) {
	call := `{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}`
	stream := separatorChatStream([]string{"thought"}, []string{"\n\n"}, call)
	for _, target := range separatorTargets {
		t.Run(string(target), func(t *testing.T) {
			playback := separatorStreamThrough(t, llmprotocol.OpenAIChatV1, target, stream, nil)
			require.Empty(t, playback.answer)
			require.Equal(t, "thought", playback.reasoning)
			require.Contains(t, playback.wire, "lookup")
			switch target {
			case llmprotocol.AnthropicMessagesV1:
				require.NotContains(t, playback.wire, `"type":"text"`, "no empty text block")
			case llmprotocol.OpenAIResponsesV1:
				require.NotContains(t, playback.wire, `"type":"message"`, "no empty message item")
			}
		})
	}
}

// A request can hide the reasoning from the client; the provider still split it
// from the answer, so the separator goes all the same.
func TestStreamTrimsBeforeRequestPolicyHidesTheReasoning(t *testing.T) {
	hideReasoning := func(event *llmprotocol.Event) error {
		if event.Type == llmprotocol.EventReasoningDelta {
			return ErrOmitStreamEvent
		}
		return nil
	}
	playback := separatorStreamThrough(t, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
		separatorChatStream([]string{"thought"}, []string{"\n\n", "Red"}), hideReasoning)
	require.Equal(t, "Red", playback.answer)
	require.NotContains(t, playback.wire, `"type":"thinking"`)
}

func TestStreamKeepsCitationsInRange(t *testing.T) {
	annotation := `{"annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"site","start_index":10,"end_index":14}}]}`
	stream := separatorChatStream([]string{"thought"}, []string{"\n\nSee the site"}, annotation)
	playback := separatorStreamThrough(t, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, stream, nil)
	require.Equal(t, "See the site", playback.answer)
	require.Contains(t, playback.wire, `"start_index":8`)
	require.Contains(t, playback.wire, `"end_index":12`)
}

func filterSeparatorStream(t *testing.T, frames ...string) string {
	t.Helper()
	filter := NewChatPublicStreamFilter(llmprotocol.DefaultPolicy().Limits.SSEFrameBytes, true)
	var out []byte
	for _, frame := range frames {
		filtered, err := filter.Push([]byte(frame))
		require.NoError(t, err)
		out = append(out, filtered...)
	}
	tail, err := filter.Finalize()
	require.NoError(t, err)
	return string(append(out, tail...))
}

func separatorFrames(deltas ...string) []string {
	frames := make([]string, 0, len(deltas)+1)
	for index, delta := range deltas {
		finish := "null"
		if index == len(deltas)-1 {
			finish = `"stop"`
		}
		frames = append(frames, `data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":`+
			delta+`,"logprobs":null,"finish_reason":`+finish+`}]}`+"\n\n")
	}
	return append(frames, "data: [DONE]\n\n")
}

// A same-format Chat stream is forwarded as the provider sent it, so the filter
// edits only the answer text of the chunk that carries the separator.
func TestChatPublicStreamFilterDropsTheAnswerSeparator(t *testing.T) {
	for name, content := range map[string][]string{
		"separator in its own delta":     {`"\n\n"`, `"Red"`},
		"separator split over two":       {`"\n"`, `"\n"`, `"Red"`},
		"separator joined to the answer": {`"\n\nRed"`, `" square"`},
	} {
		t.Run(name, func(t *testing.T) {
			deltas := []string{`{"role":"assistant","content":""}`, `{"reasoning":"thought"}`, `{"reasoning":" more","content":""}`}
			for _, piece := range content {
				deltas = append(deltas, `{"content":`+piece+`}`)
			}
			frames := separatorFrames(append(deltas, `{}`)...)
			out := filterSeparatorStream(t, frames...)
			answer, reasoning := clientStreamText(t, llmprotocol.OpenAIChatV1, out)
			want := strings.TrimLeft(strings.Join(content, ""), `"\n`)
			want = strings.ReplaceAll(want, `"`, "")
			require.Equal(t, want, answer, out)
			require.Equal(t, "thought more", reasoning, out)
			require.Equal(t, strings.Count(strings.Join(frames, ""), "data:"), strings.Count(out, "data:"), "every chunk is still sent")
			require.Equal(t, len(frames)-1, strings.Count(out, `"logprobs":null`), "the provider's other members stay")
		})
	}
}

// The filter also drops the model server's own members from the chunks whose
// separator it trims, and keeps the chunk's other members.
func TestChatPublicStreamFilterTrimsChunksThatCarryServerMembers(t *testing.T) {
	chunk := func(delta, finish string) string {
		return `data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"vllm-0.31.0-1c103593",` +
			`"choices":[{"index":0,"delta":` + delta + `,"logprobs":null,"finish_reason":` + finish + `,"token_ids":[271]}]}` + "\n\n"
	}
	out := filterSeparatorStream(t,
		chunk(`{"reasoning":"thought"}`, "null"), chunk(`{"content":"\n\nRed"}`, `"stop"`), "data: [DONE]\n\n")
	answer, reasoning := clientStreamText(t, llmprotocol.OpenAIChatV1, out)
	require.Equal(t, "Red", answer, out)
	require.Equal(t, "thought", reasoning, out)
	require.NotContains(t, out, "system_fingerprint")
	require.NotContains(t, out, "token_ids")
	require.Equal(t, 2, strings.Count(out, `"logprobs":null`), "the provider's other members stay")
}

// Streams with nothing to trim keep their bytes, apart from the null usage a
// client that asked for usage receives on every chunk.
func TestChatPublicStreamFilterLeavesOtherStreamsAlone(t *testing.T) {
	for name, test := range map[string][]string{
		"deepseek style, clean answer after reasoning": {`{"reasoning":"thought"}`, `{"content":"Re"}`, `{"content":"d"}`},
		"no reasoning, whitespace is part of the text": {`{"content":"\n\n"}`, `{"content":"Red"}`},
		"no reasoning, indented answer":                {`{"content":"    "}`, `{"content":"return x"}`},
	} {
		t.Run(name, func(t *testing.T) {
			frames := separatorFrames(append(test, `{}`)...)
			var want strings.Builder
			for _, frame := range frames[:len(frames)-1] {
				want.Write(withNullUsage([]byte(frame)))
			}
			want.WriteString(frames[len(frames)-1])
			require.Equal(t, want.String(), filterSeparatorStream(t, frames...))
		})
	}
}

// A chunk whose separator the filter trims carries the null usage too.
func TestChatPublicStreamFilterTrimmedChunksCarryNullUsage(t *testing.T) {
	out := filterSeparatorStream(t, separatorFrames(`{"reasoning":"thought"}`, `{"content":"\n\nRed"}`, `{}`)...)
	for index, usage := range chunkUsages(t, []byte(out)) {
		require.Equal(t, "null", usage, "chunk %d of %s", index, out)
	}
	answer, _ := clientStreamText(t, llmprotocol.OpenAIChatV1, out)
	require.Equal(t, "Red", answer, out)
}

func TestChatPublicStreamFilterShiftsLateAnnotations(t *testing.T) {
	annotation := `{"annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com","title":"site","start_index":10,"end_index":14}}]}`
	frames := separatorFrames(`{"reasoning":"thought"}`, `{"content":"\n\nSee the site"}`, annotation, `{}`)
	out := filterSeparatorStream(t, frames...)
	answer, _ := clientStreamText(t, llmprotocol.OpenAIChatV1, out)
	require.Equal(t, "See the site", answer)
	require.Contains(t, out, `"start_index":8`)
	require.Contains(t, out, `"end_index":12`)
}
