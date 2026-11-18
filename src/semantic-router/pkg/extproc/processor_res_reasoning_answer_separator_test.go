package extproc

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"unicode"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vLLM's qwen3 reasoning parser leaves the blank line the model writes after
// its closing think tag at the start of the answer (see the vLLM 0.33 capture
// in protocolcodec/testdata/providers). The Router must not hand it to a client.
const (
	separatorTestThought     = "The image is one solid red square, so the answer is red."
	separatorTestFingerprint = "vllm-0.31.0-1c103593"
)

var separatorTestClients = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1,
}

func separatorBufferedReply(t *testing.T, reasoning, content string) []byte {
	t.Helper()
	message := map[string]any{"role": "assistant", "content": content}
	if reasoning != "" {
		message["reasoning"] = reasoning
	}
	body, err := json.Marshal(map[string]any{
		"id": "chatcmpl-separator", "object": "chat.completion", "created": 1794416301, "model": "qwen/qwen3.8-27b",
		"choices": []any{map[string]any{"index": 0, "message": message, "logprobs": nil, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
	})
	require.NoError(t, err)
	return body
}

func separatorContext(t *testing.T, client llmprotocol.WireFormat, stream bool) *RequestContext {
	t.Helper()
	return &RequestContext{
		SourceFormat: client, TargetFormat: llmprotocol.OpenAIChatV1, RequestModel: "public", TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public", Stream: stream},
	}
}

// clientBodyOf is what the client receives: the mutated body, or the provider
// body when the Router forwards it untouched.
func clientBodyOf(reply *ext_proc.ProcessingResponse, provider []byte) []byte {
	if mutation := reply.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		return mutation.GetBody()
	}
	return provider
}

type separatorPlayback struct {
	answer    string
	reasoning string
}

func separatorBufferedPlayback(t *testing.T, client llmprotocol.WireFormat, body []byte) separatorPlayback {
	t.Helper()
	var wire struct {
		Choices []struct {
			Message struct {
				Content   *string `json:"content"`
				Reasoning string  `json:"reasoning"`
			} `json:"message"`
		} `json:"choices"`
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Thinking string `json:"thinking"`
		} `json:"content"`
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(body, &wire), string(body))
	var playback separatorPlayback
	switch client {
	case llmprotocol.AnthropicMessagesV1:
		for _, block := range wire.Content {
			playback.answer += block.Text
			playback.reasoning += block.Thinking
		}
	case llmprotocol.OpenAIResponsesV1:
		for _, item := range wire.Output {
			for _, part := range item.Content {
				if item.Type == "reasoning" {
					playback.reasoning += part.Text
				} else {
					playback.answer += part.Text
				}
			}
		}
	default:
		require.Len(t, wire.Choices, 1, string(body))
		if wire.Choices[0].Message.Content != nil {
			playback.answer = *wire.Choices[0].Message.Content
		}
		playback.reasoning = wire.Choices[0].Message.Reasoning
	}
	return playback
}

func separatorChunk(delta string, finish string) string {
	finishReason := "null"
	if finish != "" {
		finishReason = `"` + finish + `"`
	}
	return `data: {"id":"chatcmpl-separator","object":"chat.completion.chunk","created":1794416301,"model":"qwen/qwen3.8-27b",` +
		`"choices":[{"index":0,"delta":` + delta + `,"logprobs":null,"finish_reason":` + finishReason + `}]}` + "\n\n"
}

func separatorStreamFrames(reasoning []string, content []string) []string {
	frames := []string{separatorChunk(`{"role":"assistant","content":""}`, "")}
	for _, piece := range reasoning {
		encoded, _ := json.Marshal(piece)
		frames = append(frames, separatorChunk(`{"reasoning":`+string(encoded)+`}`, ""))
	}
	for _, piece := range content {
		encoded, _ := json.Marshal(piece)
		frames = append(frames, separatorChunk(`{"content":`+string(encoded)+`}`, ""))
	}
	frames = append(frames, separatorChunk(`{}`, "stop"))
	frames = append(frames,
		`data: {"id":"chatcmpl-separator","object":"chat.completion.chunk","created":1794416301,"model":"qwen/qwen3.8-27b","choices":[],`+
			`"usage":{"prompt_tokens":20,"completion_tokens":12,"total_tokens":32},"system_fingerprint":"`+separatorTestFingerprint+`"}`+"\n\n",
		"data: [DONE]\n\n")
	return frames
}

// streamSeparatorThroughRouter sends each provider frame through the response
// stream handler and returns exactly what the client receives.
func streamSeparatorThroughRouter(t *testing.T, client llmprotocol.WireFormat, frames []string) string {
	t.Helper()
	router := &OpenAIRouter{}
	ctx := separatorContext(t, client, true)
	var received strings.Builder
	for index, frame := range frames {
		reply := router.handleSemanticStreamingResponseBody([]byte(frame), index == len(frames)-1, ctx)
		require.Nil(t, reply.GetImmediateResponse())
		if mutation := reply.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			received.Write(mutation.GetBody())
		} else {
			received.WriteString(frame)
		}
	}
	require.False(t, ctx.StreamingAborted, received.String())
	require.NotContains(t, received.String(), "event: error", received.String())
	return received.String()
}

// separatorStreamPlayback joins the answer and reasoning deltas a client sees.
func separatorStreamPlayback(t *testing.T, client llmprotocol.WireFormat, received string) separatorPlayback {
	t.Helper()
	var playback separatorPlayback
	for _, frame := range strings.Split(received, "\n\n") {
		var data string
		for _, line := range strings.Split(frame, "\n") {
			if strings.HasPrefix(line, "data: ") {
				data = strings.TrimPrefix(line, "data: ")
			}
		}
		if data == "" || data == "[DONE]" {
			continue
		}
		switch client {
		case llmprotocol.AnthropicMessagesV1:
			var event struct {
				Delta struct {
					Text     string `json:"text"`
					Thinking string `json:"thinking"`
				} `json:"delta"`
			}
			require.NoError(t, json.Unmarshal([]byte(data), &event), data)
			playback.answer += event.Delta.Text
			playback.reasoning += event.Delta.Thinking
		case llmprotocol.OpenAIResponsesV1:
			var event struct {
				Type  string `json:"type"`
				Delta string `json:"delta"`
			}
			require.NoError(t, json.Unmarshal([]byte(data), &event), data)
			switch event.Type {
			case "response.output_text.delta":
				playback.answer += event.Delta
			case "response.reasoning_text.delta":
				playback.reasoning += event.Delta
			}
		default:
			var event struct {
				Choices []struct {
					Delta struct {
						Content   *string `json:"content"`
						Reasoning string  `json:"reasoning"`
					} `json:"delta"`
				} `json:"choices"`
			}
			require.NoError(t, json.Unmarshal([]byte(data), &event), data)
			for _, choice := range event.Choices {
				if choice.Delta.Content != nil {
					playback.answer += *choice.Delta.Content
				}
				playback.reasoning += choice.Delta.Reasoning
			}
		}
	}
	return playback
}

func TestBufferedReasoningReplyDropsTheAnswerSeparator(t *testing.T) {
	for _, client := range separatorTestClients {
		t.Run(string(client), func(t *testing.T) {
			body := separatorBufferedReply(t, separatorTestThought, "\n\nRed")
			reply := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, separatorContext(t, client, false), 0, false)
			require.Nil(t, reply.GetImmediateResponse())
			received := clientBodyOf(reply, body)
			playback := separatorBufferedPlayback(t, client, received)
			t.Logf("%s client answer = %q", client, playback.answer)
			require.Equal(t, "Red", playback.answer, string(received))
			require.Equal(t, separatorTestThought, playback.reasoning, string(received))
		})
	}
}

// A Chat client of a Chat backend loses the model server's own members and the
// separator, and keeps every other member the provider sent.
func TestBufferedReasoningReplyKeepsTheRestOfAChatBody(t *testing.T) {
	var original, forwarded map[string]any
	require.NoError(t, json.Unmarshal(separatorBufferedReply(t, separatorTestThought, "\n\nRed"), &original))
	original["system_fingerprint"] = separatorTestFingerprint
	body, err := json.Marshal(original)
	require.NoError(t, err)
	reply := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, separatorContext(t, llmprotocol.OpenAIChatV1, false), 0, false)
	received := clientBodyOf(reply, body)
	require.NoError(t, json.Unmarshal(received, &forwarded), string(received))
	choice := func(object map[string]any) map[string]any { return object["choices"].([]any)[0].(map[string]any) }
	message := func(object map[string]any) map[string]any { return choice(object)["message"].(map[string]any) }
	require.Equal(t, "Red", message(forwarded)["content"])
	message(original)["content"], message(forwarded)["content"] = nil, nil
	delete(original, "system_fingerprint")
	require.Equal(t, original, forwarded, "only the answer text and the server's own members may change")
}

func TestBufferedReplyWithoutTheSeparatorIsUntouched(t *testing.T) {
	for name, test := range map[string]struct{ reasoning, content string }{
		"deepseek style, clean answer after reasoning": {separatorTestThought, "Red"},
		"no reasoning, whitespace is part of the text": {"", "\n\nRed"},
		"no reasoning, indented answer":                {"", "    return x"},
	} {
		for _, client := range separatorTestClients {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				body := separatorBufferedReply(t, test.reasoning, test.content)
				reply := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, separatorContext(t, client, false), 0, false)
				require.Nil(t, reply.GetImmediateResponse())
				if client == llmprotocol.OpenAIChatV1 {
					require.Nil(t, reply.GetResponseBody().GetResponse().GetBodyMutation(), "a same-format reply is forwarded as sent")
				}
				playback := separatorBufferedPlayback(t, client, clientBodyOf(reply, body))
				require.Equal(t, test.content, playback.answer)
				require.Equal(t, test.reasoning, playback.reasoning)
			})
		}
	}
}

func TestStreamedReasoningDropsTheAnswerSeparator(t *testing.T) {
	reasoning := []string{"The image is one solid red square,", " so the answer is red."}
	for name, content := range map[string][]string{
		"separator in its own delta":     {"\n\n", "Red"},
		"separator split over two":       {"\n", "\n", "Red"},
		"separator joined to the answer": {"\n\nRed"},
		"separator split with the word":  {"\n", "\nRed", " square"},
	} {
		for _, client := range separatorTestClients {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				received := streamSeparatorThroughRouter(t, client, separatorStreamFrames(reasoning, content))
				playback := separatorStreamPlayback(t, client, received)
				t.Logf("%s client answer = %q", client, playback.answer)
				require.Equal(t, strings.TrimLeft(strings.Join(content, ""), "\n"), playback.answer, received)
				require.Equal(t, strings.Join(reasoning, ""), playback.reasoning, received)
			})
		}
	}
}

func TestStreamedReplyWithoutTheSeparatorIsUntouched(t *testing.T) {
	for name, test := range map[string]struct{ reasoning, content []string }{
		"deepseek style, clean answer after reasoning": {[]string{"thought"}, []string{"Re", "d"}},
		"no reasoning, whitespace is part of the text": {nil, []string{"\n\n", "Red"}},
		"no reasoning, indented answer":                {nil, []string{"    ", "return x"}},
	} {
		for _, client := range separatorTestClients {
			t.Run(name+"/"+string(client), func(t *testing.T) {
				received := streamSeparatorThroughRouter(t, client, separatorStreamFrames(test.reasoning, test.content))
				playback := separatorStreamPlayback(t, client, received)
				require.Equal(t, strings.Join(test.content, ""), playback.answer, received)
				require.Equal(t, strings.Join(test.reasoning, ""), playback.reasoning, received)
			})
		}
	}
}

// vLLM serves Messages and Responses too, and a client of the same format gets
// the backend's reply as it is unless the Router has to change it.
func TestBufferedSameFormatBackendReplyDropsTheAnswerSeparator(t *testing.T) {
	for format, reply := range map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` +
			`{"type":"thinking","thinking":"` + separatorTestThought + `","signature":"c2ln"},{"type":"text","text":"\n\nRed"}],` +
			`"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":12}}`,
		llmprotocol.OpenAIResponsesV1: `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m","output":[` +
			`{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"` + separatorTestThought + `"}]},` +
			`{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"\n\nRed","annotations":[]}]}],` +
			`"usage":{"input_tokens":20,"output_tokens":12,"total_tokens":32}}`,
	} {
		t.Run(string(format), func(t *testing.T) {
			ctx := separatorContext(t, format, false)
			ctx.TargetFormat = format
			result := (&OpenAIRouter{}).handleNonStreamingResponseBody([]byte(reply), ctx, 0, false)
			require.Nil(t, result.GetImmediateResponse())
			playback := separatorBufferedPlayback(t, format, clientBodyOf(result, []byte(reply)))
			require.Equal(t, "Red", playback.answer, string(clientBodyOf(result, []byte(reply))))
			require.Equal(t, separatorTestThought, playback.reasoning)
		})
	}
}

// The vLLM 0.33 captures (Qwen3-0.6B, --reasoning-parser qwen3) hold the
// separator exactly as the server sent it, buffered and streamed.
func TestVLLMReasoningCapturesReachEveryClientWithoutTheSeparator(t *testing.T) {
	buffered, err := os.ReadFile("../protocolcodec/testdata/providers/vllm-0.33-chat-reasoning-out.json")
	require.NoError(t, err)
	var capture struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(buffered, &capture))
	sent := capture.Choices[0].Message
	require.True(t, strings.HasPrefix(sent.Content, "\n\n"), "the capture keeps the separator")
	answer := strings.TrimLeftFunc(sent.Content, unicode.IsSpace)

	for _, client := range separatorTestClients {
		t.Run("buffered/"+string(client), func(t *testing.T) {
			reply := (&OpenAIRouter{}).handleNonStreamingResponseBody(buffered, separatorContext(t, client, false), 0, false)
			require.Nil(t, reply.GetImmediateResponse())
			playback := separatorBufferedPlayback(t, client, clientBodyOf(reply, buffered))
			require.Equal(t, answer, playback.answer)
			require.Equal(t, sent.Reasoning, playback.reasoning)
		})
		t.Run("streamed/"+string(client), func(t *testing.T) {
			frames := ollamaStreamFrames(t, "vllm-0.33-chat-reasoning-stream.sse")
			var wantAnswer, wantReasoning string
			for _, frame := range frames {
				data := strings.TrimSpace(strings.TrimPrefix(frame, "data:"))
				var chunk struct {
					Choices []struct {
						Delta struct {
							Content   string `json:"content"`
							Reasoning string `json:"reasoning"`
						} `json:"delta"`
					} `json:"choices"`
				}
				if json.Unmarshal([]byte(data), &chunk) == nil {
					for _, choice := range chunk.Choices {
						wantAnswer += choice.Delta.Content
						wantReasoning += choice.Delta.Reasoning
					}
				}
			}
			require.True(t, strings.HasPrefix(wantAnswer, "\n\n"), "the capture keeps the separator")
			received := streamSeparatorThroughRouter(t, client, frames)
			playback := separatorStreamPlayback(t, client, received)
			require.Equal(t, strings.TrimLeftFunc(wantAnswer, unicode.IsSpace), playback.answer, received)
			require.Equal(t, wantReasoning, playback.reasoning, received)
		})
	}
}
