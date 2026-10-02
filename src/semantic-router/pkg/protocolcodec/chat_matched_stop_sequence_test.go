package protocolcodec

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Captured from vLLM 0.26.0 with Qwen/Qwen3.8-27B-FP8 and "stop": ["CHARLIE"].
// vLLM reports the matched stop string in the non-standard choices[].stop_reason.
func vllmStopSequenceCapture(t *testing.T, name string) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "providers", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func translateMatchedStopStream(t *testing.T, source, target llmprotocol.WireFormat, body string) string {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStreamWithMutation(source, target, llmprotocol.StreamContext{
		Source: source, Target: target, PublicModel: "m",
	}, nil)
	if err != nil {
		t.Fatalf("NewStreamWithMutation: %v", err)
	}
	var out strings.Builder
	frames, _, _, pushErr := stream.Push([]byte(body))
	for _, frame := range frames {
		out.Write(frame)
	}
	frames, _, _, finalErr := stream.Finalize(pushErr)
	for _, frame := range frames {
		out.Write(frame)
	}
	if pushErr != nil || finalErr != nil {
		t.Fatalf("stream failed: push=%v finalize=%v\n%s", pushErr, finalErr, out.String())
	}
	return out.String()
}

func TestChatStopSequenceReachesAnthropicClient(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, []byte(vllmStopSequenceCapture(t, "vllm-chat-stop-sequence-out.json")), nil)
	if err != nil {
		t.Fatalf("TranslateResponse: %v", err)
	}
	var message struct {
		StopReason   string  `json:"stop_reason"`
		StopSequence *string `json:"stop_sequence"`
	}
	if err := json.Unmarshal(result.Body, &message); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if message.StopReason != "stop_sequence" || message.StopSequence == nil || *message.StopSequence != "CHARLIE" {
		t.Fatalf("want stop_reason=stop_sequence stop_sequence=CHARLIE, got %s", result.Body)
	}
}

func TestChatStopSequenceReachesAnthropicStream(t *testing.T) {
	out := translateMatchedStopStream(t, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, vllmStopSequenceCapture(t, "vllm-chat-stop-sequence-stream.sse"))
	if !strings.Contains(out, `"stop_reason":"stop_sequence","stop_sequence":"CHARLIE"`) {
		t.Fatalf("message_delta lost the matched stop sequence:\n%s", out)
	}
}

// Controls: only a string stop_reason with finish_reason "stop" is a matched sequence.
func TestChatStopReasonWithoutMatchedSequenceStaysEndTurn(t *testing.T) {
	for name, stopReason := range map[string]string{"end of sequence": `null`, "stop token id": `151645`} {
		body := strings.Replace(vllmStopSequenceCapture(t, "vllm-chat-stop-sequence-out.json"), `"stop_reason":"CHARLIE"`, `"stop_reason":`+stopReason, 1)
		result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, []byte(body), nil)
		if err != nil {
			t.Fatalf("%s: TranslateResponse: %v", name, err)
		}
		if !strings.Contains(string(result.Body), `"stop_reason":"end_turn","stop_sequence":null`) {
			t.Fatalf("%s: want end_turn with null stop_sequence, got %s", name, result.Body)
		}
	}
}

func TestChatToChatKeepsStopReason(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(vllmStopSequenceCapture(t, "vllm-chat-stop-sequence-out.json")), nil)
	if err != nil {
		t.Fatalf("TranslateResponse: %v", err)
	}
	if !strings.Contains(string(result.Body), `"stop_reason":"CHARLIE"`) {
		t.Fatalf("Chat client lost vLLM's stop_reason: %s", result.Body)
	}
}

// A provider that repeats the terminal chunk (OpenRouter does, with usage) keeps the match.
func TestChatStopSequenceRepeatedTerminalChunk(t *testing.T) {
	final := "data: {\"id\":\"chatcmpl-90fa570aff8dc47d\",\"object\":\"chat.completion.chunk\",\"created\":1790715607,\"model\":\"Qwen/Qwen3.8-27B-FP8\",\"choices\":[{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"stop\"%s}]}\n\n"
	withRepeat := func(stopReason string) string {
		return strings.Replace(vllmStopSequenceCapture(t, "vllm-chat-stop-sequence-stream.sse"), "data: [DONE]", fmt.Sprintf(final, stopReason)+"data: [DONE]", 1)
	}
	out := translateMatchedStopStream(t, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, withRepeat(""))
	if !strings.Contains(out, `"stop_reason":"stop_sequence","stop_sequence":"CHARLIE"`) {
		t.Fatalf("repeat without stop_reason lost the match:\n%s", out)
	}
	// A repeat that names another sequence, or a stop token id, is a changed terminal.
	for _, stopReason := range []string{`,"stop_reason":"DELTA"`, `,"stop_reason":151645`} {
		stream, err := NewBuiltinEngine().NewStreamWithMutation(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
			Source: llmprotocol.OpenAIChatV1, Target: llmprotocol.AnthropicMessagesV1, PublicModel: "m",
		}, nil)
		if err != nil {
			t.Fatalf("NewStreamWithMutation: %v", err)
		}
		_, _, _, pushErr := stream.Push([]byte(withRepeat(stopReason)))
		var protocolErr *llmprotocol.ProtocolError
		if !errors.As(pushErr, &protocolErr) || protocolErr.Code != "stream_finish_reason_changed" {
			t.Fatalf("repeat %s: want stream_finish_reason_changed, got %v", stopReason, pushErr)
		}
	}
}
