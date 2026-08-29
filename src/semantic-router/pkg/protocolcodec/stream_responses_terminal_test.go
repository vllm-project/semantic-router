package protocolcodec

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesToolStreamInfersToolCallTerminalReason(t *testing.T) {
	engine := NewBuiltinEngine()
	targets := []struct {
		name     string
		format   llmprotocol.WireFormat
		expected []byte
	}{
		{name: "chat", format: llmprotocol.OpenAIChatV1, expected: []byte(`"finish_reason":"tool_calls"`)},
		{name: "anthropic", format: llmprotocol.AnthropicMessagesV1, expected: []byte(`"stop_reason":"tool_use"`)},
	}

	for _, target := range targets {
		t.Run(target.name, func(t *testing.T) {
			stream, err := engine.NewStream(
				llmprotocol.OpenAIResponsesV1,
				target.format,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			frames, events, _, err := stream.Push(toolStreamFixture(llmprotocol.OpenAIResponsesV1))
			if err != nil {
				t.Fatal(err)
			}
			finalFrames, finalEvents, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			frames = append(frames, finalFrames...)
			events = append(events, finalEvents...)
			if !bytes.Contains(bytes.Join(frames, nil), target.expected) {
				t.Fatalf("translated terminal did not preserve tool-call semantics: %s", bytes.Join(frames, nil))
			}
			foundTerminal := false
			for _, event := range events {
				if event.Type == llmprotocol.EventResponseCompleted {
					foundTerminal = true
					if event.StopReason != llmprotocol.StopToolCall {
						t.Fatalf("terminal stop reason = %q, want %q", event.StopReason, llmprotocol.StopToolCall)
					}
				}
			}
			if !foundTerminal {
				t.Fatal("translated stream did not emit a terminal event")
			}
		})
	}
}

func TestResponsesToolResponseInfersToolCallStopReason(t *testing.T) {
	body := []byte(`{
		"id":"response_tool","object":"response","model":"provider-model","status":"completed",
		"output":[{"type":"function_call","id":"item_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}]
	}`)
	response, _, _, err := (OpenAIResponsesCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if response.StopReason != llmprotocol.StopToolCall {
		t.Fatalf("stop reason = %q, want %q", response.StopReason, llmprotocol.StopToolCall)
	}
}
