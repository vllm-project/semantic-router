package protocolcodec

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestMidstreamProviderErrorTerminatesEveryProtocolPair(t *testing.T) {
	fixtures := map[llmprotocol.WireFormat][]byte{
		llmprotocol.OpenAIChatV1: []byte(
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"error\":{\"message\":\"mock provider stream failed\",\"type\":\"server_error\",\"param\":null,\"code\":\"provider_overloaded\"}}\n\n",
		),
		llmprotocol.OpenAIResponsesV1: []byte(
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
				"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n" +
				"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}\n\n" +
				"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"partial\"}\n\n" +
				"event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":4,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"failed\",\"output\":[],\"error\":{\"code\":\"provider_overloaded\",\"message\":\"mock provider stream failed\"}}}\n\n",
		),
		llmprotocol.AnthropicMessagesV1: []byte(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n" +
				"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"mock provider stream failed\"}}\n\n",
		),
	}

	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
					Context: context.Background(), PublicModel: "public-model",
				})
				if err != nil {
					t.Fatal(err)
				}
				frames, events, _, err := stream.Push(fixtures[source])
				if err != nil {
					t.Fatal(err)
				}
				assertMidstreamFailureEvents(t, events)
				assertMidstreamFailureWire(t, target, bytes.Join(frames, nil))

				finalFrames, finalEvents, _, err := stream.Finalize(nil)
				if err != nil {
					t.Fatal(err)
				}
				if len(finalFrames) != 0 || len(finalEvents) != 0 {
					t.Fatalf("terminal stream emitted duplicate final state: frames=%q events=%+v", finalFrames, finalEvents)
				}
			})
		}
	}
}

func assertMidstreamFailureEvents(t *testing.T, events []llmprotocol.Event) {
	t.Helper()
	deltaIndex, failureIndex, failures := -1, -1, 0
	for index, event := range events {
		switch event.Type {
		case llmprotocol.EventOutputTextDelta:
			if event.Delta == "partial" {
				deltaIndex = index
			}
		case llmprotocol.EventResponseFailed:
			failures++
			failureIndex = index
			if event.Error == nil || event.Error.Message != "mock provider stream failed" {
				t.Fatalf("failure lost provider error: %+v", event)
			}
		}
	}
	if deltaIndex < 0 || failureIndex <= deltaIndex || failures != 1 {
		t.Fatalf("midstream event order is invalid: %+v", events)
	}
}

func assertMidstreamFailureWire(t *testing.T, target llmprotocol.WireFormat, wire []byte) {
	t.Helper()
	if bytes.Count(wire, []byte("mock provider stream failed")) != 1 {
		t.Fatalf("provider failure was not emitted exactly once: %s", wire)
	}
	partialIndex := bytes.Index(wire, []byte("partial"))
	failureIndex := bytes.Index(wire, []byte("mock provider stream failed"))
	if partialIndex < 0 || failureIndex <= partialIndex {
		t.Fatalf("public stream lost midstream ordering: %s", wire)
	}
	assertMidstreamProtocolTerminal(t, target, wire)
}

func assertMidstreamProtocolTerminal(t *testing.T, target llmprotocol.WireFormat, wire []byte) {
	t.Helper()
	switch target {
	case llmprotocol.OpenAIChatV1:
		if !bytes.Contains(wire, []byte(`"error":`)) || bytes.Contains(wire, []byte("data: [DONE]")) {
			t.Fatalf("Chat stream terminal is invalid: %s", wire)
		}
	case llmprotocol.OpenAIResponsesV1:
		if !bytes.Contains(wire, []byte("event: error")) && !bytes.Contains(wire, []byte("event: response.failed")) {
			t.Fatalf("Responses failure terminal is missing: %s", wire)
		}
		if bytes.Contains(wire, []byte("event: response.completed")) {
			t.Fatalf("Responses failure ended as success: %s", wire)
		}
	case llmprotocol.AnthropicMessagesV1:
		if !bytes.Contains(wire, []byte("event: error")) || bytes.Contains(wire, []byte("event: message_stop")) {
			t.Fatalf("Messages stream terminal is invalid: %s", wire)
		}
	}
}
