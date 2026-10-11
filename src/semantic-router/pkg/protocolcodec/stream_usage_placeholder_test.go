package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatStreamAcceptsAllZeroUsageSnapshotBetweenCumulativeFrames(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first \"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n" +
			"data: [DONE]\n\n",
	)
	engine := NewBuiltinEngine()
	for _, target := range builtinFormats {
		t.Run(string(target), func(t *testing.T) {
			stream, err := engine.NewStream(
				llmprotocol.OpenAIChatV1,
				target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			_, events, _, err := stream.Push(payload)
			if err != nil {
				t.Fatal(err)
			}
			_, finalEvents, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			text, terminal, usage := summarizeStreamEvents(t, append(events, finalEvents...))
			if text != "first second" || terminal != 1 || usage == nil ||
				usage.Total.Value == nil || *usage.Total.Value != 5 ||
				usage.InputTotal.Value == nil || *usage.InputTotal.Value != 3 ||
				usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != 2 {
				t.Fatalf("translated stream text=%q terminals=%d usage=%+v", text, terminal, usage)
			}
		})
	}
}

func TestChatStreamZeroUsageSnapshotKeepsPositiveDecreaseStrict(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first \"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n" +
			"data: [DONE]\n\n",
	)
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := stream.Push(payload); err == nil {
		t.Fatal("positive usage decrease between cumulative snapshots was accepted")
	} else {
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "usage_decreased")
	}
}

func TestChatStreamLeadingZeroUsageSnapshotStillMerges(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first \"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n" +
			"data: [DONE]\n\n",
	)
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	_, finalEvents, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	_, terminal, usage := summarizeStreamEvents(t, append(events, finalEvents...))
	if terminal != 1 || usage == nil || usage.Total.Value == nil || *usage.Total.Value != 5 {
		t.Fatalf("leading zero snapshot broke the stream merge: terminals=%d usage=%+v", terminal, usage)
	}
}

func summarizeStreamEvents(t *testing.T, events []llmprotocol.Event) (string, int, *llmprotocol.Usage) {
	t.Helper()
	text, terminal := "", 0
	var usage *llmprotocol.Usage
	for _, event := range events {
		switch event.Type {
		case llmprotocol.EventOutputTextDelta:
			text += event.Delta
		case llmprotocol.EventResponseCompleted:
			terminal++
			usage = event.Usage
		}
	}
	return text, terminal, usage
}
