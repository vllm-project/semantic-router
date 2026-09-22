package protocolcodec

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDynamoStreamReturnsAcceptedPrefixWithDecodeError(t *testing.T) {
	for _, tc := range []struct{ name, frame string }{
		{"nvext", "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"prefix\"},\"finish_reason\":null}],\"nvext\":{\"token_ids\":[1]}}\n\n"},
		{"request_id", "event: request_id\n: \"req-prefix\"\n\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := newDynamoTestStream(t, llmprotocol.DefaultPolicy(), llmprotocol.OpenAIChatV1)
			frames, events, _, err := stream.Push([]byte(tc.frame + "data: {malformed\n\n"))
			if err == nil {
				t.Fatal("missing trailing decode error")
			}
			if !bytes.Contains(bytes.Join(frames, nil), []byte(tc.name)) {
				t.Fatalf("missing accepted prefix frames: %q", frames)
			}
			found := false
			for _, event := range events {
				if tc.name == "nvext" && event.DynamoNVExt != nil || tc.name == "request_id" && event.DynamoRequestID {
					found = true
				}
			}
			if !found {
				t.Fatalf("missing accepted Dynamo event: %+v", events)
			}
		})
	}
}
