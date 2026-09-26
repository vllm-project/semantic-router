package protocolcodec

import (
	"bytes"
	"context"
	"errors"
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

func TestDynamoBoundaryMutationFailureAcrossChunksAndEOF(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		for _, mode := range []string{"split", "malformed-tail", "eof"} {
			t.Run(string(target)+"/"+mode, func(t *testing.T) {
				rejection := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "unexpected_dynamo_nvext_backend", "provider extension rejected", nil)
				stream, err := NewBuiltinEngine().NewStreamWithMutation(
					llmprotocol.OpenAIChatV1, target,
					llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
					func(event *llmprotocol.Event) error {
						if event.DynamoNVExt != nil {
							return rejection
						}
						return nil
					},
				)
				if err != nil {
					t.Fatal(err)
				}
				body := []byte(`data: {"id":"response_1","model":"provider-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"nvext":{"token_ids":[1]}}`)
				if mode != "eof" {
					body = append(body, []byte("\n\n")...)
				}
				if mode == "malformed-tail" {
					body = append(body, []byte("data: {malformed\n\n")...)
				}
				prefix, events, _, pushErr := stream.Push(body)
				if mode == "eof" {
					if pushErr != nil {
						t.Fatal(pushErr)
					}
				} else {
					if !errors.Is(pushErr, rejection) {
						t.Fatalf("want boundary rejection, got %v", pushErr)
					}
					later, _, _, err := stream.Push([]byte("data: [DONE]\n\n"))
					if !errors.Is(err, rejection) || len(later) != 0 {
						t.Fatalf("later body recovered from rejection: frames=%q err=%v", later, err)
					}
				}
				// Deliberately pass nil, like a fresh response-body buffer.
				final, finalEvents, _, err := stream.Finalize(nil)
				if err != nil && !errors.Is(err, rejection) {
					t.Fatalf("unexpected finalization error: %v", err)
				}
				if !errors.Is(stream.failure, rejection) {
					t.Fatal("finalization lost the first rejection")
				}
				assertNoCompletedEvent(t, append(events, finalEvents...), "rejected extension completed successfully")
				wire := append(bytes.Join(prefix, nil), bytes.Join(final, nil)...)
				assertNoSuccessfulStreamTerminal(t, target, wire)
				if bytes.Count(wire, []byte("provider extension rejected")) != 1 {
					t.Fatalf("want exactly one protocol error frame: %s", wire)
				}
				if bytes.Contains(wire, []byte(`"nvext"`)) || bytes.Contains(wire, []byte(`"token_ids"`)) {
					t.Fatalf("rejected metadata reached the encoder: %s", wire)
				}
				again, _, _, err := stream.Finalize(nil)
				if err != nil || len(again) != 0 {
					t.Fatalf("duplicate finalization: frames=%q err=%v", again, err)
				}
			})
		}
	}
}
