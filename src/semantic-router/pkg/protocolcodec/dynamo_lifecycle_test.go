package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func dynamoLateLifecycleFrames(kind string) [][]byte {
	status := "in_progress"
	if kind == "response.queued" {
		status = "queued"
	}
	base := bytes.Split(bytes.TrimSpace(streamFixture(llmprotocol.OpenAIResponsesV1)), []byte("\n\n"))
	chunks := [][]byte{append(base[0], []byte("\n\n")...)}
	chunks = append(chunks, []byte(fmt.Sprintf("event: %s\ndata: {\"type\":%q,\"sequence_number\":1,\"response\":{\"id\":\"response_1\",\"model\":\"source-model\",\"status\":%q,\"output\":[],\"nvext\":{\"worker_id\":{\"prefill_worker_id\":11}}}}\n\n", kind, kind, status)))
	for index, frame := range base[1:] {
		frame = bytes.Replace(frame, []byte(fmt.Sprintf(`"sequence_number":%d`, index+1)), []byte(fmt.Sprintf(`"sequence_number":%d`, index+2)), 1)
		chunks = append(chunks, append(frame, []byte("\n\n")...))
	}
	return chunks
}

func TestDynamoLateResponsesLifecycleMetadata(t *testing.T) {
	for _, kind := range []string{"response.in_progress", "response.queued"} {
		for _, coalesced := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/coalesced=%t", kind, coalesced), func(t *testing.T) {
				stream := newDynamoResponsesTestStream(t, llmprotocol.DefaultPolicy(), llmprotocol.OpenAIResponsesV1)
				chunks := dynamoLateLifecycleFrames(kind)
				if coalesced {
					chunks = [][]byte{bytes.Join(chunks, nil)}
				}
				var frames [][]byte
				starts, metadata := 0, 0
				for _, chunk := range chunks {
					out, events, _, err := stream.Push(chunk)
					if err != nil {
						t.Fatal(err)
					}
					frames = append(frames, out...)
					for _, event := range events {
						if event.Type == llmprotocol.EventResponseStarted {
							starts++
						}
						if event.DynamoNVExt != nil {
							metadata++
							if event.Type != llmprotocol.EventProviderOpaque || event.DynamoResponsesLifecycle != kind {
								t.Fatalf("unexpected metadata event: %+v", event)
							}
						}
					}
				}
				frames = append(frames, finalizeDynamoStream(t, stream)...)
				if starts != 1 || metadata != 1 {
					t.Fatalf("starts=%d metadata=%d, want one each", starts, metadata)
				}
				found := 0
				for index, frame := range frames {
					parsed, parseErr := parseSSEFrame(frame, llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
					if parseErr != nil {
						t.Fatal(parseErr)
					}
					var wire responsesEventWire
					if err := json.Unmarshal(parsed.Data, &wire); err != nil {
						t.Fatal(err)
					}
					if wire.Sequence != uint64(index) {
						t.Fatalf("sequence=%d, want %d", wire.Sequence, index)
					}
					if wire.Response != nil && len(wire.Response.NVExt) != 0 {
						found++
						if wire.Type != kind || wire.Response.Model != "public-model" || wire.Response.ID != "response_1" {
							t.Fatalf("incorrect lifecycle identity: %+v", wire)
						}
						assertNestedJSONField(t, wire.Response.NVExt, "worker_id", "prefill_worker_id", float64(11))
					}
				}
				if found != 1 {
					t.Fatalf("encoded extensions=%d, want 1", found)
				}
			})
		}
	}
}

func TestDynamoLateLifecycleReachesBoundaryValidation(t *testing.T) {
	rejection := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "unexpected_dynamo_nvext_backend", "rejected provider metadata", nil)
	stream, err := NewBuiltinEngine().NewStreamWithMutation(
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1,
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
	chunks := dynamoLateLifecycleFrames("response.in_progress")
	if _, _, _, startErr := stream.Push(chunks[0]); startErr != nil {
		t.Fatal(startErr)
	}
	frames, _, _, err := stream.Push(chunks[1])
	if err != rejection || len(frames) != 0 {
		t.Fatalf("metadata bypassed boundary: frames=%q err=%v", frames, err)
	}
	final, _, _, _ := stream.Finalize(nil)
	wire := bytes.Join(final, nil)
	assertNoSuccessfulStreamTerminal(t, llmprotocol.OpenAIResponsesV1, wire)
	if !bytes.Contains(wire, []byte(rejection.Code)) || bytes.Contains(wire, []byte(`"nvext"`)) {
		t.Fatalf("unexpected failure output: %s", wire)
	}
}

func TestDynamoLateLifecycleKeepsTranslationAndSizeGuards(t *testing.T) {
	for _, tc := range []struct {
		name   string
		target llmprotocol.WireFormat
		limit  int
		code   string
	}{
		{"chat", llmprotocol.OpenAIChatV1, 0, "unsupported_dynamo_nvext_translation"},
		{"anthropic", llmprotocol.AnthropicMessagesV1, 0, "unsupported_dynamo_nvext_translation"},
		{"size", llmprotocol.OpenAIResponsesV1, 1, "dynamo_nvext_stream_size_limit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			if tc.limit > 0 {
				policy.Limits.DynamoNVExtStreamBytes = tc.limit
			}
			stream := newDynamoResponsesTestStream(t, policy, tc.target)
			chunks := dynamoLateLifecycleFrames("response.in_progress")
			if _, _, _, err := stream.Push(chunks[0]); err != nil {
				t.Fatal(err)
			}
			_, _, _, err := stream.Push(chunks[1])
			assertErrorCodeContains(t, err, tc.code)
		})
	}
}
