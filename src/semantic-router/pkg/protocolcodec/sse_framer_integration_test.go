package protocolcodec

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestSSEFramerLargeEventsAcrossProtocolMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	content := strings.Repeat("你好🌍", 8192)
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		body := strings.ReplaceAll(string(streamFixture(source)), "hello", content)
		input := goldenStreamInput{PublicModel: "public-model", ProviderModel: "source-model", ResponseID: "response_1"}
		baseline := translateGoldenStream(t, engine, source, target, input, []string{body})
		assertSSETranslatedText(t, engine, target, baseline, content)
		transcript := marshalGoldenStreamTranscript(t, baseline)
		for _, ending := range []string{"\n", "\r", "\r\n"} {
			t.Run(fmt.Sprintf("ending=%q", ending), func(t *testing.T) {
				wire := strings.ReplaceAll(body, "\n", ending)
				// No final blank line: the final event must be processed at EOF.
				wire = strings.TrimSuffix(wire, ending+ending)
				var chunks []string
				for offset := 0; offset < len(wire); offset += 257 {
					chunks = append(chunks, "", wire[offset:min(offset+257, len(wire))])
				}
				frames := translateGoldenStream(t, engine, source, target, input, chunks)
				if !jsonEquivalent(transcript, marshalGoldenStreamTranscript(t, frames)) {
					t.Fatal("large-event translation changed with line endings, fragmentation, or EOF")
				}
				assertSSETranslatedText(t, engine, target, frames, content)
			})
		}
	})
}

func assertSSETranslatedText(t *testing.T, engine *Engine, format llmprotocol.WireFormat, frames [][]byte, want string) {
	t.Helper()
	stream, err := engine.NewStream(format, format, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "public-model", ResponseID: "response_1",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(bytes.Join(frames, nil))
	if err != nil {
		t.Fatal(err)
	}
	_, final, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	completed := 0
	for _, event := range append(events, final...) {
		switch event.Type {
		case llmprotocol.EventOutputTextDelta:
			text.WriteString(event.Delta)
		case llmprotocol.EventResponseCompleted:
			completed++
		case llmprotocol.EventResponseFailed:
			t.Fatalf("translated stream failed: %v", event.Error)
		}
	}
	if completed != 1 || text.String() != want {
		t.Fatalf("translated stream completed %d times and returned %d text bytes; want one completion and %d exact bytes",
			completed, text.Len(), len(want))
	}
}

func TestSSEFramerChatReplayAcrossCRLFSplits(t *testing.T) {
	frame := "data: { \"choices\": [{\"index\":0,\"delta\":{\"content\":\"你好🌍\"}}] }\r\n\r\n"
	terminal := "data: [DONE]"
	body := []byte(frame + frame + terminal)
	for split := 0; split <= len(body); split++ {
		filter := NewChatPublicStreamFilter(len(body), true)
		var output []byte
		for _, chunk := range [][]byte{body[:split], nil, body[split:]} {
			part, err := filter.Push(chunk)
			if err != nil {
				t.Fatalf("split %d: %v", split, err)
			}
			output = append(output, part...)
		}
		if !bytes.Equal(output, []byte(frame+frame)) {
			t.Fatalf("split %d changed content or published the terminal early: %q", split, output)
		}
		final, err := filter.Finalize()
		if err != nil || !bytes.Equal(append(output, final...), body) {
			t.Fatalf("split %d changed replay bytes at EOF: %v", split, err)
		}
	}
}

func TestSSEFramerRejectsLargeFramesAcrossProtocolDecoders(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.Limits.SSEFrameBytes = 64
	for _, codec := range []StreamCodec{OpenAIChatCodec{}, OpenAIResponsesCodec{}, AnthropicMessagesCodec{}} {
		t.Run(fmt.Sprintf("%T", codec), func(t *testing.T) {
			decoder := codec.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, policy)
			for _, chunk := range []string{": keepalive\n\ndata: ", strings.Repeat("x", 64)} {
				_, _, err := decoder.Push([]byte(chunk))
				if len(chunk) == 64 {
					assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit")
				} else if err != nil {
					t.Fatal(err)
				}
			}
			_, _, err := decoder.Finalize(nil)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit")
		})
	}
}
