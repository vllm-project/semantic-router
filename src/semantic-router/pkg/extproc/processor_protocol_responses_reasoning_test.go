package extproc

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestExtProcResponsesReasoningParts(t *testing.T) {
	payload, err := os.ReadFile("../protocolcodec/testdata/responses-reasoning-parts.sse")
	if err != nil {
		t.Fatal(err)
	}
	for _, client := range extProcMatrixFormats {
		t.Run(string(client), func(t *testing.T) {
			const chunkSize = 17
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				SourceFormat: client, TargetFormat: llmprotocol.OpenAIResponsesV1,
				RequestModel: "provider-model", TraceContext: t.Context(),
			}
			var clientWire bytes.Buffer
			for offset := 0; offset < len(payload); offset += chunkSize {
				end := min(offset+chunkSize, len(payload))
				result := router.handleSemanticStreamingResponseBody(payload[offset:end], end == len(payload), ctx)
				if ctx.StreamingAborted || result.GetResponseBody() == nil {
					t.Fatalf("stream aborted at %d: %+v", offset, ctx.ProtocolDiagnostics)
				}
				mutation := result.GetResponseBody().GetResponse().GetBodyMutation()
				if mutation == nil {
					clientWire.Write(payload[offset:end])
				} else {
					clientWire.Write(mutation.GetBody())
				}
			}
			if !ctx.StreamingComplete || ctx.SemanticResponse == nil ||
				ctx.SemanticResponse.Usage.Total.Value == nil || *ctx.SemanticResponse.Usage.Total.Value != 5 {
				t.Fatalf("stream did not complete with usage: %+v", ctx.SemanticResponse)
			}
			if client == llmprotocol.OpenAIResponsesV1 && !bytes.Equal(clientWire.Bytes(), payload) {
				t.Fatal("same-format replay changed provider bytes")
			}
			assertResponsesReasoningClientContent(t, client, clientWire.Bytes())
		})
	}
}

func assertResponsesReasoningClientContent(t *testing.T, format llmprotocol.WireFormat, body []byte) {
	t.Helper()
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(format, format, llmprotocol.StreamContext{Context: t.Context()})
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(body)
	if err != nil {
		t.Fatal(err)
	}
	_, terminal, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	events = append(events, terminal...)
	var reasoning, answer strings.Builder
	scopes := map[llmprotocol.ReasoningScope]string{}
	for _, event := range events {
		switch event.Type {
		case llmprotocol.EventReasoningDelta:
			reasoning.WriteString(event.Delta)
			scopes[event.Content.Reasoning] += event.Delta
		case llmprotocol.EventOutputTextDelta:
			answer.WriteString(event.Delta)
		}
	}
	if reasoning.String() != "summarythink" || answer.String() != "done" ||
		events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("translated content or completion changed: reasoning=%q answer=%q", reasoning.String(), answer.String())
	}
	if format == llmprotocol.OpenAIResponsesV1 &&
		(scopes[llmprotocol.ReasoningScopeSummary] != "summary" || scopes[llmprotocol.ReasoningScopeText] != "think") {
		t.Fatalf("Responses replay lost reasoning scopes: %+v", scopes)
	}
}
