package extproc

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

var extProcMatrixFormats = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1,
	llmprotocol.OpenAIResponsesV1,
	llmprotocol.AnthropicMessagesV1,
}

// This test owns the ExtProc orientation contract: an upstream response is
// decoded with the backend format and encoded with the original client format.
// The codec package has its own exhaustive semantic tests; this matrix makes
// sure the data-plane seam never reverses those two axes.
func TestExtProcBufferedResponseProtocolMatrix(t *testing.T) {
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	for _, clientFormat := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			t.Run(string(clientFormat)+"_client_"+string(backendFormat)+"_backend", func(t *testing.T) {
				ctx := &RequestContext{SourceFormat: clientFormat, TargetFormat: backendFormat}
				semantic, err := router.decodeClientResponse(extProcResponseFixture(backendFormat), ctx)
				if err != nil {
					t.Fatalf("decodeClientResponse(): %v", err)
				}
				body, err := router.encodeClientResponse(*semantic, ctx)
				if err != nil {
					t.Fatalf("encodeClientResponse(): %v", err)
				}
				translated, err := engine.TranslateResponse(clientFormat, clientFormat, body, nil)
				if err != nil {
					t.Fatalf("client response is not valid %s: %v\n%s", clientFormat, err, body)
				}
				assertExtProcMatrixResponse(t, translated.Response, "source-model")
			})
		}
	}
}

// Streaming translation is stateful, so it needs an independent 3x3 matrix at
// the ExtProc seam rather than relying on buffered coverage or codec-only tests.
func TestExtProcStreamingResponseProtocolMatrix(t *testing.T) {
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	forEachExtProcMatrixPair(t, func(t *testing.T, clientFormat, backendFormat llmprotocol.WireFormat) {
		assertExtProcStreamingPair(t, router, engine, clientFormat, backendFormat)
	})
}

func forEachExtProcMatrixPair(
	t *testing.T,
	assertPair func(*testing.T, llmprotocol.WireFormat, llmprotocol.WireFormat),
) {
	t.Helper()
	for _, clientFormat := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			t.Run(string(clientFormat)+"_client_"+string(backendFormat)+"_backend", func(t *testing.T) {
				assertPair(t, clientFormat, backendFormat)
			})
		}
	}
}

func assertExtProcStreamingPair(
	t *testing.T,
	router *OpenAIRouter,
	engine *protocolcodec.Engine,
	clientFormat,
	backendFormat llmprotocol.WireFormat,
) {
	t.Helper()
	ctx := &RequestContext{
		SourceFormat: clientFormat, TargetFormat: backendFormat,
		RequestModel: "public-model", TraceContext: context.Background(),
	}
	if err := router.ensureSemanticResponseStream(ctx); err != nil {
		t.Fatalf("ensureSemanticResponseStream(): %v", err)
	}
	clientWire := pushExtProcStreamFixture(t, ctx, extProcStreamFixture(backendFormat))
	semantic, err := ctx.SemanticStreamState.response()
	if err != nil {
		t.Fatalf("reconstruct response: %v", err)
	}
	assertExtProcMatrixResponse(t, *semantic, "public-model")
	assertClientStreamDecodes(t, engine, clientFormat, clientWire.Bytes())
}

func pushExtProcStreamFixture(t *testing.T, ctx *RequestContext, payload []byte) bytes.Buffer {
	t.Helper()
	var clientWire bytes.Buffer
	for offset := 0; offset < len(payload); {
		end := offset + 1 + offset%19
		if end > len(payload) {
			end = len(payload)
		}
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Push(payload[offset:end])
		if err != nil {
			t.Fatalf("Push(%d:%d): %v", offset, end, err)
		}
		observeExtProcStream(ctx, frames, events, diagnostics, &clientWire)
		offset = end
	}
	frames, events, diagnostics, err := ctx.ProtocolResponseStream.Finalize(nil)
	if err != nil {
		t.Fatalf("Finalize(): %v", err)
	}
	observeExtProcStream(ctx, frames, events, diagnostics, &clientWire)
	return clientWire
}

func observeExtProcStream(
	ctx *RequestContext,
	frames [][]byte,
	events []llmprotocol.Event,
	diagnostics llmprotocol.Diagnostics,
	clientWire *bytes.Buffer,
) {
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	ctx.SemanticStreamState.observe(events)
	for _, frame := range frames {
		clientWire.Write(frame)
	}
}

func assertClientStreamDecodes(
	t *testing.T,
	engine *protocolcodec.Engine,
	format llmprotocol.WireFormat,
	body []byte,
) {
	t.Helper()
	stream, err := engine.NewStream(format, format, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(body)
	if err != nil {
		t.Fatalf("translated stream is not valid %s: %v\n%s", format, err, body)
	}
	_, terminal, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("translated stream finalize: %v", err)
	}
	events = append(events, terminal...)
	if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("translated stream has no terminal event: %+v", events)
	}
}

func assertExtProcMatrixResponse(t *testing.T, response llmprotocol.Response, expectedModel string) {
	t.Helper()
	if response.ID != "response_1" || response.Model != expectedModel ||
		len(response.Output) != 1 || len(response.Output[0].Content) != 1 ||
		response.Output[0].Content[0].Kind != llmprotocol.ContentText ||
		response.Output[0].Content[0].Text != "hello" {
		t.Fatalf("response semantics changed: %+v", response)
	}
	if response.Usage.State != llmprotocol.UsageAvailable ||
		response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 2 ||
		response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 1 {
		t.Fatalf("authoritative usage changed: %+v", response.Usage)
	}
}

func extProcResponseFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"id":"output_1","role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"id":"response_1","model":"source-model","status":"completed","output":[{"type":"message","id":"output_1","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"id":"response_1","type":"message","role":"assistant","model":"source-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
	default:
		panic(fmt.Sprintf("unsupported response fixture format %q", format))
	}
}

func extProcStreamFixture(format llmprotocol.WireFormat) []byte {
	join := func(events ...string) []byte { return []byte(strings.Join(events, "")) }
	switch format {
	case llmprotocol.OpenAIChatV1:
		return join(
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n",
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
			"data: [DONE]\n\n",
		)
	case llmprotocol.OpenAIResponsesV1:
		return join(
			"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"response_1\",\"model\":\"source-model\",\"status\":\"in_progress\"}}\n\n",
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\"}}\n\n",
			"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"output_1\",\"delta\":\"hello\"}\n\n",
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item_id\":\"output_1\"}\n\n",
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"response_1\",\"model\":\"source-model\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
		)
	case llmprotocol.AnthropicMessagesV1:
		return join(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"source-model\",\"content\":[],\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"type\":\"message_delta\",\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		)
	default:
		panic(fmt.Sprintf("unsupported stream fixture format %q", format))
	}
}
