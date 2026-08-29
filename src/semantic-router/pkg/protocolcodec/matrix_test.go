package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var builtinFormats = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1,
	llmprotocol.OpenAIResponsesV1,
	llmprotocol.AnthropicMessagesV1,
}

func TestBufferedRequestMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				translated, err := engine.TranslateRequest(source, target, requestFixture(source), nil)
				if err != nil {
					t.Fatalf("TranslateRequest() error = %v", err)
				}
				decoded, _, _, err := engine.DecodeRequest(target, translated.Body)
				if err != nil {
					t.Fatalf("target DecodeRequest() error = %v, body=%s", err, translated.Body)
				}
				if decoded.Model != "source-model" || len(decoded.Messages) != 1 ||
					decoded.Messages[0].Content[0].Text != "hello" || len(decoded.Tools) != 1 {
					t.Fatalf("translated request = %+v", decoded)
				}
			})
		}
	}
}

func TestToolChoiceModesSurviveEveryProtocolPair(t *testing.T) {
	engine := NewBuiltinEngine()
	modes := []llmprotocol.ToolChoice{
		{Mode: llmprotocol.ToolChoiceAuto},
		{Mode: llmprotocol.ToolChoiceNone},
		{Mode: llmprotocol.ToolChoiceRequired},
		{Mode: llmprotocol.ToolChoiceNamed, Name: "lookup"},
	}
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		for _, want := range modes {
			t.Run(string(want.Mode), func(t *testing.T) {
				assertToolChoiceTranslation(t, engine, source, target, want)
			})
		}
	})
}

func assertToolChoiceTranslation(
	t *testing.T,
	engine *Engine,
	source, target llmprotocol.WireFormat,
	want llmprotocol.ToolChoice,
) {
	t.Helper()
	translated, err := engine.TranslateRequest(source, target, toolChoiceFixture(source, want.Mode), nil)
	if err != nil {
		t.Fatal(err)
	}
	request, _, _, err := engine.DecodeRequest(target, translated.Body)
	if err != nil {
		t.Fatalf("decode translated request: %v\n%s", err, translated.Body)
	}
	if request.ToolChoice != want {
		t.Fatalf("tool choice = %+v, want %+v\n%s", request.ToolChoice, want, translated.Body)
	}
}

func TestOpenAIRequestsWithoutOutputLimitRemainValidForAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   []byte
	}{
		{
			name:   "Chat Completions",
			format: llmprotocol.OpenAIChatV1,
			body:   []byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}]}`),
		},
		{
			name:   "Responses",
			format: llmprotocol.OpenAIResponsesV1,
			body:   []byte(`{"model":"source-model","input":"hello"}`),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translated, err := engine.TranslateRequest(test.format, llmprotocol.AnthropicMessagesV1, test.body, nil)
			if err != nil {
				t.Fatal(err)
			}
			var wire anthropicRequestWire
			if err := json.Unmarshal(translated.Body, &wire); err != nil {
				t.Fatal(err)
			}
			if wire.MaxTokens == nil || *wire.MaxTokens != defaultAnthropicMaxOutputTokens {
				t.Fatalf("max_tokens = %v, want generated default %d", wire.MaxTokens, defaultAnthropicMaxOutputTokens)
			}
			if len(translated.Diagnostics) != 1 || translated.Diagnostics[0].Action != llmprotocol.DiagnosticGenerated {
				t.Fatalf("diagnostics = %+v, want one generated max_tokens diagnostic", translated.Diagnostics)
			}
		})
	}
}

func TestAnthropicTemperatureRangeCannotBeCrossedSilently(t *testing.T) {
	engine := NewBuiltinEngine()
	_, _, _, err := engine.DecodeRequest(
		llmprotocol.AnthropicMessagesV1,
		[]byte(`{"model":"source-model","max_tokens":64,"temperature":1.5,"messages":[{"role":"user","content":"hello"}]}`),
	)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_anthropic_temperature")

	_, err = engine.TranslateRequest(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		[]byte(`{"model":"source-model","temperature":1.5,"messages":[{"role":"user","content":"hello"}]}`),
		nil,
	)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_anthropic_temperature")
}

func TestChatStopSequenceCountCannotBeCrossedSilently(t *testing.T) {
	engine := NewBuiltinEngine()
	_, _, _, err := engine.DecodeRequest(
		llmprotocol.OpenAIChatV1,
		[]byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}],"stop":["1","2","3","4","5"]}`),
	)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "chat_stop_sequence_limit")

	_, err = engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		[]byte(`{"model":"source-model","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"stop_sequences":["1","2","3","4","5"]}`),
		nil,
	)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_chat_stop_sequence_limit")
}

func TestBufferedToolLifecycleMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				translated, err := engine.TranslateRequest(source, target, toolLifecycleFixture(source), nil)
				if err != nil {
					t.Fatalf("TranslateRequest() error = %v", err)
				}
				decoded, _, _, err := engine.DecodeRequest(target, translated.Body)
				if err != nil {
					t.Fatalf("target DecodeRequest() error = %v, body=%s", err, translated.Body)
				}
				assertToolLifecycle(t, decoded)
			})
		}
	}
}

func TestBufferedOrderedImageMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertBufferedOrderedImagePair(t, engine, source, target)
	})
}

func assertBufferedOrderedImagePair(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	translated, err := engine.TranslateRequest(source, target, orderedImageFixture(source), nil)
	if err != nil {
		t.Fatalf("TranslateRequest() error = %v", err)
	}
	decoded, _, _, err := engine.DecodeRequest(target, translated.Body)
	if err != nil {
		t.Fatalf("target DecodeRequest() error = %v, body=%s", err, translated.Body)
	}
	if len(decoded.Messages) != 1 || len(decoded.Messages[0].Content) != 3 {
		t.Fatalf("ordered multimodal request = %+v", decoded.Messages)
	}
	content := decoded.Messages[0].Content
	if content[0].Kind != llmprotocol.ContentText || content[0].Text != "before" ||
		content[1].Kind != llmprotocol.ContentImage || content[1].URL != "https://example.com/image.png" ||
		content[2].Kind != llmprotocol.ContentText || content[2].Text != "after" {
		t.Fatalf("multimodal content order changed: %+v", content)
	}
}

func TestBufferedFileInputMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertBufferedFilePair(t, engine, source, target)
	})
}

func assertBufferedFilePair(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	translated, err := engine.TranslateRequest(source, target, fileInputFixture(source), nil)
	if err != nil {
		t.Fatalf("TranslateRequest() error = %v", err)
	}
	decoded, _, _, err := engine.DecodeRequest(target, translated.Body)
	if err != nil {
		t.Fatalf("target DecodeRequest() error = %v, body=%s", err, translated.Body)
	}
	if len(decoded.Messages) != 1 || len(decoded.Messages[0].Content) != 1 {
		t.Fatalf("file request = %+v", decoded.Messages)
	}
	content := decoded.Messages[0].Content[0]
	if content.Kind != llmprotocol.ContentFile || content.FileID != "file_1" {
		t.Fatalf("file semantics changed: %+v", content)
	}
}

func TestBufferedResponseMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertBufferedResponsePair(t, engine, source, target)
	})
}

func TestResponsesEncoderSynthesizesOutputText(t *testing.T) {
	response := llmprotocol.Response{
		Generation: 1,
		ID:         "resp_1",
		Model:      "public-model",
		Output: []llmprotocol.OutputItem{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "first"},
				{Kind: llmprotocol.ContentReasoning, Text: "hidden"},
			}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: " second"},
			}},
		},
	}
	envelope := llmprotocol.Envelope{
		Format: llmprotocol.OpenAIResponsesV1, Generation: 1,
		Response:       []byte(`{"id":"stale-replay"}`),
		ResponseRender: llmprotocol.ResponseRenderContext{PreviousResponseID: "resp_previous"},
	}
	body, _, err := (OpenAIResponsesCodec{}).EncodeResponse(response, envelope, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	var wire responsesResponseWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	var outputText string
	if err := json.Unmarshal(wire.OutputText, &outputText); err != nil {
		t.Fatalf("output_text is not a JSON string: %v", err)
	}
	if wire.PreviousResponseID != "resp_previous" {
		t.Fatalf("previous_response_id = %q, want request lineage", wire.PreviousResponseID)
	}
	if outputText != "first second" {
		t.Fatalf("output_text = %q, want %q", outputText, "first second")
	}
}

func assertBufferedResponsePair(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	translated, err := engine.TranslateResponse(source, target, responseFixture(source), nil)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	decoded, err := decodeTranslatedResponse(engine, target, translated.Body)
	if err != nil {
		t.Fatalf("target response decode error = %v, body=%s", err, translated.Body)
	}
	if decoded.ID != "response_1" || len(decoded.Output) == 0 || decoded.Output[0].Content[0].Text != "hello" {
		t.Fatalf("translated response = %+v", decoded)
	}
	if decoded.Usage.State != llmprotocol.UsageAvailable || decoded.Usage.InputTotal.Value == nil || *decoded.Usage.InputTotal.Value != 2 {
		t.Fatalf("translated usage = %+v", decoded.Usage)
	}
}

func forEachBuiltinFormatPair(
	t *testing.T,
	assertPair func(*testing.T, llmprotocol.WireFormat, llmprotocol.WireFormat),
) {
	t.Helper()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				assertPair(t, source, target)
			})
		}
	}
}

func TestAuthoritativeUsageEvidenceSurvivesClientRepresentationLoss(t *testing.T) {
	engine := NewBuiltinEngine()
	chat := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":2}}}`)
	for _, target := range builtinFormats {
		t.Run("reasoning_and_cache_read_to_"+string(target), func(t *testing.T) {
			assertReasoningAndCacheReadEvidence(t, engine, target, chat)
		})
	}

	messages := []byte(`{"id":"response_1","type":"message","role":"assistant","model":"source-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}`)
	for _, target := range builtinFormats {
		t.Run("cache_write_to_"+string(target), func(t *testing.T) {
			assertCacheWriteEvidence(t, engine, target, messages)
		})
	}
}

func assertReasoningAndCacheReadEvidence(t *testing.T, engine *Engine, target llmprotocol.WireFormat, body []byte) {
	t.Helper()
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	usage := translated.Response.Usage
	if !hasAuthoritativeTokenCount(usage.InputCacheRead, 4) || !hasAuthoritativeTokenCount(usage.OutputReasoning, 2) {
		t.Fatalf("settlement evidence changed: %+v", usage)
	}
	decoded, _, _, err := engine.DecodeResponse(target, translated.Body)
	if err != nil {
		t.Fatalf("decode translated usage: %v\n%s", err, translated.Body)
	}
	if !hasTokenCount(decoded.Usage.InputCacheRead, 4) || !hasTokenCount(decoded.Usage.OutputReasoning, 2) {
		t.Fatalf("target accounting evidence changed: %+v", decoded.Usage)
	}
}

func hasTokenCount(count llmprotocol.TokenCount, want int64) bool {
	return count.Value != nil && *count.Value == want
}

func hasAuthoritativeTokenCount(count llmprotocol.TokenCount, want int64) bool {
	return hasTokenCount(count, want) && count.Provenance == llmprotocol.UsageAuthoritative
}

func assertCacheWriteEvidence(t *testing.T, engine *Engine, target llmprotocol.WireFormat, body []byte) {
	t.Helper()
	translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	usage := translated.Response.Usage
	if usage.InputCacheWrite.Value == nil || *usage.InputCacheWrite.Value != 3 ||
		usage.InputCacheWrite.Provenance != llmprotocol.UsageAuthoritative {
		t.Fatalf("cache-write settlement evidence changed: %+v", usage)
	}
	decoded, _, _, err := engine.DecodeResponse(target, translated.Body)
	if err != nil {
		t.Fatalf("decode translated usage: %v\n%s", err, translated.Body)
	}
	if decoded.Usage.InputCacheWrite.Value == nil || *decoded.Usage.InputCacheWrite.Value != 3 {
		t.Fatalf("target cache-write evidence changed: %+v", decoded.Usage)
	}
}

func TestRefusalAndReasoningFidelityIsExplicitAcrossFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	reasoning := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"check","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}`)
	for _, target := range builtinFormats {
		assertReasoningFidelity(t, engine, target, reasoning)
	}

	refusal := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","refusal":"cannot comply"},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	for _, target := range builtinFormats {
		assertRefusalFidelity(t, engine, target, refusal)
	}
}

func assertReasoningFidelity(t *testing.T, engine *Engine, target llmprotocol.WireFormat, body []byte) {
	t.Helper()
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
	if err != nil {
		t.Fatalf("reasoning to %s: %v", target, err)
	}
	for _, item := range translated.Response.Output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentReasoning && content.Text == "check" {
				return
			}
		}
	}
	t.Fatalf("reasoning semantic lost for %s: %+v", target, translated.Response.Output)
}

func assertRefusalFidelity(t *testing.T, engine *Engine, target llmprotocol.WireFormat, body []byte) {
	t.Helper()
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
	if target == llmprotocol.AnthropicMessagesV1 {
		if err == nil {
			t.Fatal("refusal was silently weakened to ordinary Messages text")
		}
		return
	}
	if err != nil || len(translated.Response.Output) != 1 ||
		translated.Response.Output[0].Content[0].Kind != llmprotocol.ContentRefusal {
		t.Fatalf("refusal to %s = %+v, %v", target, translated.Response.Output, err)
	}
}

func TestStreamingMatrixAcceptsArbitraryTransportChunks(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertStreamingMatrixPair(t, engine, source, target)
	})
}

func assertStreamingMatrixPair(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	stream := mustNewMatrixStream(t, engine, source, target)
	events, encoded := pushChunkedFixture(t, stream, streamFixture(source), 17)
	assertTerminalStream(t, events, encoded)
	assertTargetStreamDecodes(t, engine, target, encoded.Bytes())
}

func TestStreamingToolCallMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	forEachBuiltinFormatPair(t, func(t *testing.T, source, target llmprotocol.WireFormat) {
		assertStreamingToolPair(t, engine, source, target)
	})
}

func assertStreamingToolPair(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) {
	t.Helper()
	stream := mustNewMatrixStream(t, engine, source, target)
	events, encoded := pushChunkedFixture(t, stream, toolStreamFixture(source), 13)
	assertStreamToolCall(t, events)
	assertTargetToolStreamDecodes(t, engine, target, encoded.Bytes())
}

func mustNewMatrixStream(t *testing.T, engine *Engine, source, target llmprotocol.WireFormat) *StreamEngine {
	t.Helper()
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
		Options: llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func pushChunkedFixture(t *testing.T, stream *StreamEngine, payload []byte, modulus int) ([]llmprotocol.Event, bytes.Buffer) {
	t.Helper()
	var events []llmprotocol.Event
	var encoded bytes.Buffer
	for offset := 0; offset < len(payload); {
		end := offset + 1 + offset%modulus
		if end > len(payload) {
			end = len(payload)
		}
		frames, decoded, _, err := stream.Push(payload[offset:end])
		if err != nil {
			t.Fatalf("Push(%d:%d): %v", offset, end, err)
		}
		writeFrames(&encoded, frames)
		events = append(events, decoded...)
		offset = end
	}
	frames, decoded, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("Finalize() error = %v", err)
	}
	writeFrames(&encoded, frames)
	return append(events, decoded...), encoded
}

func writeFrames(output *bytes.Buffer, frames [][]byte) {
	for _, frame := range frames {
		output.Write(frame)
	}
}

func assertTerminalStream(t *testing.T, events []llmprotocol.Event, encoded bytes.Buffer) {
	t.Helper()
	if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted || events[len(events)-1].Usage == nil {
		t.Fatalf("terminal events = %+v", events)
	}
	if encoded.Len() == 0 {
		t.Fatal("target stream is empty")
	}
}

func assertTargetStreamDecodes(t *testing.T, engine *Engine, target llmprotocol.WireFormat, encoded []byte) {
	t.Helper()
	verify := mustNewMatrixStream(t, engine, target, target)
	_, targetEvents, _, err := verify.Push(encoded)
	if err != nil {
		t.Fatalf("target wire did not decode: %v\n%s", err, encoded)
	}
	_, finalized, _, err := verify.Finalize(nil)
	if err != nil {
		t.Fatalf("target Finalize() error = %v", err)
	}
	targetEvents = append(targetEvents, finalized...)
	if len(targetEvents) == 0 || targetEvents[len(targetEvents)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("target terminal events = %+v", targetEvents)
	}
}

func assertTargetToolStreamDecodes(t *testing.T, engine *Engine, target llmprotocol.WireFormat, encoded []byte) {
	t.Helper()
	verify := mustNewMatrixStream(t, engine, target, target)
	_, events, _, err := verify.Push(encoded)
	if err != nil {
		t.Fatalf("target stream did not decode: %v\n%s", err, encoded)
	}
	_, finalEvents, _, err := verify.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	assertStreamToolCall(t, append(events, finalEvents...))
}
