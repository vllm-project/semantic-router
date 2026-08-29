package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
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
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			for _, want := range modes {
				name := fmt.Sprintf("%s_to_%s_%s", source, target, want.Mode)
				t.Run(name, func(t *testing.T) {
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
				})
			}
		}
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
	if usage.InputCacheRead.Value == nil || *usage.InputCacheRead.Value != 4 ||
		usage.InputCacheRead.Provenance != llmprotocol.UsageAuthoritative ||
		usage.OutputReasoning.Value == nil || *usage.OutputReasoning.Value != 2 ||
		usage.OutputReasoning.Provenance != llmprotocol.UsageAuthoritative {
		t.Fatalf("settlement evidence changed: %+v", usage)
	}
	if target == llmprotocol.AnthropicMessagesV1 && len(translated.Diagnostics) == 0 {
		t.Fatal("reasoning accounting omission was not diagnosed")
	}
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
	if target != llmprotocol.AnthropicMessagesV1 && len(translated.Diagnostics) == 0 {
		t.Fatal("cache-write accounting omission was not diagnosed")
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
	stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
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

func TestUnknownPreservationIsOnlyExactUnchangedSameFormat(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.UnknownFields = llmprotocol.UnknownPreserveSameFormat
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}],"future_field":{"value":1}}`)
	same, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil || !bytes.Equal(same.Body, body) {
		t.Fatalf("same-format replay = %s, %v", same.Body, err)
	}
	if _, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, body, nil); err == nil {
		t.Fatal("cross-format translation silently dropped an unknown field")
	}
	if _, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, func(*llmprotocol.Request) error { return nil }); err == nil {
		t.Fatal("mutated same-format translation silently dropped an unknown field")
	}
}

func TestPolicyEnumsAndEveryLimitAreClosed(t *testing.T) {
	for name, mutate := range map[string]func(*llmprotocol.Policy){
		"unknown fields": func(policy *llmprotocol.Policy) {
			policy.UnknownFields = llmprotocol.UnknownFieldPolicy("future")
		},
		"lossy features": func(policy *llmprotocol.Policy) {
			policy.LossyFeatures = llmprotocol.LossyPolicy("future")
		},
		"missing IDs": func(policy *llmprotocol.Policy) {
			policy.MissingStableIDs = llmprotocol.MissingIDPolicy("future")
		},
		"source preservation": func(policy *llmprotocol.Policy) {
			policy.SourcePreservation = llmprotocol.SourcePreservationPolicy("future")
		},
	} {
		t.Run(name, func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			mutate(&policy)
			if _, err := NewEngine(NewBuiltinRegistry(), policy); err == nil {
				t.Fatal("unknown policy value was accepted")
			}
		})
	}

	limitType := reflect.TypeOf(llmprotocol.Limits{})
	for index := 0; index < limitType.NumField(); index++ {
		field := limitType.Field(index)
		t.Run("zero "+field.Name, func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			limitValue := reflect.ValueOf(&policy.Limits).Elem().FieldByName(field.Name)
			if limitValue.Kind() != reflect.Int {
				t.Fatalf("limit %s has unsupported kind %s", field.Name, limitValue.Kind())
			}
			limitValue.SetInt(0)
			if _, err := NewEngine(NewBuiltinRegistry(), policy); err == nil {
				t.Fatalf("zero %s limit was accepted", field.Name)
			}
		})
	}
}

func TestCrossFormatFidelityAndCapabilityFailuresAreExplicit(t *testing.T) {
	engine := NewBuiltinEngine()
	developer := []byte(`{"model":"source-model","messages":[{"role":"developer","content":"preserve authority"},{"role":"user","content":"hello"}],"max_tokens":8}`)
	if _, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, developer, nil); err == nil {
		t.Fatal("developer authority was silently collapsed")
	}
	strictTool := []byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"},"strict":true}}]}`)
	translatedTool, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, strictTool, nil)
	if err != nil {
		t.Fatalf("official Anthropic strict tool schema was rejected: %v", err)
	}
	decodedTool, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, translatedTool.Body)
	if err != nil || len(decodedTool.Tools) != 1 || decodedTool.Tools[0].Strict == nil || !*decodedTool.Tools[0].Strict {
		t.Fatalf("strict tool schema changed: %+v, %v", decodedTool.Tools, err)
	}
	refusal := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","refusal":"no"},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	if _, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, refusal, nil); err == nil {
		t.Fatal("refusal semantics were silently converted to text")
	}
}

func TestRequestOptionMatrixNeverSilentlyDropsSemantics(t *testing.T) {
	tests := []struct {
		name   string
		source llmprotocol.WireFormat
		target llmprotocol.WireFormat
		body   string
	}{
		{
			name:   "reasoning budget cannot become Responses effort",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"reasoning_budget_tokens":512}`,
		},
		{
			name:   "Anthropic top k cannot disappear in Chat",
			source: llmprotocol.AnthropicMessagesV1, target: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"top_k":8}`,
		},
		{
			name:   "Chat seed cannot disappear in Responses",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"seed":7}`,
		},
		{
			name:   "Chat penalties cannot disappear in Anthropic",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"frequency_penalty":0.25}`,
		},
		{
			name:   "stop sequences cannot disappear in Responses",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"stop":["END"]}`,
		},
		{
			name:   "arbitrary metadata cannot collapse into Anthropic user id",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"metadata":{"trace":"keep"}}`,
		},
		{
			name:   "automatic storage is Responses only",
			source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","input":"hello","auto_store":true}`,
		},
		{
			name:   "conversation state cannot disappear in Chat",
			source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","input":"hello","previous_response_id":"resp_previous"}`,
		},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := engine.TranslateRequest(test.source, test.target, []byte(test.body), nil)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
				t.Fatalf("TranslateRequest() error = %T %v, want typed unsupported_feature", err, err)
			}
		})
	}
}

func TestRequestOptionMatrixPreservesSharedSemantics(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		source llmprotocol.WireFormat
		target llmprotocol.WireFormat
		body   string
		assert func(*testing.T, llmprotocol.Request)
	}{
		{
			name:   "reasoning budget Chat to Anthropic",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"reasoning_budget_tokens":1024}`,
			assert: func(t *testing.T, request llmprotocol.Request) {
				if request.ReasoningBudgetTokens == nil || *request.ReasoningBudgetTokens != 1024 {
					t.Fatalf("reasoning budget = %v", request.ReasoningBudgetTokens)
				}
			},
		},
		{
			name:   "stop sequences Chat to Anthropic",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"stop":["END","DONE"]}`,
			assert: func(t *testing.T, request llmprotocol.Request) {
				if !reflect.DeepEqual(request.Sampling.Stop, []string{"END", "DONE"}) {
					t.Fatalf("stop sequences = %v", request.Sampling.Stop)
				}
			},
		},
		{
			name:   "request metadata Chat to Responses",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"metadata":{"trace":"keep"}}`,
			assert: func(t *testing.T, request llmprotocol.Request) {
				if request.Metadata["trace"] != "keep" {
					t.Fatalf("metadata = %v", request.Metadata)
				}
			},
		},
		{
			name:   "reasoning effort Chat to Anthropic output config",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"reasoning_effort":"high"}`,
			assert: func(t *testing.T, request llmprotocol.Request) {
				if request.ReasoningEffort != "high" || request.ReasoningBudgetTokens != nil {
					t.Fatalf("reasoning controls = effort %q budget %v", request.ReasoningEffort, request.ReasoningBudgetTokens)
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translated, err := engine.TranslateRequest(test.source, test.target, []byte(test.body), nil)
			if err != nil {
				t.Fatal(err)
			}
			decoded, _, _, err := engine.DecodeRequest(test.target, translated.Body)
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v, body=%s", err, translated.Body)
			}
			test.assert(t, decoded)
		})
	}
}

func TestCandidateCountIsBoundedAndNeverSilentlyDropped(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"n":2}`)
	decoded, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
	if err != nil || decoded.CandidateCount == nil || *decoded.CandidateCount != 2 {
		t.Fatalf("candidate count decode = %+v, %v", decoded.CandidateCount, err)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		if _, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, target, body, nil); err == nil {
			t.Fatalf("candidate count was silently dropped for %s", target)
		}
	}
}

func TestResponsesContinuationMarksRetainedToolResultLink(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"source-model","previous_response_id":"response_previous","input":[{"type":"function_call_output","call_id":"call_previous","output":"done"}],"max_output_tokens":8}`)
	request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	result := request.Messages[0].Content[0].ToolResult
	if result == nil || !result.DeferredLink {
		t.Fatalf("retained tool result link = %+v", result)
	}
}

func TestRegistryRejectsDuplicateAndStatefulCodecsAndIsConcurrentReadSafe(t *testing.T) {
	if _, err := NewRegistry(OpenAIChatCodec{}, OpenAIChatCodec{}); err == nil {
		t.Fatal("duplicate codec was accepted")
	}
	if _, err := NewRegistry(statefulCodec{OpenAIChatCodec{}}); err == nil {
		t.Fatal("stateful codec was accepted")
	}
	registry := NewBuiltinRegistry()
	errorsChannel := make(chan error, 32)
	for index := 0; index < cap(errorsChannel); index++ {
		go func() { errorsChannel <- registry.Check(builtinFormats) }()
	}
	for index := 0; index < cap(errorsChannel); index++ {
		if err := <-errorsChannel; err != nil {
			t.Fatal(err)
		}
	}
}

type statefulCodec struct{ OpenAIChatCodec }

func (statefulCodec) Stateless() bool { return false }

func FuzzTranslateOpenAIChatRequestNeverPanics(f *testing.F) {
	f.Add(requestFixture(llmprotocol.OpenAIChatV1))
	engine := NewBuiltinEngine()
	f.Fuzz(func(t *testing.T, body []byte) {
		_, _ = engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, body, nil)
	})
}

func decodeTranslatedResponse(engine *Engine, format llmprotocol.WireFormat, body []byte) (llmprotocol.Response, error) {
	result, err := engine.TranslateResponse(format, format, body, nil)
	return result.Response, err
}

func requestFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"lookup","description":"Lookup","parameters":{"type":"object"}}}],"tool_choice":"auto"}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"source-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}],"max_output_tokens":8,"tools":[{"type":"function","name":"lookup","description":"Lookup","parameters":{"type":"object"}}],"tool_choice":"auto"}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"source-model","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"tools":[{"name":"lookup","description":"Lookup","input_schema":{"type":"object"}}],"tool_choice":{"type":"auto"}}`)
	default:
		panic("unknown fixture format")
	}
}

func toolChoiceFixture(format llmprotocol.WireFormat, mode llmprotocol.ToolChoiceMode) []byte {
	choice := ""
	switch format {
	case llmprotocol.OpenAIChatV1:
		switch mode {
		case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired:
			choice = fmt.Sprintf(`,"tool_choice":%q`, mode)
		case llmprotocol.ToolChoiceNamed:
			choice = `,"tool_choice":{"type":"function","function":{"name":"lookup"}}`
		}
		return []byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]` + choice + `}`)
	case llmprotocol.OpenAIResponsesV1:
		switch mode {
		case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceRequired:
			choice = fmt.Sprintf(`,"tool_choice":%q`, mode)
		case llmprotocol.ToolChoiceNamed:
			choice = `,"tool_choice":{"type":"function","name":"lookup"}`
		}
		return []byte(`{"model":"source-model","input":"hello","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]` + choice + `}`)
	case llmprotocol.AnthropicMessagesV1:
		switch mode {
		case llmprotocol.ToolChoiceAuto:
			choice = `,"tool_choice":{"type":"auto"}`
		case llmprotocol.ToolChoiceNone:
			choice = `,"tool_choice":{"type":"none"}`
		case llmprotocol.ToolChoiceRequired:
			choice = `,"tool_choice":{"type":"any"}`
		case llmprotocol.ToolChoiceNamed:
			choice = `,"tool_choice":{"type":"tool","name":"lookup"}`
		}
		return []byte(`{"model":"source-model","max_tokens":8,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]` + choice + `}`)
	default:
		panic("unknown fixture format")
	}
}

func toolLifecycleFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"model":"source-model","messages":[{"role":"user","content":"weather"},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Paris\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"sunny"},{"role":"user","content":"summarize"}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"lookup","description":"Lookup weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":{"type":"function","function":{"name":"lookup"}},"parallel_tool_calls":false}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"source-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather"}]},{"type":"function_call","id":"item_call_1","call_id":"call_1","name":"lookup","arguments":"{\"city\":\"Paris\"}"},{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_text","text":"sunny"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"summarize"}]}],"max_output_tokens":8,"tools":[{"type":"function","name":"lookup","description":"Lookup weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":{"type":"function","name":"lookup"},"parallel_tool_calls":false}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"source-model","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"weather"}]},{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{"city":"Paris"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"sunny"}]}]},{"role":"user","content":[{"type":"text","text":"summarize"}]}],"tools":[{"name":"lookup","description":"Lookup weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":{"type":"tool","name":"lookup","disable_parallel_tool_use":true}}`)
	default:
		panic("unknown fixture format")
	}
}

func orderedImageFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"model":"source-model","messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}},{"type":"text","text":"after"}]}],"max_tokens":8}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"source-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"before"},{"type":"input_image","image_url":"https://example.com/image.png"},{"type":"input_text","text":"after"}]}],"max_output_tokens":8}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"source-model","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image","source":{"type":"url","url":"https://example.com/image.png"}},{"type":"text","text":"after"}]}]}`)
	default:
		panic("unknown fixture format")
	}
}

func fileInputFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"model":"source-model","max_completion_tokens":32,"messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file_1"}}]}]}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"source-model","max_output_tokens":32,"input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"file_1"}]}]}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"source-model","max_tokens":32,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":"file_1"}}]}]}`)
	default:
		panic("unknown format")
	}
}

func assertToolLifecycle(t *testing.T, request llmprotocol.Request) {
	t.Helper()
	assertToolPolicy(t, request)
	callIndex, resultIndex := findToolLifecycle(t, request.Messages)
	if callIndex < 0 || resultIndex <= callIndex {
		t.Fatalf("tool lifecycle order = call:%d result:%d messages=%+v", callIndex, resultIndex, request.Messages)
	}
}

func assertToolPolicy(t *testing.T, request llmprotocol.Request) {
	t.Helper()
	if len(request.Tools) != 1 || request.Tools[0].Name != "lookup" ||
		request.ToolChoice.Mode != llmprotocol.ToolChoiceNamed || request.ToolChoice.Name != "lookup" ||
		request.ParallelToolCalls == nil || *request.ParallelToolCalls {
		t.Fatalf("tool policy changed: tools=%+v choice=%+v parallel=%v", request.Tools, request.ToolChoice, request.ParallelToolCalls)
	}
}

func findToolLifecycle(t *testing.T, messages []llmprotocol.Message) (int, int) {
	t.Helper()
	callIndex, resultIndex := -1, -1
	for messageIndex, message := range messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil {
				assertToolCallContent(t, content.ToolCall)
				callIndex = messageIndex
			}
			if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
				assertToolResultContent(t, content.ToolResult)
				resultIndex = messageIndex
			}
		}
	}
	return callIndex, resultIndex
}

func assertToolCallContent(t *testing.T, call *llmprotocol.ToolCall) {
	t.Helper()
	if call.ID != "call_1" || call.Name != "lookup" || call.Arguments != `{"city":"Paris"}` {
		t.Fatalf("tool call changed: %+v", call)
	}
}

func assertToolResultContent(t *testing.T, result *llmprotocol.ToolResult) {
	t.Helper()
	if result.CallID != "call_1" || len(result.Content) != 1 ||
		result.Content[0].Kind != llmprotocol.ContentText || result.Content[0].Text != "sunny" {
		t.Fatalf("tool result changed: %+v", result)
	}
}

func responseFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"id":"output_1","role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"id":"response_1","model":"source-model","status":"completed","output":[{"type":"message","id":"output_1","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"id":"response_1","type":"message","role":"assistant","model":"source-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
	default:
		panic("unknown fixture format")
	}
}

func streamFixture(format llmprotocol.WireFormat) []byte {
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
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"created_at\":100,\"model\":\"source-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
			"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":2,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}\n\n",
			"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"delta\":\"hello\"}\n\n",
			"event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"sequence_number\":4,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"text\":\"hello\"}\n\n",
			"event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":5,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"part\":{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}}\n\n",
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":6,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}}\n\n",
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":7,\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"created_at\":100,\"model\":\"source-model\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":2,\"input_tokens_details\":{\"cached_tokens\":0,\"cache_write_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":3}}}\n\n",
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
		panic(fmt.Sprintf("unknown fixture format %q", format))
	}
}

func toolStreamFixture(format llmprotocol.WireFormat) []byte {
	join := func(events ...string) []byte { return []byte(strings.Join(events, "")) }
	arguments := `{"protocol":"source"}`
	first, second := arguments[:len(arguments)/2], arguments[len(arguments)/2:]
	switch format {
	case llmprotocol.OpenAIChatV1:
		return join(
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":"+fmt.Sprintf("%q", first)+"}}]},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":"+fmt.Sprintf("%q", second)+"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
			"data: [DONE]\n\n",
		)
	case llmprotocol.OpenAIResponsesV1:
		return join(
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"created_at\":100,\"model\":\"source-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\",\"status\":\"in_progress\"}}\n\n",
			"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"item_1\",\"delta\":"+fmt.Sprintf("%q", first)+"}\n\n",
			"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":3,\"output_index\":0,\"item_id\":\"item_1\",\"delta\":"+fmt.Sprintf("%q", second)+"}\n\n",
			"event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"sequence_number\":4,\"output_index\":0,\"item_id\":\"item_1\",\"name\":\"lookup\",\"arguments\":"+fmt.Sprintf("%q", arguments)+"}\n\n",
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":5,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":"+fmt.Sprintf("%q", arguments)+",\"status\":\"completed\"}}\n\n",
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":6,\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"created_at\":100,\"model\":\"source-model\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":"+fmt.Sprintf("%q", arguments)+",\"status\":\"completed\"}],\"usage\":{\"input_tokens\":2,\"input_tokens_details\":{\"cached_tokens\":0,\"cache_write_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":3}}}\n\n",
		)
	case llmprotocol.AnthropicMessagesV1:
		return join(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"source-model\",\"content\":[],\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"lookup\",\"input\":{}}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":"+fmt.Sprintf("%q", first)+"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":"+fmt.Sprintf("%q", second)+"}}\n\n",
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"type\":\"message_delta\",\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":1}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		)
	default:
		panic(fmt.Sprintf("unknown fixture format %q", format))
	}
}

func assertStreamToolCall(t *testing.T, events []llmprotocol.Event) {
	t.Helper()
	completed, terminal := collectStreamToolCall(t, events)
	if completed == nil || completed.ID != "call_1" || completed.Name != "lookup" || completed.Arguments != `{"protocol":"source"}` {
		t.Fatalf("completed tool call = %+v", completed)
	}
	if terminal != 1 {
		t.Fatalf("terminal events = %d, events=%+v", terminal, events)
	}
}

func collectStreamToolCall(t *testing.T, events []llmprotocol.Event) (*llmprotocol.ToolCall, int) {
	t.Helper()
	var completed *llmprotocol.ToolCall
	terminal := 0
	for _, event := range events {
		if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil {
			call := *event.ToolCall
			completed = &call
		}
		if event.Type == llmprotocol.EventResponseCompleted {
			terminal++
			assertAuthoritativeTerminalUsage(t, event.Usage)
		}
	}
	return completed, terminal
}

func assertAuthoritativeTerminalUsage(t *testing.T, usage *llmprotocol.Usage) {
	t.Helper()
	if usage == nil || usage.State != llmprotocol.UsageAvailable ||
		usage.InputTotal.Value == nil || usage.InputTotal.Provenance != llmprotocol.UsageAuthoritative ||
		usage.OutputTotal.Value == nil || usage.OutputTotal.Provenance != llmprotocol.UsageAuthoritative {
		t.Fatalf("terminal usage = %+v", usage)
	}
}
