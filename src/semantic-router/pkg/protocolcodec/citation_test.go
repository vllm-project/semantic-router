package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatResponseAcceptsNullAnnotationsWithoutWeakeningUnknownFields(t *testing.T) {
	engine := NewBuiltinEngine()
	withNull := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello","annotations":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, withNull, func(response *llmprotocol.Response) error {
		response.Model = "public-model"
		return nil
	})
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if !bytes.Contains(translated.Body, []byte(`"model":"public-model"`)) {
		t.Fatalf("translated body = %s", translated.Body)
	}

	unknown := bytes.Replace(withNull, []byte(`"annotations":null`), []byte(`"annotations":null,"future_field":true`), 1)
	if _, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, unknown, func(*llmprotocol.Response) error { return nil }); err == nil {
		t.Fatal("unknown Chat response field was accepted")
	}
}

func TestURLCitationTranslatesBetweenOpenAIFormatsAndFailsClosedForMessages(t *testing.T) {
	engine := NewBuiltinEngine()
	chat := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":"source","annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com/source","title":"Source","start_index":0,"end_index":6}}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)

	translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, chat, nil)
	if err != nil {
		t.Fatalf("Chat to Responses error = %v", err)
	}
	if len(translated.Response.Output) != 1 || len(translated.Response.Output[0].Content) != 1 ||
		len(translated.Response.Output[0].Content[0].Citations) != 1 {
		t.Fatalf("neutral citations = %+v", translated.Response.Output)
	}
	if !bytes.Contains(translated.Body, []byte(`"type":"url_citation"`)) ||
		!bytes.Contains(translated.Body, []byte(`"url":"https://example.com/source"`)) {
		t.Fatalf("Responses citation missing: %s", translated.Body)
	}
	decoded, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, translated.Body)
	if err != nil || decoded.Output[0].Content[0].Citations[0].Title != "Source" {
		t.Fatalf("Responses citation decode = %+v, %v", decoded.Output, err)
	}
	responses := []byte(`{"id":"response_2","model":"source-model","status":"completed","output":[{"type":"message","id":"item_1","role":"assistant","content":[{"type":"output_text","text":"source","annotations":[{"type":"url_citation","url":"https://example.com/source","title":"Source","start_index":0,"end_index":6}]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	reverse, err := engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, responses, nil)
	if err != nil {
		t.Fatalf("Responses to Chat error = %v", err)
	}
	if !bytes.Contains(reverse.Body, []byte(`"type":"url_citation"`)) ||
		!bytes.Contains(reverse.Body, []byte(`"url_citation":{"url":"https://example.com/source"`)) {
		t.Fatalf("Chat citation missing: %s", reverse.Body)
	}
	if _, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, chat, nil); err == nil {
		t.Fatal("Chat citation was silently dropped for Messages")
	}
}

func TestChatToolResponseAcceptsNullAnnotations(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"annotations":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	decoded, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(decoded.Output) != 1 || len(decoded.Output[0].Content) != 1 || decoded.Output[0].Content[0].Kind != llmprotocol.ContentToolCall {
		t.Fatalf("tool response = %+v", decoded.Output)
	}
}

func TestChatCitationStreamTranslatesWithMonotonicResponsesIndexes(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	payload := strings.Join([]string{
		`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"role":"assistant","annotations":null},"finish_reason":null}]}`,
		`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"content":"first second"},"finish_reason":null}]}`,
		`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com/one","title":"One","start_index":0,"end_index":5}}]},"finish_reason":null}]}`,
		`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"annotations":[{"type":"url_citation","url_citation":{"url":"https://example.com/two","title":"Two","start_index":6,"end_index":12}}]},"finish_reason":"stop"}]}`,
		`data: {"id":"response_1","model":"source-model","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		`data: [DONE]`,
		"",
	}, "\n\n")
	frames, _, _, err := stream.Push([]byte(payload))
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("Finalize() error = %v", err)
	}
	encoded := string(bytes.Join(append(frames, finalFrames...), nil))
	if strings.Count(encoded, `"type":"response.output_text.annotation.added"`) != 2 ||
		!strings.Contains(encoded, `"annotation_index":0`) || !strings.Contains(encoded, `"annotation_index":1`) {
		t.Fatalf("Responses annotation stream = %s", encoded)
	}
}

func TestResponsesStreamTextDeltaWithoutContentDoesNotPanic(t *testing.T) {
	encoder := OpenAIResponsesCodec{}.NewEncoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model", ResponseID: "response_1"},
		llmprotocol.DefaultPolicy(),
	)
	events := []llmprotocol.Event{
		{Type: llmprotocol.EventResponseStarted, ResponseID: "response_1", Model: "model"},
		{Type: llmprotocol.EventOutputItemStarted, ItemIndex: 0, ItemID: "item_1", Role: llmprotocol.RoleAssistant},
		{Type: llmprotocol.EventOutputTextDelta, ItemIndex: 0, ItemID: "item_1", Delta: "hello"},
	}
	var encoded bytes.Buffer
	for _, event := range events {
		frames, _, err := encoder.Push(event)
		if err != nil {
			t.Fatalf("Push(%s) error = %v", event.Type, err)
		}
		for _, frame := range frames {
			encoded.Write(frame)
		}
	}
	if !strings.Contains(encoded.String(), `"type":"response.output_text.delta"`) ||
		!strings.Contains(encoded.String(), `"delta":"hello"`) {
		t.Fatalf("Responses text delta = %s", encoded.String())
	}
}

func TestChatCitationStreamRejectsInvalidSemanticCitation(t *testing.T) {
	limits := llmprotocol.DefaultPolicy().Limits
	valid := chatAnnotationWire{Type: "url_citation", URLCitation: &chatURLCitationAnnotationWire{
		URL: "https://example.com/source", Title: "Source", StartIndex: 0, EndIndex: 6,
	}}
	tooMany := make([]chatAnnotationWire, limits.Citations+1)
	for index := range tooMany {
		copy := valid
		copy.URLCitation = &chatURLCitationAnnotationWire{
			URL: valid.URLCitation.URL, Title: valid.URLCitation.Title,
			StartIndex: valid.URLCitation.StartIndex, EndIndex: valid.URLCitation.EndIndex,
		}
		tooMany[index] = copy
	}
	for name, testCase := range map[string]struct {
		annotations []chatAnnotationWire
		code        string
		text        string
	}{
		"scheme": {annotations: []chatAnnotationWire{{Type: "url_citation", URLCitation: &chatURLCitationAnnotationWire{
			URL: "file:///tmp/source", StartIndex: 0, EndIndex: 6,
		}}}, code: "invalid_citation_url"},
		"range": {annotations: []chatAnnotationWire{{Type: "url_citation", URLCitation: &chatURLCitationAnnotationWire{
			URL: "https://example.com/source", StartIndex: 0, EndIndex: 7,
		}}}, code: "citation_range"},
		"unicode range": {annotations: []chatAnnotationWire{{Type: "url_citation", URLCitation: &chatURLCitationAnnotationWire{
			URL: "https://example.com/source", StartIndex: 0, EndIndex: 6,
		}}}, code: "citation_range", text: "来源"},
		"count": {annotations: tooMany, code: "citation_limit"},
	} {
		t.Run(name, func(t *testing.T) {
			stream := newCitationTestStream(t, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1)
			text := testCase.text
			if text == "" {
				text = "source"
			}
			pushCitationWire(t, stream, chatChunkWire{
				ID: "response_1", Model: "source-model", Choices: []chatChunkChoiceWire{{
					Index: 0, Delta: chatChunkDeltaWire{Role: "assistant", Content: &text},
				}},
			})
			_, _, _, err := stream.Push(citationSSEFrame(t, chatChunkWire{
				ID: "response_1", Model: "source-model", Choices: []chatChunkChoiceWire{{
					Index: 0, Delta: chatChunkDeltaWire{Annotations: testCase.annotations},
				}},
			}))
			requireProtocolErrorCode(t, err, testCase.code)
		})
	}
}

func TestResponsesCitationStreamRejectsInvalidAnnotationCoordinates(t *testing.T) {
	zero, one := 0, 1
	for name, testCase := range map[string]struct {
		mutate func(*responsesEventWire)
		code   string
	}{
		"annotation index": {mutate: func(wire *responsesEventWire) { wire.AnnotationIndex = &one }, code: "stream_annotation_index"},
		"missing annotation index": {mutate: func(wire *responsesEventWire) {
			wire.AnnotationIndex = nil
		}, code: "stream_annotation_index"},
		"content index": {mutate: func(wire *responsesEventWire) { wire.ContentIndex = &one }, code: "stream_annotation_content_index"},
		"missing content index": {mutate: func(wire *responsesEventWire) {
			wire.ContentIndex = nil
		}, code: "stream_annotation_content_index"},
		"item identity": {mutate: func(wire *responsesEventWire) { wire.ItemID = "item_other" }, code: "stream_annotation_item"},
	} {
		t.Run(name, func(t *testing.T) {
			stream := newCitationTestStream(t, llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1)
			pushResponsesCitationPrefix(t, stream)
			wire := responsesEventWire{
				Type: "response.output_text.annotation.added", Sequence: 4,
				ItemID: "item_1", OutputIndex: 0, ContentIndex: &zero, AnnotationIndex: &zero,
				Annotation: &responsesAnnotationWire{
					Type: "url_citation", URL: "https://example.com/source",
					StartIndex: 0, EndIndex: 6,
				},
			}
			testCase.mutate(&wire)
			_, _, _, err := stream.Push(citationSSEFrame(t, wire))
			requireProtocolErrorCode(t, err, testCase.code)
		})
	}
}

func TestCitationStreamsFailClosedWhenMessagesCannotRepresentThem(t *testing.T) {
	for _, source := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
	} {
		t.Run(string(source), func(t *testing.T) {
			stream := newCitationTestStream(t, source, llmprotocol.AnthropicMessagesV1)
			var err error
			switch source {
			case llmprotocol.OpenAIChatV1:
				text := "source"
				pushCitationWire(t, stream, chatChunkWire{
					ID: "response_1", Model: "source-model", Choices: []chatChunkChoiceWire{{
						Index: 0, Delta: chatChunkDeltaWire{Role: "assistant", Content: &text},
					}},
				})
				_, _, _, err = stream.Push(citationSSEFrame(t, chatChunkWire{
					ID: "response_1", Model: "source-model", Choices: []chatChunkChoiceWire{{
						Index: 0, Delta: chatChunkDeltaWire{Annotations: []chatAnnotationWire{{
							Type: "url_citation", URLCitation: &chatURLCitationAnnotationWire{
								URL: "https://example.com/source", StartIndex: 0, EndIndex: 6,
							},
						}}},
					}},
				}))
			case llmprotocol.OpenAIResponsesV1:
				pushResponsesCitationPrefix(t, stream)
				annotationIndex := 0
				_, _, _, err = stream.Push(citationSSEFrame(t, responsesEventWire{
					Type: "response.output_text.annotation.added", Sequence: 4,
					ItemID: "item_1", OutputIndex: 0, ContentIndex: &annotationIndex,
					AnnotationIndex: &annotationIndex,
					Annotation: &responsesAnnotationWire{
						Type: "url_citation", URL: "https://example.com/source",
						StartIndex: 0, EndIndex: 6,
					},
				}))
			}
			requireProtocolErrorCode(t, err, "lossy_translation")
		})
	}
}

func newCitationTestStream(
	t *testing.T,
	source llmprotocol.WireFormat,
	target llmprotocol.WireFormat,
) *StreamEngine {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(source, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func pushResponsesCitationPrefix(t *testing.T, stream *StreamEngine) {
	t.Helper()
	for _, wire := range []responsesEventWire{
		{
			Type: "response.created", Sequence: 1,
			Response: &responsesResponseWire{ID: "response_1", Model: "source-model", Status: "in_progress"},
		},
		{
			Type: "response.output_item.added", Sequence: 2, OutputIndex: 0,
			Item: &responsesItemWire{Type: "message", ID: "item_1", Role: "assistant"},
		},
		{
			Type: "response.output_text.delta", Sequence: 3,
			ItemID: "item_1", OutputIndex: 0, ContentIndex: responsesContentIndex(), Delta: "source",
		},
	} {
		pushCitationWire(t, stream, wire)
	}
}

func pushCitationWire(t *testing.T, stream *StreamEngine, wire any) {
	t.Helper()
	if _, _, _, err := stream.Push(citationSSEFrame(t, wire)); err != nil {
		t.Fatalf("stream prefix push error = %v", err)
	}
}

func citationSSEFrame(t *testing.T, wire any) []byte {
	t.Helper()
	payload, err := json.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	return append(append([]byte("data: "), payload...), []byte("\n\n")...)
}

func requireProtocolErrorCode(t *testing.T, err error, code string) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != code {
		t.Fatalf("protocol error = %v, want code %q", err, code)
	}
}
