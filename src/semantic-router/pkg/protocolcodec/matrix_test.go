package protocolcodec

import (
	"bytes"
	"context"
	"fmt"
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
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
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
			})
		}
	}
}

func TestBufferedResponseMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
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
			})
		}
	}
}

func TestAuthoritativeUsageEvidenceSurvivesClientRepresentationLoss(t *testing.T) {
	engine := NewBuiltinEngine()
	chat := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":2}}}`)
	for _, target := range builtinFormats {
		t.Run("reasoning_and_cache_read_to_"+string(target), func(t *testing.T) {
			translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, chat, nil)
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
		})
	}

	messages := []byte(`{"id":"response_1","type":"message","role":"assistant","model":"source-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}`)
	for _, target := range builtinFormats {
		t.Run("cache_write_to_"+string(target), func(t *testing.T) {
			translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, messages, nil)
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
		})
	}
}

func TestRefusalAndReasoningFidelityIsExplicitAcrossFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	reasoning := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"check","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}`)
	for _, target := range builtinFormats {
		translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, reasoning, nil)
		if err != nil {
			t.Fatalf("reasoning to %s: %v", target, err)
		}
		found := false
		for _, item := range translated.Response.Output {
			for _, content := range item.Content {
				found = found || content.Kind == llmprotocol.ContentReasoning && content.Text == "check"
			}
		}
		if !found {
			t.Fatalf("reasoning semantic lost for %s: %+v", target, translated.Response.Output)
		}
	}

	refusal := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","refusal":"cannot comply"},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	for _, target := range builtinFormats {
		translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, refusal, nil)
		if target == llmprotocol.AnthropicMessagesV1 {
			if err == nil {
				t.Fatal("refusal was silently weakened to ordinary Messages text")
			}
			continue
		}
		if err != nil || len(translated.Response.Output) != 1 ||
			translated.Response.Output[0].Content[0].Kind != llmprotocol.ContentRefusal {
			t.Fatalf("refusal to %s = %+v, %v", target, translated.Response.Output, err)
		}
	}
}

func TestStreamingMatrixAcceptsArbitraryTransportChunks(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				payload := streamFixture(source)
				var events []llmprotocol.Event
				var encoded bytes.Buffer
				for offset := 0; offset < len(payload); {
					size := 1 + offset%17
					end := offset + size
					if end > len(payload) {
						end = len(payload)
					}
					frames, decoded, _, pushErr := stream.Push(payload[offset:end])
					if pushErr != nil {
						t.Fatalf("Push(%d:%d) error = %v", offset, end, pushErr)
					}
					for _, frame := range frames {
						encoded.Write(frame)
					}
					events = append(events, decoded...)
					offset = end
				}
				frames, decoded, _, err := stream.Finalize(nil)
				if err != nil {
					t.Fatalf("Finalize() error = %v", err)
				}
				for _, frame := range frames {
					encoded.Write(frame)
				}
				events = append(events, decoded...)
				if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted || events[len(events)-1].Usage == nil {
					t.Fatalf("terminal events = %+v", events)
				}
				if encoded.Len() == 0 {
					t.Fatal("target stream is empty")
				}

				verify, err := engine.NewStream(target, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				_, targetEvents, _, err := verify.Push(encoded.Bytes())
				if err != nil {
					t.Fatalf("target wire did not decode: %v\n%s", err, encoded.Bytes())
				}
				_, finalized, _, finalizeErr := verify.Finalize(nil)
				targetEvents = append(targetEvents, finalized...)
				if finalizeErr != nil {
					t.Fatalf("target Finalize() error = %v", finalizeErr)
				}
				if len(targetEvents) == 0 || targetEvents[len(targetEvents)-1].Type != llmprotocol.EventResponseCompleted {
					t.Fatalf("target terminal events = %+v", targetEvents)
				}
			})
		}
	}
}

func TestStreamingToolCallMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		for _, target := range builtinFormats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				stream, err := engine.NewStream(source, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				payload := toolStreamFixture(source)
				var events []llmprotocol.Event
				var encoded bytes.Buffer
				for offset := 0; offset < len(payload); {
					end := offset + 1 + offset%13
					if end > len(payload) {
						end = len(payload)
					}
					frames, decoded, _, pushErr := stream.Push(payload[offset:end])
					if pushErr != nil {
						t.Fatalf("Push(%d:%d): %v", offset, end, pushErr)
					}
					for _, frame := range frames {
						encoded.Write(frame)
					}
					events = append(events, decoded...)
					offset = end
				}
				frames, decoded, _, finalizeErr := stream.Finalize(nil)
				if finalizeErr != nil {
					t.Fatal(finalizeErr)
				}
				for _, frame := range frames {
					encoded.Write(frame)
				}
				events = append(events, decoded...)
				assertStreamToolCall(t, events)

				verify, err := engine.NewStream(target, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
				if err != nil {
					t.Fatal(err)
				}
				_, targetEvents, _, pushErr := verify.Push(encoded.Bytes())
				if pushErr != nil {
					t.Fatalf("target stream did not decode: %v\n%s", pushErr, encoded.Bytes())
				}
				_, finalEvents, _, finalizeErr := verify.Finalize(nil)
				if finalizeErr != nil {
					t.Fatal(finalizeErr)
				}
				assertStreamToolCall(t, append(targetEvents, finalEvents...))
			})
		}
	}
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

func TestCrossFormatFidelityAndCapabilityFailuresAreExplicit(t *testing.T) {
	engine := NewBuiltinEngine()
	developer := []byte(`{"model":"source-model","messages":[{"role":"developer","content":"preserve authority"},{"role":"user","content":"hello"}],"max_tokens":8}`)
	if _, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, developer, nil); err == nil {
		t.Fatal("developer authority was silently collapsed")
	}
	strictTool := []byte(`{"model":"source-model","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"},"strict":true}}]}`)
	if _, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, strictTool, nil); err == nil {
		t.Fatal("strict tool schema was silently weakened")
	}
	refusal := []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","refusal":"no"},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	if _, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, refusal, nil); err == nil {
		t.Fatal("refusal semantics were silently converted to text")
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

func assertToolLifecycle(t *testing.T, request llmprotocol.Request) {
	t.Helper()
	if len(request.Tools) != 1 || request.Tools[0].Name != "lookup" ||
		request.ToolChoice.Mode != llmprotocol.ToolChoiceNamed || request.ToolChoice.Name != "lookup" ||
		request.ParallelToolCalls == nil || *request.ParallelToolCalls {
		t.Fatalf("tool policy changed: tools=%+v choice=%+v parallel=%v", request.Tools, request.ToolChoice, request.ParallelToolCalls)
	}
	callIndex, resultIndex := -1, -1
	for messageIndex, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil {
				if content.ToolCall.ID != "call_1" || content.ToolCall.Name != "lookup" || content.ToolCall.Arguments != `{"city":"Paris"}` {
					t.Fatalf("tool call changed: %+v", content.ToolCall)
				}
				callIndex = messageIndex
			}
			if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
				if content.ToolResult.CallID != "call_1" || len(content.ToolResult.Content) != 1 ||
					content.ToolResult.Content[0].Kind != llmprotocol.ContentText || content.ToolResult.Content[0].Text != "sunny" {
					t.Fatalf("tool result changed: %+v", content.ToolResult)
				}
				resultIndex = messageIndex
			}
		}
	}
	if callIndex < 0 || resultIndex <= callIndex {
		t.Fatalf("tool lifecycle order = call:%d result:%d messages=%+v", callIndex, resultIndex, request.Messages)
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
			"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"response_1\",\"model\":\"source-model\",\"status\":\"in_progress\"}}\n\n",
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n",
			"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"item_1\",\"delta\":"+fmt.Sprintf("%q", first)+"}\n\n",
			"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"item_1\",\"delta\":"+fmt.Sprintf("%q", second)+"}\n\n",
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item_1\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":"+fmt.Sprintf("%q", arguments)+"}}\n\n",
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"response_1\",\"model\":\"source-model\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
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
	var completed *llmprotocol.ToolCall
	terminal := 0
	for _, event := range events {
		if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil {
			call := *event.ToolCall
			completed = &call
		}
		if event.Type == llmprotocol.EventResponseCompleted {
			terminal++
			if event.Usage == nil || event.Usage.State != llmprotocol.UsageAvailable ||
				event.Usage.InputTotal.Value == nil || event.Usage.InputTotal.Provenance != llmprotocol.UsageAuthoritative ||
				event.Usage.OutputTotal.Value == nil || event.Usage.OutputTotal.Provenance != llmprotocol.UsageAuthoritative {
				t.Fatalf("terminal usage = %+v", event.Usage)
			}
		}
	}
	if completed == nil || completed.ID != "call_1" || completed.Name != "lookup" || completed.Arguments != `{"protocol":"source"}` {
		t.Fatalf("completed tool call = %+v", completed)
	}
	if terminal != 1 {
		t.Fatalf("terminal events = %d, events=%+v", terminal, events)
	}
}
