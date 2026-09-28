package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const responsesCustomToolRequest = `{"model":"m","input":[{"role":"user","content":"edit hello.txt"},{"type":"custom_tool_call","id":"item_1","call_id":"call_1","name":"apply_patch","input":"*** Begin Patch\n+hello\n*** End Patch","status":"completed"},{"type":"custom_tool_call_output","call_id":"call_1","output":"Success","status":"completed"}],"tools":[{"type":"custom","name":"apply_patch","description":"Apply a patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}}]}`

func TestResponsesCustomToolLoopCrossesChatAndResponses(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(responsesCustomToolRequest))
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Tools) != 1 || request.Tools[0].Kind != llmprotocol.ToolKindCustom ||
		request.Tools[0].CustomFormat == nil || request.Tools[0].CustomFormat.Syntax != "lark" {
		t.Fatalf("custom grammar lost at ingress: %+v", request.Tools)
	}
	call, result := request.Messages[1].Content[0].ToolCall, request.Messages[2].Content[0].ToolResult
	if call.Kind != llmprotocol.ToolKindCustom || result.Kind != llmprotocol.ToolKindCustom ||
		call.ID != result.CallID || !strings.Contains(call.Arguments, "Begin Patch") || result.Content[0].Text != "Success" {
		t.Fatalf("custom call or standalone result lost: call=%+v result=%+v", call, result)
	}
	request.Model, request.Generation = "routed-model", request.Generation+1
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		encoded, encodeErr := engine.EncodeRequest(format, request, envelope)
		if encodeErr != nil {
			t.Fatalf("to %s: %v", format, encodeErr)
		}
		if format == llmprotocol.OpenAIResponsesV1 {
			var wire responsesRequestWire
			if decodeErr := json.Unmarshal(encoded.Body, &wire); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			if !bytes.Contains(wire.Tools, []byte(`"syntax":"lark"`)) ||
				!bytes.Contains(wire.Input, []byte(`"type":"custom_tool_call_output"`)) {
				t.Fatalf("Responses dispatch changed custom variant: %s", encoded.Body)
			}
			if _, _, _, decodeErr := engine.DecodeRequest(format, encoded.Body); decodeErr != nil {
				t.Fatalf("Responses dispatch does not decode: %v", decodeErr)
			}
		} else if !bytes.Contains(encoded.Body, []byte(`"type":"custom"`)) ||
			!bytes.Contains(encoded.Body, []byte(`"tool_call_id":"call_1"`)) {
			t.Fatalf("Chat dispatch changed custom call: %s", encoded.Body)
		}
	}
	_, err = engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_capability" {
		t.Fatalf("Anthropic custom tools returned %v, want unsupported_capability", err)
	}
}

func TestResponsesStandaloneCustomOutputKeepsKind(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","previous_response_id":"resp_1","input":[{"type":"custom_tool_call_output","call_id":"call_1","output":"done"}]}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Model, request.Generation = "routed-model", request.Generation+1
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil || !bytes.Contains(encoded.Body, []byte(`"type":"custom_tool_call_output"`)) {
		t.Fatalf("standalone custom result was relabeled: %v %s", err, encoded.Body)
	}
}

func TestResponsesBufferedCustomCallReachesChat(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"resp_1","object":"response","created_at":100,"model":"m","status":"completed","output":[{"type":"custom_tool_call","id":"item_1","call_id":"call_1","name":"apply_patch","input":"free form patch","status":"completed"}]}`)
	response, envelope, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	call := response.Output[0].Content[0].ToolCall
	if call.Kind != llmprotocol.ToolKindCustom || call.Arguments != "free form patch" {
		t.Fatalf("buffered custom call lost: %+v", call)
	}
	response.Model, response.Generation = "public-model", response.Generation+1
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, envelope)
	if err != nil || !bytes.Contains(encoded.Body, []byte(`"type":"custom_tool_call"`)) {
		t.Fatalf("buffered response changed custom variant: %v %s", err, encoded.Body)
	}
	chat, err := engine.EncodeResponse(llmprotocol.OpenAIChatV1, response, envelope)
	if err != nil || !bytes.Contains(chat.Body, []byte(`"type":"custom"`)) {
		t.Fatalf("Chat response lost custom call: %v %s", err, chat.Body)
	}
}

func TestResponsesCustomToolStreamTranslatesWithoutLosingInput(t *testing.T) {
	payload := customToolStreamFixture()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIResponsesV1, target,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model"})
		if err != nil {
			t.Fatal(err)
		}
		frames, events, _, err := stream.Push(payload)
		if err != nil {
			t.Fatalf("stream to %s: %v", target, err)
		}
		finalFrames, finalEvents, _, err := stream.Finalize(nil)
		if err != nil {
			t.Fatalf("finalize to %s: %v", target, err)
		}
		frames, events = append(frames, finalFrames...), append(events, finalEvents...)
		found := false
		for _, event := range events {
			if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil &&
				event.ToolCall.Kind == llmprotocol.ToolKindCustom && event.ToolCall.Arguments == `{"protocol":"source"}` {
				found = true
			}
		}
		if !found {
			t.Fatalf("stream to %s lost custom tool input: %+v", target, events)
		}
		if target == llmprotocol.OpenAIResponsesV1 && !bytes.Contains(bytes.Join(frames, nil), []byte("response.custom_tool_call_input.done")) {
			t.Fatalf("Responses stream used wrong event type: %s", bytes.Join(frames, nil))
		}
		if target == llmprotocol.OpenAIChatV1 && !bytes.Contains(bytes.Join(frames, nil), []byte(`"type":"custom"`)) {
			t.Fatalf("Chat stream lost custom kind: %s", bytes.Join(frames, nil))
		}
	}
}

func customToolStreamFixture() []byte {
	return []byte(strings.NewReplacer(
		"response.function_call_arguments.", "response.custom_tool_call_input.",
		`"function_call"`, `"custom_tool_call"`,
		`"arguments":`, `"input":`,
	).Replace(string(toolStreamFixture(llmprotocol.OpenAIResponsesV1))))
}

func TestResponsesCustomToolStreamRejectsChangedCompletedInput(t *testing.T) {
	frames := strings.Split(string(customToolStreamFixture()), "\n\n")
	for index, frame := range frames {
		const prefix = "event: response.output_item.done\ndata: "
		if !strings.HasPrefix(frame, prefix) {
			continue
		}
		var wire map[string]json.RawMessage
		if err := json.Unmarshal([]byte(strings.TrimPrefix(frame, prefix)), &wire); err != nil {
			t.Fatal(err)
		}
		var item map[string]json.RawMessage
		if err := json.Unmarshal(wire["item"], &item); err != nil {
			t.Fatal(err)
		}
		item["input"] = json.RawMessage(`""`)
		wire["item"], _ = json.Marshal(item)
		encoded, _ := json.Marshal(wire)
		frames[index] = prefix + string(encoded)
	}
	decoder := newProviderStreamDecoder(llmprotocol.OpenAIResponsesV1)
	_, _, err := decoder.Push([]byte(strings.Join(frames, "\n\n")))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_mismatch")
}

func TestChatIDOnlyToolDeltaWaitsForCustomKindBeforeResponsesItem(t *testing.T) {
	// The first Chat delta announces only an ID. The next delta declares the
	// custom kind and name, as Chat providers may do in incremental streams.
	payload := strings.Join([]string{
		`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1"}]},"finish_reason":null}]}`,
		`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"custom","custom":{"name":"apply_patch","input":"*** Begin Patch\n"}}]},"finish_reason":null}]}`,
		`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"custom":{"input":"*** End Patch\n"}}]},"finish_reason":"tool_calls"}]}`,
		`data: {"id":"response_1","model":"source-model","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(payload))
	if err != nil {
		t.Fatal(err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	output := bytes.Join(append(frames, final...), nil)
	if bytes.Contains(output, []byte(`"type":"function_call"`)) ||
		!bytes.Contains(output, []byte(`"type":"custom_tool_call"`)) ||
		!bytes.Contains(output, []byte("response.custom_tool_call_input.done")) {
		t.Fatalf("Chat id-only delta was given the wrong Responses kind: %s", output)
	}
}

func TestChatDelayedToolIdentityFlushesInputAfterResponsesItem(t *testing.T) {
	tests := []struct {
		name, kind, expectedInput string
		calls                     []string
	}{
		{
			name: "custom input before name", kind: "custom_tool_call",
			expectedInput: "*** Begin Patch\n+hello\n*** End Patch",
			calls: []string{
				`{"index":0,"type":"custom","custom":{"input":"*** Begin Patch\n"}}`,
				`{"index":0,"id":"call_1","custom":{"input":"+hello\n"}}`,
				`{"index":0,"custom":{"name":"apply_patch","input":"*** End Patch"}}`,
			},
		},
		{
			name: "custom name before ID", kind: "custom_tool_call",
			expectedInput: "*** Begin Patch\n+hello\n*** End Patch",
			calls: []string{
				`{"index":0,"type":"custom","custom":{"name":"apply_patch","input":"*** Begin Patch\n"}}`,
				`{"index":0,"custom":{"input":"+hello\n"}}`,
				`{"index":0,"id":"call_1","custom":{"input":"*** End Patch"}}`,
			},
		},
		{
			name: "function arguments before name", kind: "function_call",
			expectedInput: `{"a":1}`,
			calls: []string{
				`{"index":0,"type":"function","function":{"arguments":"{\"a\":"}}`,
				`{"index":0,"id":"call_1","function":{"arguments":"1"}}`,
				`{"index":0,"function":{"name":"lookup","arguments":"}"}}`,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model"})
			if err != nil {
				t.Fatal(err)
			}
			var provider strings.Builder
			for _, call := range test.calls {
				fmt.Fprintf(&provider, `data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[%s]},"finish_reason":null}]}`+"\n\n", call)
			}
			provider.WriteString(`data: {"id":"response_1","model":"source-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n")
			provider.WriteString(`data: {"id":"response_1","model":"source-model","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n")
			provider.WriteString("data: [DONE]\n\n")
			frames, _, _, err := stream.Push([]byte(provider.String()))
			if err != nil {
				t.Fatal(err)
			}
			final, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			frames = append(frames, final...)
			var added, done, completed int
			var streamedInput strings.Builder
			for _, frame := range frames {
				parsed, parseErr := parseSSEFrame(frame, 1<<20)
				if parseErr != nil {
					t.Fatal(parseErr)
				}
				var wire struct {
					Type      string `json:"type"`
					Delta     string `json:"delta"`
					Input     string `json:"input"`
					Arguments string `json:"arguments"`
					Item      struct {
						Type      string `json:"type"`
						CallID    string `json:"call_id"`
						Name      string `json:"name"`
						Input     string `json:"input"`
						Arguments string `json:"arguments"`
					} `json:"item"`
				}
				if decodeErr := json.Unmarshal(parsed.Data, &wire); decodeErr != nil {
					t.Fatal(decodeErr)
				}
				switch wire.Type {
				case "response.output_item.added":
					added++
					if wire.Item.Type != test.kind || wire.Item.CallID != "call_1" || wire.Item.Name == "" ||
						wire.Item.Input != "" || wire.Item.Arguments != "" {
						t.Fatalf("invalid added item: %s", parsed.Data)
					}
				case "response.custom_tool_call_input.delta", "response.function_call_arguments.delta":
					if added != 1 || wire.Delta == "" {
						t.Fatalf("tool input emitted before identified item: %s", parsed.Data)
					}
					streamedInput.WriteString(wire.Delta)
				case "response.custom_tool_call_input.done", "response.function_call_arguments.done":
					done++
					input := wire.Input
					if test.kind == "function_call" {
						input = wire.Arguments
					}
					if input != test.expectedInput || streamedInput.String() != input {
						t.Fatalf("streamed input changed: deltas=%q want=%q event=%s", streamedInput.String(), test.expectedInput, parsed.Data)
					}
				case "response.output_item.done":
					completed++
					input := wire.Item.Input
					if test.kind == "function_call" {
						input = wire.Item.Arguments
					}
					if wire.Item.Type != test.kind || input != test.expectedInput || done != 1 {
						t.Fatalf("completed item changed tool input: %s", parsed.Data)
					}
				}
			}
			if added != 1 || done != 1 || completed != 1 {
				t.Fatalf("tool item lifecycle incomplete: added=%d done=%d completed=%d", added, done, completed)
			}
		})
	}
}

func TestResponsesNamedCustomToolChoiceSurvivesRouting(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
		var body []byte
		if source == llmprotocol.OpenAIResponsesV1 {
			body = []byte(`{"model":"m","input":"hello","tools":[{"type":"custom","name":"apply_patch","format":{"type":"text"}}],"tool_choice":{"type":"custom","name":"apply_patch"}}`)
		} else {
			body = []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"custom","custom":{"name":"apply_patch","format":{"type":"text"}}}],"tool_choice":{"type":"custom","custom":{"name":"apply_patch"}}}`)
		}
		request, envelope, _, err := engine.DecodeRequestForMutation(source, body)
		if err != nil || request.ToolChoice.Mode != llmprotocol.ToolChoiceNamed ||
			request.ToolChoice.Kind != llmprotocol.ToolKindCustom || request.ToolChoice.Name != "apply_patch" {
			t.Fatalf("%s custom choice decode: %+v, %v", source, request.ToolChoice, err)
		}
		request.Model, request.Generation = "routed-model", request.Generation+1
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
			encoded, err := engine.EncodeRequest(target, request, envelope)
			if err != nil {
				t.Fatalf("%s -> %s custom choice: %s, %v", source, target, encoded.Body, err)
			}
			var wire struct {
				ToolChoice struct {
					Type string `json:"type"`
				} `json:"tool_choice"`
			}
			if err := json.Unmarshal(encoded.Body, &wire); err != nil || wire.ToolChoice.Type != "custom" {
				t.Fatalf("%s -> %s changed custom choice: %s, %v", source, target, encoded.Body, err)
			}
			if _, _, _, err := engine.DecodeRequest(target, encoded.Body); err != nil {
				t.Fatalf("%s -> %s re-decode: %v", source, target, err)
			}
		}
	}
}

func TestResponsesTextVerbositySurvivesRouting(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, verbosity := range []string{"low", "medium", "high"} {
		body := []byte(`{"model":"m","input":"hello","text":{"verbosity":"` + verbosity + `"}}`)
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
		if err != nil || request.TextVerbosity != verbosity {
			t.Fatalf("decode %s: %v %+v", verbosity, err, request)
		}
		request.Model, request.Generation = "routed-model", request.Generation+1
		for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
			encoded, encodeErr := engine.EncodeRequest(format, request, envelope)
			if encodeErr != nil || !bytes.Contains(encoded.Body, []byte(`"verbosity":"`+verbosity+`"`)) {
				t.Fatalf("%s to %s: %v %s", verbosity, format, encodeErr, encoded.Body)
			}
		}
		encoded, encodeErr := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
		if encodeErr != nil || encoded.Request.TextVerbosity != "" ||
			bytes.Contains(encoded.Body, []byte(`"verbosity"`)) || request.TextVerbosity != verbosity {
			t.Fatalf("Messages projection of %s: %v, request=%+v, body=%s", verbosity, encodeErr, encoded.Request, encoded.Body)
		}
		if !hasDroppedVerbosityDiagnostic(encoded.Diagnostics, "text.verbosity") {
			t.Fatalf("Messages omitted %s without a warning: %+v", verbosity, encoded.Diagnostics)
		}
	}
	for _, raw := range []string{`"verbose"`, `7`, `true`} {
		body := []byte(`{"model":"m","input":"hello","text":{"verbosity":` + raw + `}}`)
		_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
		assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_text_verbosity")
	}
	chat := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"verbosity":"medium"}`)
	translated, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, chat, func(request *llmprotocol.Request) error {
		request.Model = "routed-model"
		return nil
	})
	if err != nil || !bytes.Contains(translated.Body, []byte(`"verbosity":"medium"`)) {
		t.Fatalf("Chat verbosity lost on Responses backend: %s, %v", translated.Body, err)
	}
	translated, err = engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, chat, nil)
	if err != nil || bytes.Contains(translated.Body, []byte(`"verbosity"`)) ||
		!hasDroppedVerbosityDiagnostic(translated.Diagnostics, "verbosity") {
		t.Fatalf("Chat verbosity to Messages: %v, body=%s, diagnostics=%+v", err, translated.Body, translated.Diagnostics)
	}
}

func hasDroppedVerbosityDiagnostic(diagnostics llmprotocol.Diagnostics, field string) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field && diagnostic.Action == llmprotocol.DiagnosticDropped &&
			diagnostic.Target == llmprotocol.AnthropicMessagesV1 {
			return true
		}
	}
	return false
}

func TestResponsesCustomToolVariantsRejectMissingAndForeignFields(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, item := range []string{
		`{"type":"custom_tool_call","call_id":"call_1","name":"apply_patch"}`,
		`{"type":"custom_tool_call_output","call_id":"call_1"}`,
		`{"type":"custom_tool_call","call_id":"call_1","name":"apply_patch","input":"patch","arguments":"{}"}`,
	} {
		body := []byte(`{"model":"m","input":[` + item + `]}`)
		_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
		assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_input_item_variant")
	}
}
