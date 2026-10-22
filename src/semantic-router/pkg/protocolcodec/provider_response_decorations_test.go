package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicResponseDiagnosticsDecodeAcrossClientFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	base := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":1},"diagnostics":null}`)
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			clean, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, base, nil)
			if err != nil || !strings.Contains(string(clean.Body), "done") {
				t.Fatalf("null diagnostics prevented translation: body=%s err=%v", clean.Body, err)
			}
			if hasDiagnostic(clean.Diagnostics, "diagnostics", llmprotocol.DiagnosticDropped) {
				t.Fatalf("null diagnostics reported a loss: %+v", clean.Diagnostics)
			}
			withDiagnostics := bytes.Replace(base, []byte(`"diagnostics":null`),
				[]byte(`"diagnostics":{"cache_miss_reason":{"type":"tools_changed","cache_missed_input_tokens":4}}`), 1)
			translated, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, withDiagnostics, nil)
			if err != nil || !strings.Contains(string(translated.Body), "done") ||
				!hasDiagnostic(translated.Diagnostics, "diagnostics", llmprotocol.DiagnosticDropped) {
				t.Fatalf("cache diagnostics prevented translation or lacked omission: body=%s diagnostics=%+v err=%v",
					translated.Body, translated.Diagnostics, err)
			}
		})
	}
}

func TestAnthropicMessageStartAcceptsDiagnostics(t *testing.T) {
	for _, diagnostic := range []struct {
		name    string
		value   string
		dropped bool
	}{
		{name: "null", value: "null"},
		{name: "cache_miss", value: `{"cache_miss_reason":{"type":"tools_changed","cache_missed_input_tokens":4}}`, dropped: true},
	} {
		t.Run(diagnostic.name, func(t *testing.T) {
			decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
			frame := `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0},"diagnostics":` + diagnostic.value + `}}

`
			events, diagnostics, err := decoder.Push([]byte(frame))
			if err != nil || len(events) != 1 || events[0].Type != llmprotocol.EventResponseStarted {
				t.Fatalf("message_start failed: events=%+v diagnostics=%+v err=%v", events, diagnostics, err)
			}
			if got := hasDiagnostic(diagnostics, "stream.message.diagnostics", llmprotocol.DiagnosticDropped); got != diagnostic.dropped {
				t.Fatalf("diagnostic present=%t want=%t: %+v", got, diagnostic.dropped, diagnostics)
			}
		})
	}
}

func TestResponsesProviderDecorationsDecodeAcrossClientFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"resp_1","object":"response","created_at":1,"model":"gpt-5-nano","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3},"access_programs":null,"billing":{"payer":"openai"},"frequency_penalty":0.0,"presence_penalty":0.0,"tool_usage":{"web_search":{"num_requests":0}}}`)
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			translated, err := engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, target, body, nil)
			if err != nil || !strings.Contains(string(translated.Body), "done") {
				t.Fatalf("decorated Responses reply failed translation: body=%s err=%v", translated.Body, err)
			}
			for _, field := range []string{"billing", "frequency_penalty", "presence_penalty", "tool_usage"} {
				if !hasDiagnostic(translated.Diagnostics, field, llmprotocol.DiagnosticDropped) {
					t.Errorf("missing %s omission: %+v", field, translated.Diagnostics)
				}
			}
			if hasDiagnostic(translated.Diagnostics, "access_programs", llmprotocol.DiagnosticDropped) {
				t.Fatalf("null access_programs reported a loss: %+v", translated.Diagnostics)
			}
		})
	}
	access := bytes.Replace(body, []byte(`"access_programs":null`), []byte(`"access_programs":{"cyber":"standard"}`), 1)
	_, _, diagnostics, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, access)
	if err != nil || !hasDiagnostic(diagnostics, "access_programs", llmprotocol.DiagnosticDropped) {
		t.Fatalf("non-null access_programs omitted silently: diagnostics=%+v err=%v", diagnostics, err)
	}
}

func TestResponsesStreamResourceAcceptsProviderDecorations(t *testing.T) {
	decoder := OpenAIResponsesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frame := []byte("event: response.created\ndata: " +
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"model":"gpt-5-nano","status":"in_progress","output":[],"access_programs":null,"billing":{"payer":"openai"},"frequency_penalty":0.0,"presence_penalty":0.0,"tool_usage":{"web_search":{"num_requests":0}}}}` + "\n\n")
	events, diagnostics, err := decoder.Push(frame)
	if err != nil || len(events) != 1 || events[0].Type != llmprotocol.EventResponseStarted {
		t.Fatalf("decorated Responses stream start failed: events=%+v diagnostics=%+v err=%v", events, diagnostics, err)
	}
	for _, field := range []string{"billing", "frequency_penalty", "presence_penalty", "tool_usage"} {
		if !hasDiagnostic(diagnostics, "stream.response."+field, llmprotocol.DiagnosticDropped) {
			t.Errorf("missing streamed %s omission: %+v", field, diagnostics)
		}
	}
	if hasDiagnostic(diagnostics, "stream.response.access_programs", llmprotocol.DiagnosticDropped) {
		t.Fatalf("null access_programs reported a loss: %+v", diagnostics)
	}
	repeat := []byte("event: response.in_progress\ndata: " +
		`{"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_1","object":"response","created_at":1,"model":"gpt-5-nano","status":"in_progress","output":[],"billing":{"payer":"openai"},"frequency_penalty":0.0,"presence_penalty":0.0,"tool_usage":{"web_search":{"num_requests":0}}}}` + "\n\n")
	_, repeatedDiagnostics, err := decoder.Push(repeat)
	if err != nil {
		t.Fatalf("repeated response resource failed: %v", err)
	}
	for _, field := range []string{"billing", "frequency_penalty", "presence_penalty", "tool_usage"} {
		if hasDiagnostic(repeatedDiagnostics, "stream.response."+field, llmprotocol.DiagnosticDropped) {
			t.Errorf("repeated streamed %s omission consumed a second diagnostic: %+v", field, repeatedDiagnostics)
		}
	}
}
