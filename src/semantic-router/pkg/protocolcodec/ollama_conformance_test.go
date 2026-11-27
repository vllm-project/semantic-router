package protocolcodec

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Ollama live conformance receipt, observed 2026-11-21 on a local Ollama
// 0.40.17 (the macOS release build, Metal) serving qwen3:0.6b on its
// OpenAI-compatible surface without authentication: POST /v1/chat/completions
// and GET /v1/models. Each testdata/providers/ollama-0.40.17-* file is one
// reply byte for byte, and its size is pinned here so an edited capture
// cannot pass as conformance.
//
// Every chat request sends "model":"qwen3:0.6b","seed":7,"temperature":0 and:
//   - text: "Say hello in one word." with "reasoning_effort":"none";
//   - reasoning: "What is 2+3? Answer with just the number.";
//   - tool-call: "What is the weather in Paris? Use the tool." with one
//     function tool, get_weather, that takes a required string "city";
//   - stop-string: "Write one sentence about the sea." with
//     "reasoning_effort":"none" and "stop":["."];
//   - length: "Count from one to twenty in words." with
//     "reasoning_effort":"none" and "max_tokens":4.
//
// A -stream capture adds "stream":true and
// "stream_options":{"include_usage":true}; text-no-usage-stream leaves the
// option out. unknown-model-error asks for the model "no-such-model" (HTTP
// 404), and malformed-request-error sends "messages":"hi" (HTTP 400).
//
// Ollama parses a tool call out of each model's own output format, so the two
// chat-llama3.2-tool-call captures hold a second model family to the same
// checks. They repeat the tool-call request with "model":"llama3.2:1b" on the
// same server. That model sends no reasoning.
//
// Ollama names reasoning "reasoning", sends "content":"" beside every
// reasoning delta and beside tool calls, numbers buffered tool calls with
// "index", and adds a "timings" object to the usage chunk that ends a stream.
// It does not say which stop string ended a generation, so that reply ends as
// an ordinary stop.
const ollamaCapturePrefix = "ollama-0.40.17-"

var ollamaCaptureBytes = map[string]int{
	"chat-text-out.json":                    335,
	"chat-text-stream.sse":                  1325,
	"chat-text-no-usage-stream.sse":         850,
	"chat-reasoning-out.json":               688,
	"chat-reasoning-stream.sse":             23051,
	"chat-tool-call-out.json":               889,
	"chat-tool-call-stream.sse":             21276,
	"chat-llama3.2-tool-call-out.json":      474,
	"chat-llama3.2-tool-call-stream.sse":    1075,
	"chat-stop-string-out.json":             407,
	"chat-stop-string-stream.sse":           4269,
	"chat-length-out.json":                  339,
	"chat-length-stream.sse":                1534,
	"chat-unknown-model-error-out.json":     106,
	"chat-malformed-request-error-out.json": 186,
	"models-out.json":                       106,
}

const (
	ollamaServedModel  = "qwen3:0.6b"
	ollamaSecondModel  = "llama3.2:1b"
	ollamaSeaSentence  = "The sea is a vast, ever-changing body of water that shapes the world around us"
	ollamaToolCallArgs = `{"city":"Paris"}`
	ollamaTimingsField = "stream.timings"
)

func ollamaCapture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "providers", ollamaCapturePrefix+name))
	if err != nil {
		t.Fatalf("no provider conformance capture %s: %v", name, err)
	}
	if want, pinned := ollamaCaptureBytes[name]; !pinned || len(body) != want {
		t.Fatalf("%s = %d bytes, want the observed %d", name, len(body), want)
	}
	return body
}

func ollamaPolicies() []llmprotocol.Policy {
	return []llmprotocol.Policy{llmprotocol.DefaultPolicy(), conformancePolicy()}
}

func ollamaDiagnosticFields(diagnostics llmprotocol.Diagnostics) []string {
	var fields []string
	for _, diagnostic := range diagnostics {
		fields = append(fields, diagnostic.Field)
	}
	return fields
}

// ollamaInOrder reports the first of wants that out does not carry after the
// ones before it.
func ollamaInOrder(out string, wants []string) string {
	for _, want := range wants {
		index := strings.Index(out, want)
		if index < 0 {
			return want
		}
		out = out[index+len(want):]
	}
	return ""
}

// ollamaReplyModel is the model a reply reports: the served one, unless the
// capture came from another.
func ollamaReplyModel(other string) string {
	if other != "" {
		return other
	}
	return ollamaServedModel
}

var ollamaReplies = []struct {
	name        string
	model       string
	stop        llmprotocol.StopReason
	kinds       string
	text        string
	diagnostics []string
}{
	{name: "chat-text-out.json", stop: llmprotocol.StopEndTurn, kinds: "text", text: `"Hi!"`},
	{name: "chat-reasoning-out.json", stop: llmprotocol.StopEndTurn, kinds: "reasoning,text", text: "5"},
	{
		name: "chat-tool-call-out.json", stop: llmprotocol.StopToolCall, kinds: "reasoning,tool_call",
		diagnostics: []string{"choices.message.tool_calls.index"},
	},
	{
		name: "chat-llama3.2-tool-call-out.json", model: ollamaSecondModel,
		stop: llmprotocol.StopToolCall, kinds: "tool_call",
		diagnostics: []string{"choices.message.tool_calls.index"},
	},
	{name: "chat-stop-string-out.json", stop: llmprotocol.StopEndTurn, kinds: "text", text: ollamaSeaSentence},
	{name: "chat-length-out.json", stop: llmprotocol.StopMaxTokens, kinds: "text", text: "One, two,"},
}

// Every member of these replies is declared, so the conformance policy, which
// rejects undeclared members, decodes them too. The empty content beside a
// tool call is not an answer, and the only member left out of the neutral
// response is the tool call's index.
func TestOllamaRepliesDecode(t *testing.T) {
	for _, reply := range ollamaReplies {
		t.Run(reply.name, func(t *testing.T) {
			body := ollamaCapture(t, reply.name)
			for _, policy := range ollamaPolicies() {
				response, _, diagnostics, err := OpenAIChatCodec{}.DecodeResponse(body, policy)
				if err != nil {
					t.Fatalf("DecodeResponse (conformance %t): %v", policy.RejectUnknownUpstreamFields, err)
				}
				if got := ollamaDiagnosticFields(diagnostics); !reflect.DeepEqual(got, reply.diagnostics) {
					t.Fatalf("diagnostics = %+v, want only %v", diagnostics, reply.diagnostics)
				}
				if response.Model != ollamaReplyModel(reply.model) || response.StopReason != reply.stop {
					t.Fatalf("model = %q, stop reason = %q, want %q", response.Model, response.StopReason, reply.stop)
				}
				if response.Usage.State != llmprotocol.UsageAvailable || response.Usage.InputCacheRead.Value == nil {
					t.Fatalf("usage = %+v, want token counts with the cached prompt tokens", response.Usage)
				}
				if len(response.Output) != 1 {
					t.Fatalf("output messages = %d, want 1", len(response.Output))
				}
				var kinds []string
				text, arguments := "", ""
				for _, content := range response.Output[0].Content {
					kinds = append(kinds, string(content.Kind))
					switch content.Kind {
					case llmprotocol.ContentText:
						text = content.Text
					case llmprotocol.ContentToolCall:
						arguments = content.ToolCall.Name + content.ToolCall.Arguments
					}
				}
				if strings.Join(kinds, ",") != reply.kinds || text != reply.text {
					t.Fatalf("content kinds = %v, text = %q, want %s and %q", kinds, text, reply.kinds, reply.text)
				}
				if reply.stop == llmprotocol.StopToolCall && arguments != "get_weather"+ollamaToolCallArgs {
					t.Fatalf("tool call = %q", arguments)
				}
			}
		})
	}
}

var ollamaStreams = []struct {
	name      string
	timings   bool
	blocks    int
	chat      []string
	responses []string
	messages  []string
}{
	{
		name: "chat-text-stream.sse", timings: true, blocks: 1,
		chat:      []string{`"delta":{"content":"Hi"}`, `"finish_reason":"stop"`, "data: [DONE]"},
		responses: []string{`"text":"\"Hi!\"","type":"response.output_text.done"`, `"status":"completed"`, `"output_tokens":4`},
		messages:  []string{`"type":"text_delta","text":"Hi"`, `"stop_reason":"end_turn"`, `"output_tokens":4`, "event: message_stop"},
	},
	{
		name: "chat-text-no-usage-stream.sse", blocks: 1,
		chat:      []string{`"delta":{"content":"Hi"}`, `"finish_reason":"stop"`, "data: [DONE]"},
		responses: []string{`"text":"\"Hi!\"","type":"response.output_text.done"`, `"status":"completed"`},
		messages:  []string{`"type":"text_delta","text":"Hi"`, `"stop_reason":"end_turn"`, "event: message_stop"},
	},
	{
		name: "chat-reasoning-stream.sse", timings: true, blocks: 2,
		chat:      []string{`"reasoning_content":"Okay"`, `"delta":{"content":"5"}`, `"finish_reason":"stop"`},
		responses: []string{`"type":"response.reasoning_text.delta"`, `"text":"5","type":"response.output_text.done"`, `"status":"completed"`},
		messages:  []string{`"type":"thinking_delta","thinking":"Okay"`, `"type":"text_delta","text":"5"`, `"stop_reason":"end_turn"`},
	},
	{
		name: "chat-tool-call-stream.sse", timings: true, blocks: 2,
		chat:      []string{`"reasoning_content":"Okay"`, `"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}`, `"finish_reason":"tool_calls"`},
		responses: []string{`"type":"response.reasoning_text.delta"`, `"name":"get_weather"`, `"delta":"{\"city\":\"Paris\"}"`, `"arguments":"{\"city\":\"Paris\"}"`, `"type":"response.function_call_arguments.done"`, `"status":"completed"`},
		messages:  []string{`"type":"thinking_delta","thinking":"Okay"`, `"name":"get_weather","type":"tool_use"`, `"partial_json":"{\"city\":\"Paris\"}"`, `"stop_reason":"tool_use"`},
	},
	{
		name: "chat-llama3.2-tool-call-stream.sse", timings: true, blocks: 1,
		chat:      []string{`"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}`, `"finish_reason":"tool_calls"`},
		responses: []string{`"name":"get_weather"`, `"delta":"{\"city\":\"Paris\"}"`, `"arguments":"{\"city\":\"Paris\"}"`, `"type":"response.function_call_arguments.done"`, `"status":"completed"`},
		messages:  []string{`"name":"get_weather","type":"tool_use"`, `"partial_json":"{\"city\":\"Paris\"}"`, `"stop_reason":"tool_use"`},
	},
	{
		name: "chat-stop-string-stream.sse", timings: true, blocks: 1,
		chat:      []string{`"delta":{"content":"The"}`, `"finish_reason":"stop"`},
		responses: []string{`"text":"` + ollamaSeaSentence + `","type":"response.output_text.done"`, `"status":"completed"`},
		messages:  []string{`"type":"text_delta","text":"The"`, `"stop_reason":"end_turn","stop_sequence":null`},
	},
	{
		name: "chat-length-stream.sse", timings: true, blocks: 1,
		chat:      []string{`"delta":{"content":"One"}`, `"finish_reason":"length"`},
		responses: []string{`"text":"One, two,","type":"response.output_text.done"`, `"incomplete_details":{"reason":"max_output_tokens"}`},
		messages:  []string{`"type":"text_delta","text":"One"`, `"stop_reason":"max_tokens"`},
	},
}

// Each stream reaches a Chat, a Responses and a Messages client under both
// policies, and a tool call arrives at each with its exact arguments. The empty
// content Ollama sends beside reasoning and tool calls opens no block of its
// own, and the timings on the final usage chunk are reported once as left out
// instead of failing the strict decode.
func TestOllamaStreamsTranslateToEveryClient(t *testing.T) {
	for _, stream := range ollamaStreams {
		t.Run(stream.name, func(t *testing.T) {
			body := ollamaCapture(t, stream.name)
			targets := []struct {
				format llmprotocol.WireFormat
				wants  []string
				block  string
			}{
				{llmprotocol.OpenAIChatV1, stream.chat, ""},
				{llmprotocol.OpenAIResponsesV1, stream.responses, "event: response.output_item.added"},
				{llmprotocol.AnthropicMessagesV1, stream.messages, "event: content_block_start"},
			}
			for _, policy := range ollamaPolicies() {
				engine, err := NewEngine(NewBuiltinRegistry(), policy)
				if err != nil {
					t.Fatal(err)
				}
				for _, target := range targets {
					out, diagnostics, err := translateVLLMStream(engine, target.format, body)
					if err != nil {
						t.Fatalf("to %s (conformance %t): %v\n%s", target.format, policy.RejectUnknownUpstreamFields, err, out)
					}
					if missing := ollamaInOrder(out, target.wants); missing != "" {
						t.Fatalf("to %s: no %s in order:\n%s", target.format, missing, out)
					}
					if target.block != "" && strings.Count(out, target.block) != stream.blocks {
						t.Fatalf("to %s: %d output blocks, want %d:\n%s", target.format, strings.Count(out, target.block), stream.blocks, out)
					}
					assertOllamaStreamDiagnostics(t, diagnostics, stream.timings)
				}
			}
		})
	}
}

func assertOllamaStreamDiagnostics(t *testing.T, diagnostics llmprotocol.Diagnostics, timings bool) {
	t.Helper()
	if DiagnosticsDroppedUnknownFields(diagnostics) {
		t.Fatalf("undeclared members dropped: %+v", diagnostics)
	}
	reported := 0
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == ollamaTimingsField && diagnostic.Action == llmprotocol.DiagnosticDropped {
			reported++
		}
	}
	want := 0
	if timings {
		want = 1
	}
	if reported != want {
		t.Fatalf("timings reported %d times, want %d: %+v", reported, want, diagnostics)
	}
}

var ollamaErrors = []struct {
	name     string
	category llmprotocol.ErrorCategory
	message  string
	messages string
}{
	{
		name: "chat-unknown-model-error-out.json", category: llmprotocol.ErrorNotFound,
		message: "model 'no-such-model' not found", messages: `"type":"not_found_error"`,
	},
	{
		name: "chat-malformed-request-error-out.json", category: llmprotocol.ErrorInvalidRequest,
		message:  "json: cannot unmarshal string into Go struct field ChatCompletionRequest.messages of type []openai.Message",
		messages: `"type":"invalid_request_error"`,
	},
}

// Ollama's failures are the canonical OpenAI error object with a null param
// and code, so they need no vendor: the strict transport decode keeps the
// category and the message, and a Messages client receives the same failure.
func TestOllamaFailureEnvelopesDecode(t *testing.T) {
	for _, failure := range ollamaErrors {
		t.Run(failure.name, func(t *testing.T) {
			body := ollamaCapture(t, failure.name)
			for _, policy := range ollamaPolicies() {
				engine, err := NewEngine(NewBuiltinRegistry(), policy)
				if err != nil {
					t.Fatal(err)
				}
				transportError, diagnostics, err := engine.DecodeTransportError(llmprotocol.OpenAIChatV1, body)
				if err != nil || len(diagnostics) != 0 {
					t.Fatalf("DecodeTransportError (conformance %t): %v, diagnostics %+v", policy.RejectUnknownUpstreamFields, err, diagnostics)
				}
				if transportError.Error == nil || transportError.Error.Category != failure.category || transportError.Error.Message != failure.message {
					t.Fatalf("transport error = %+v, want %s: %s", transportError.Error, failure.category, failure.message)
				}
				translated, err := engine.TranslateTransportError(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
				if err != nil || !strings.Contains(string(translated.Body), failure.messages) {
					t.Fatalf("Messages failure = %s, err = %v, want %s", translated.Body, err, failure.messages)
				}
			}
		})
	}
}

// The model list is the closed OpenAI shape, and it names the model under the
// ID that the chat replies report, which is the ID a provider binding sends.
func TestOllamaModelListNamesTheServedModel(t *testing.T) {
	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ollamaCapture(t, "models-out.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&list); err != nil {
		t.Fatalf("model list carries an undeclared member: %v", err)
	}
	if list.Object != "list" || len(list.Data) != 1 {
		t.Fatalf("model list = %+v, want one served model", list)
	}
	if served := list.Data[0]; served.ID != ollamaServedModel || served.Object != "model" || served.Created <= 0 || served.OwnedBy == "" {
		t.Fatalf("served model = %+v", served)
	}
}
