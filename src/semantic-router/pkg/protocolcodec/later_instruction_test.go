package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func routeRequest(request *llmprotocol.Request) error {
	request.Model = "routed-model"
	return nil
}

// A system or developer message after the conversation starts stays where the
// client put it, whichever protocol the client spoke, as a Messages client's
// does (TestOfficialAnthropicSystemMessagePreservesItsPositionAcrossTheMatrix).
// One before the conversation still leads the request, and a Messages backend
// receives a later developer message as a system message in its place.
func TestLaterInstructionKeepsItsPositionAcrossTheMatrix(t *testing.T) {
	const turns = `{"role":"system","content":"LEAD"},{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},` +
		`{"role":"developer","content":"LATER"},{"role":"user","content":"again"}`
	sources := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:      `{"model":"m","max_tokens":16,"messages":[` + turns + `]}`,
		llmprotocol.OpenAIResponsesV1: `{"model":"m","max_output_tokens":16,"input":[` + turns + `]}`,
	}
	want := map[llmprotocol.WireFormat][]string{
		llmprotocol.OpenAIChatV1:        {"system", "user", "assistant", "developer", "user"},
		llmprotocol.OpenAIResponsesV1:   {"system", "user", "assistant", "developer", "user"},
		llmprotocol.AnthropicMessagesV1: {"user", "assistant", "system", "user"},
	}
	engine := NewBuiltinEngine()
	for source, body := range sources {
		for _, target := range builtinFormats {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
				translated, err := engine.TranslateRequest(source, target, []byte(body), routeRequest)
				if err != nil {
					t.Fatal(err)
				}
				if roles := requestWireMessageRoles(t, target, translated.Body); !reflect.DeepEqual(roles, want[target]) {
					t.Fatalf("roles = %v: %s", roles, translated.Body)
				}
				lead, later := bytes.Index(translated.Body, []byte("LEAD")), bytes.Index(translated.Body, []byte("LATER"))
				if lead < 0 || lead > bytes.Index(translated.Body, []byte("hi")) ||
					later < bytes.Index(translated.Body, []byte("yo")) || later > bytes.Index(translated.Body, []byte("again")) {
					t.Fatalf("instruction order changed: %s", translated.Body)
				}
				if len(translated.Diagnostics) != 0 {
					t.Fatalf("diagnostics = %+v", translated.Diagnostics)
				}
			})
		}
	}
}

// Both OpenAI clients can place an instruction between a call and its output.
// Chat and Messages backends receive the output first, including native replay.
func TestInstructionBetweenCallAndOutputMovesAfterTheOutput(t *testing.T) {
	for _, role := range []string{"system", "developer"} {
		for source, body := range map[llmprotocol.WireFormat]string{
			llmprotocol.OpenAIChatV1: `{"model":"m","max_tokens":16,"messages":[
				{"role":"user","content":"look it up"},
				{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
				{"role":"` + role + `","content":"Keep answers brief"},
				{"role":"tool","tool_call_id":"a","content":"found"}]}`,
			llmprotocol.OpenAIResponsesV1: `{"model":"m","max_output_tokens":16,"input":[
				{"role":"user","content":"look it up"},
				{"type":"function_call","call_id":"a","name":"lookup","arguments":"{}"},
				{"role":"` + role + `","content":"Keep answers brief"},
				{"type":"function_call_output","call_id":"a","output":"found"}]}`,
		} {
			for _, target := range builtinFormats {
				for _, mutate := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/%s/mutate=%t", source, target, role, mutate), func(t *testing.T) {
						var mutation func(*llmprotocol.Request) error
						if mutate {
							mutation = routeRequest
						}
						translated, err := NewBuiltinEngine().TranslateRequest(source, target, []byte(body), mutation)
						if err != nil {
							t.Fatal(err)
						}
						want := map[llmprotocol.WireFormat][]string{
							llmprotocol.OpenAIChatV1:        {"user", "assistant", "tool", role},
							llmprotocol.AnthropicMessagesV1: {"user", "assistant", "user", "system"},
							llmprotocol.OpenAIResponsesV1:   {"user", "function_call", role, "function_call_output"},
						}[target]
						if order := requestWireItemLabels(t, target, translated.Body); !reflect.DeepEqual(order, want) {
							t.Fatalf("order = %v, want %v: %s", order, want, translated.Body)
						}
						field := "messages"
						if source == llmprotocol.OpenAIResponsesV1 {
							field = "input"
						}
						if moved := hasDiagnostic(translated.Diagnostics, field, llmprotocol.DiagnosticApproximated); moved != (target != llmprotocol.OpenAIResponsesV1) {
							t.Fatalf("diagnostics = %+v", translated.Diagnostics)
						}
					})
				}
			}
		}
	}
}

func TestEmptyLaterInstructionsToAnthropic(t *testing.T) {
	for _, role := range []string{"system", "developer"} {
		for source, field := range map[llmprotocol.WireFormat]string{
			llmprotocol.OpenAIChatV1: "messages", llmprotocol.OpenAIResponsesV1: "input",
		} {
			t.Run(string(source)+"/"+role, func(t *testing.T) {
				body := `{"model":"m","` + field + `":[{"role":"user","content":"hi"},{"role":"` + role + `","content":""}]}`
				translated, err := NewBuiltinEngine().TranslateRequest(source, llmprotocol.AnthropicMessagesV1, []byte(body), nil)
				if err != nil {
					t.Fatal(err)
				}
				if roles := requestWireMessageRoles(t, llmprotocol.AnthropicMessagesV1, translated.Body); !reflect.DeepEqual(roles, []string{"user"}) {
					t.Fatalf("empty instruction introduced a turn: %s", translated.Body)
				}
			})
		}
	}
}

// requestWireItemLabels lists each message's role, or a Responses item's type
// when it is not a message.
func requestWireItemLabels(t *testing.T, format llmprotocol.WireFormat, body []byte) []string {
	t.Helper()
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	field := "messages"
	if format == llmprotocol.OpenAIResponsesV1 {
		field = "input"
	}
	var items []struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(wire[field], &items); err != nil {
		t.Fatal(err)
	}
	labels := make([]string, 0, len(items))
	for _, item := range items {
		if item.Role != "" {
			labels = append(labels, item.Role)
		} else {
			labels = append(labels, item.Type)
		}
	}
	return labels
}
