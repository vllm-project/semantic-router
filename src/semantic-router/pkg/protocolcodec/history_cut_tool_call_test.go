package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const cutHistoryArguments = `{"to": "team@example.com", "body": "Dear team, the review moves to ten.`

const sendEmailParameters = `{"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}`

// historyWithCutCall is a client's next request after the output limit cut a
// send_email call, kept in its history as the Router passed it on. Each
// fixture takes the cut call's arguments, JSON-encoded, and what follows the
// call in its turn.
var historyWithCutCall = map[llmprotocol.WireFormat]string{
	llmprotocol.OpenAIChatV1: `{"model":"m","messages":[` +
		`{"role":"user","content":"Tell the team the review moves to ten."},` +
		`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"send_email","arguments":%s}}%s]},` +
		`{"role":"tool","tool_call_id":"call_1","content":"The email was cut off; nothing was sent."},` +
		`{"role":"user","content":"Send a shorter one."}],` +
		`"tools":[{"type":"function","function":{"name":"send_email","parameters":` + sendEmailParameters + `}}]}`,
	llmprotocol.OpenAIResponsesV1: `{"model":"m","store":false,"input":[` +
		`{"type":"message","role":"user","content":"Tell the team the review moves to ten."},` +
		`{"type":"function_call","call_id":"call_1","name":"send_email","arguments":%s}%s,` +
		`{"type":"function_call_output","call_id":"call_1","output":"The email was cut off; nothing was sent."},` +
		`{"type":"message","role":"user","content":"Send a shorter one."}],` +
		`"tools":[{"type":"function","name":"send_email","parameters":` + sendEmailParameters + `}]}`,
}

// laterCallInTurn follows the cut call in its turn with a second call.
var laterCallInTurn = map[llmprotocol.WireFormat]string{
	llmprotocol.OpenAIChatV1:      `,{"id":"call_2","type":"function","function":{"name":"send_email","arguments":"{}"}}`,
	llmprotocol.OpenAIResponsesV1: `,{"type":"function_call","call_id":"call_2","name":"send_email","arguments":"{}"}`,
}

func historyBody(client llmprotocol.WireFormat, arguments, later string) []byte {
	encoded, _ := json.Marshal(arguments)
	return []byte(fmt.Sprintf(historyWithCutCall[client], encoded, later))
}

func requestCallArguments(request llmprotocol.Request) []string {
	var arguments []string
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				arguments = append(arguments, content.ToolCall.Arguments)
			}
		}
	}
	return arguments
}

func compactJSON(t *testing.T, value string) string {
	t.Helper()
	var compacted bytes.Buffer
	if err := json.Compact(&compacted, []byte(value)); err != nil {
		t.Fatalf("%q is not JSON: %v", value, err)
	}
	return compacted.String()
}

// A Chat or Responses client may keep the call an output limit cut, as the
// Router passed it on, in the history it sends next. As the last call of its
// assistant turn it is accepted, every backend receives the members completed
// before the cut, and the closure is reported as approximated. A same-format
// backend gets the request encoded again, since the client's bytes carry the
// cut arguments.
func TestHistoryCallCutAtTheEndOfItsTurnReachesEveryBackendClosed(t *testing.T) {
	engine := NewBuiltinEngine()
	for client := range historyWithCutCall {
		for _, backend := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
			t.Run(string(client)+"_to_"+string(backend), func(t *testing.T) {
				result, err := engine.TranslateRequest(client, backend, historyBody(client, cutHistoryArguments, ""), nil)
				if err != nil {
					t.Fatalf("translate: %v", err)
				}
				if got := requestCallArguments(result.Request); !slices.Equal(got, []string{`{"to": "team@example.com"}`}) {
					t.Fatalf("decoded call arguments = %q", got)
				}
				if !slices.ContainsFunc(result.Diagnostics, func(diagnostic llmprotocol.Diagnostic) bool {
					return diagnostic.Field == "tool_call.arguments" && diagnostic.Action == llmprotocol.DiagnosticApproximated
				}) {
					t.Fatalf("no approximated tool_call.arguments diagnostic: %+v", result.Diagnostics)
				}
				if strings.Contains(string(result.Body), "Dear team") {
					t.Fatalf("the backend body carries the cut arguments:\n%s", result.Body)
				}
				sent, _, _, err := engine.DecodeRequest(backend, result.Body)
				if err != nil {
					t.Fatalf("decode the backend body: %v\n%s", err, result.Body)
				}
				arguments := requestCallArguments(sent)
				if len(arguments) != 1 || compactJSON(t, arguments[0]) != `{"to":"team@example.com"}` {
					t.Fatalf("the backend received call arguments %q:\n%s", arguments, result.Body)
				}
			})
		}
	}
}

// Malformed arguments that no output limit explains are still refused: a cut
// call that another call of its turn follows, and arguments that don't start
// a JSON object.
func TestHistoryCallMalformedElsewhereIsStillRefused(t *testing.T) {
	engine := NewBuiltinEngine()
	for client := range historyWithCutCall {
		for name, body := range map[string][]byte{
			"followed by another call": historyBody(client, cutHistoryArguments, laterCallInTurn[client]),
			"not an object":            historyBody(client, `["team@example.com"`, ""),
		} {
			t.Run(string(client)+"/"+name, func(t *testing.T) {
				_, _, _, err := engine.DecodeRequest(client, body)
				requireProtocolErrorCode(t, err, "invalid_tool_call")
			})
		}
	}
}
