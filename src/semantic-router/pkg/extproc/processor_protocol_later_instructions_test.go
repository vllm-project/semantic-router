package extproc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A Chat or Responses client's later system or developer message reaches a
// Chat or Responses backend where the client put it, as a Messages client's
// does. When the operator lists messages[].role it arrives as user text in the
// same place, with one approximated diagnostic. A leading one still leads.
func TestLaterInstructionsFromChatAndResponsesClients(t *testing.T) {
	const turns = `{"role":"system","content":"LEAD"},{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},` +
		`{"role":"developer","content":"LATER"},{"role":"user","content":"again"}`
	clients := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:      `{"model":"m","messages":[` + turns + `]}`,
		llmprotocol.OpenAIResponsesV1: `{"model":"m","input":[` + turns + `]}`,
	}
	for client, body := range clients {
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
			for _, optedIn := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/opt_in=%t", client, target, optedIn), func(t *testing.T) {
					router, model := routingTestRouterForFormat(target)
					want := []string{"system", "user", "assistant", "developer", "user"}
					if optedIn {
						router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldSystemMessageRole}
						want = []string{"system", "user", "assistant", "user", "user"}
					}
					request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(client, []byte(body))
					if err != nil {
						t.Fatal(err)
					}
					ctx := routingTestContext(client, &request)
					ctx.ProtocolEnvelope = envelope
					if _, err = router.prepareProviderDispatch(&request, model, "", false, ctx); err != nil {
						t.Fatal(err)
					}
					encoded, err := router.encodeDispatchRequest(ctx)
					if err != nil {
						t.Fatal(err)
					}
					roles, items := dispatchedRoles(t, target, encoded)
					if !reflect.DeepEqual(roles, want) || !bytes.Contains(items[0]["content"], []byte("LEAD")) ||
						!bytes.Contains(items[3]["content"], []byte("LATER")) {
						t.Fatalf("backend conversation = %v: %s", roles, encoded)
					}
					reported := len(ctx.ProtocolDiagnostics) == 1 &&
						ctx.ProtocolDiagnostics[0].Field == llmprotocol.LossyFieldSystemMessageRole &&
						ctx.ProtocolDiagnostics[0].Source == client
					if reported != optedIn || (!optedIn && len(ctx.ProtocolDiagnostics) != 0) {
						t.Fatalf("diagnostics = %+v, want one messages[].role entry only when opted in", ctx.ProtocolDiagnostics)
					}
				})
			}
		}
	}
}

func dispatchedRoles(t *testing.T, target llmprotocol.WireFormat, body []byte) ([]string, []map[string]json.RawMessage) {
	t.Helper()
	key := "messages"
	if target == llmprotocol.OpenAIResponsesV1 {
		key = "input"
	}
	var wire map[string]json.RawMessage
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(wire[key], &items); err != nil {
		t.Fatal(err)
	}
	roles := make([]string, 0, len(items))
	for _, item := range items {
		var role string
		_ = json.Unmarshal(item["role"], &role)
		roles = append(roles, role)
	}
	return roles, items
}

// Placement warnings survive the role projection and are emitted once at dispatch.
func TestToolSpanInstructionDiagnosticsWithRoleProjection(t *testing.T) {
	for source, body := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"model":"m","messages":[{"role":"user","content":"lookup"},
			{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
			{"role":"developer","content":"Be brief"},{"role":"tool","tool_call_id":"a","content":"found"}]}`,
		llmprotocol.OpenAIResponsesV1: `{"model":"m","input":[{"role":"user","content":"lookup"},
			{"type":"function_call","call_id":"a","name":"lookup","arguments":"{}"},
			{"role":"developer","content":"Be brief"},{"type":"function_call_output","call_id":"a","output":"found"}]}`,
	} {
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
			for _, optedIn := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/opt_in=%t", source, target, optedIn), func(t *testing.T) {
					router, model := routingTestRouterForFormat(target)
					if optedIn {
						router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldSystemMessageRole}
					}
					request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(source, []byte(body))
					if err != nil {
						t.Fatal(err)
					}
					ctx := routingTestContext(source, &request)
					ctx.ProtocolEnvelope = envelope
					if _, err = router.prepareProviderDispatch(&request, model, "", false, ctx); err != nil {
						t.Fatal(err)
					}
					encoded, err := router.encodeDispatchRequest(ctx)
					if err != nil {
						t.Fatal(err)
					}
					field := "messages"
					if source == llmprotocol.OpenAIResponsesV1 {
						field = "input"
					}
					counts := map[string]int{}
					for _, d := range ctx.ProtocolDiagnostics {
						if d.Action == llmprotocol.DiagnosticApproximated {
							counts[d.Field]++
						}
					}
					wantMoved, wantRole := 1, 0
					if target == llmprotocol.OpenAIResponsesV1 && !optedIn {
						wantMoved = 0
					}
					if optedIn && target != llmprotocol.AnthropicMessagesV1 {
						wantRole = 1
					}
					if counts[field] != wantMoved || counts[llmprotocol.LossyFieldSystemMessageRole] != wantRole {
						t.Fatalf("diagnostics = %+v, want placement=%d role=%d", ctx.ProtocolDiagnostics, wantMoved, wantRole)
					}
					if moved := bytes.Index(encoded, []byte("found")) < bytes.Index(encoded, []byte("Be brief")); moved != (wantMoved == 1) {
						t.Fatalf("unexpected backend order: %s", encoded)
					}
				})
			}
		}
	}
}

func TestMessagesDispatchRejectsOnlyEmptyInstructions(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1,
		[]byte(`{"model":"m","max_tokens":16,"messages":[{"role":"system","content":""}]}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	ctx.ProtocolEnvelope = envelope
	_, err = router.prepareProviderDispatch(&request, model, "", false, ctx)
	if err == nil {
		_, err = router.encodeDispatchRequest(ctx)
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "anthropic_messages_required" {
		t.Fatalf("empty backend message list was not refused: %v", err)
	}
}
