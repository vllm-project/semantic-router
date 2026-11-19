package extproc

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vllmServerMembers are the members vLLM adds about itself, at the top level
// of a reply or chunk and on its choices.
var vllmServerMembers = []string{
	`"system_fingerprint"`, `"prompt_token_ids"`, `"prompt_text"`, `"prompt_logprobs"`, `"kv_transfer_params"`,
	`"ec_transfer_params"`, `"metrics"`, `"token_ids"`, `"routed_experts"`, `"weight_versions"`,
}

// replyThroughRouter sends a captured buffered Chat reply through the response
// body path and returns exactly what the client receives.
func replyThroughRouter(t *testing.T, client llmprotocol.WireFormat, fixture string) []byte {
	t.Helper()
	body, err := os.ReadFile("../protocolcodec/testdata/providers/" + fixture)
	require.NoError(t, err)
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(model, "hello")
	ctx := routingTestContext(client, request)
	_, err = router.prepareProviderDispatch(request, model, "", false, ctx)
	require.NoError(t, err)
	response := router.handleNonStreamingResponseBody(body, ctx, 0, false)
	require.Nil(t, response.GetImmediateResponse(), "reply was refused")
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		return mutation.GetBody()
	}
	return body
}

func requireNoServerMembers(t *testing.T, received []byte) {
	t.Helper()
	for _, member := range vllmServerMembers {
		require.NotContains(t, string(received), member)
	}
}

func TestChatClientReceivesNoServerMembersFromVLLM(t *testing.T) {
	buffered := replyThroughRouter(t, llmprotocol.OpenAIChatV1, "vllm-0.33-chat-return-extensions-out.json")
	requireNoServerMembers(t, buffered)
	require.Contains(t, string(buffered), `"content":"Hello!"`)

	streamed := streamOllamaThroughRouter(t, llmprotocol.OpenAIChatV1, "vllm-0.33-chat-return-extensions-stream.sse", true)
	requireNoServerMembers(t, []byte(streamed))
	require.Contains(t, streamed, `"usage":`)
	require.Equal(t, 1, strings.Count(streamed, "data: [DONE]"))

	// The rest of the reply is vLLM's own, its reasoning member included.
	buffered = replyThroughRouter(t, llmprotocol.OpenAIChatV1, "vllm-0.33-chat-reasoning-out.json")
	requireNoServerMembers(t, buffered)
	require.Contains(t, string(buffered), `"reasoning":"`)
	streamed = streamOllamaThroughRouter(t, llmprotocol.OpenAIChatV1, "vllm-0.33-chat-reasoning-stream.sse", false)
	requireNoServerMembers(t, []byte(streamed))
	require.Contains(t, streamed, `"reasoning":"`)
}

// A Chat client ends a tool loop with tool_choice none; the history still holds
// the call and its result.
func dispatchToolChoiceNone(t *testing.T, backend llmprotocol.WireFormat) (map[string]json.RawMessage, *RequestContext) {
	t.Helper()
	router, model := routingTestRouterForFormat(backend)
	body := []byte(`{"model":"` + model + `","max_tokens":64,"messages":[
		{"role":"user","content":"What time is it?"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"noon"},
		{"role":"user","content":"Now answer without tools."}],
		"tools":[{"type":"function","function":{"name":"get_time","parameters":{"type":"object"}}}],
		"tool_choice":"none","parallel_tool_calls":false}`)
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: "tool-choice-none", SourceFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()}
	request, immediate := router.prepareProtocolRequest(body, ctx)
	require.Nil(t, immediate)
	_, err := router.prepareProviderDispatch(request, model, "", false, ctx)
	require.NoError(t, err)
	dispatched, err := router.encodeDispatchRequest(ctx)
	require.NoError(t, err)
	var object map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(dispatched, &object), string(dispatched))
	return object, ctx
}

func TestToolChoiceNoneReachesAChatBackendWithoutTools(t *testing.T) {
	object, ctx := dispatchToolChoiceNone(t, llmprotocol.OpenAIChatV1)
	for _, member := range []string{"tools", "tool_choice", "parallel_tool_calls"} {
		require.NotContains(t, object, member)
	}
	require.Contains(t, string(object["messages"]), `"tool_call_id":"call_1"`)
	reported := false
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		reported = reported || diagnostic.Field == "tools" && diagnostic.Action == llmprotocol.DiagnosticDropped
	}
	require.True(t, reported, "diagnostics = %+v", ctx.ProtocolDiagnostics)

	object, _ = dispatchToolChoiceNone(t, llmprotocol.AnthropicMessagesV1)
	require.Contains(t, string(object["tools"]), `"get_time"`)
	var choice struct {
		Type string `json:"type"`
	}
	require.NoError(t, json.Unmarshal(object["tool_choice"], &choice))
	require.Equal(t, "none", choice.Type)

	object, _ = dispatchToolChoiceNone(t, llmprotocol.OpenAIResponsesV1)
	require.Contains(t, string(object["tools"]), `"get_time"`)
	require.JSONEq(t, `"none"`, string(object["tool_choice"]))
}
