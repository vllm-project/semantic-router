package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// The tools Codex CLI 0.166.0 sends with an MCP server, here the Agent tool
// vllm-sr connect codex --delegate claude-code registers, and its last call.
const codexMCPAgentRequest = `{"model":"worker","input":[
	{"type":"message","role":"user","content":[{"type":"input_text","text":"Ask Claude Code."}]},
	{"type":"function_call","call_id":"call_1","namespace":"mcp__claude","name":"Agent","arguments":"{}"},
	{"type":"function_call_output","call_id":"call_1","output":"pong"}
],"tools":[
	{"type":"function","name":"exec_command","parameters":{"type":"object"}},
	{"type":"namespace","name":"mcp__claude","description":"Tools in the mcp__claude namespace.","tools":[
		{"type":"function","name":"Agent","description":"Launch a new agent.","parameters":{"type":"object"}}
	]}
]}`

// An OpenAI-hosted Responses backend reads tool namespaces, so a Codex MCP
// tool reaches it as Codex declared it, as it would without the gateway; a
// compatible Responses backend receives the flat name it can read.
func TestResponsesNamespacesReachAnOpenAIHostedBackend(t *testing.T) {
	for _, tc := range []struct {
		name, provider, endpoint string
		hosted                   bool
	}{
		{"hosted", "openai", "https://chatgpt.com/backend-api/codex", true},
		{"compatible", "vllm", "http://127.0.0.1:8000/v1", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := routingTestRouter("worker")
			params := router.Config.ModelConfig["worker"]
			params.APIFormat = config.APIFormatResponses
			router.Config.ModelConfig["worker"] = params
			router.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: tc.provider, BaseURL: tc.endpoint}
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(codexMCPAgentRequest))
			require.NoError(t, err)
			ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, &request)
			ctx.ProtocolEnvelope = envelope
			_, err = router.prepareProviderDispatch(&request, "worker", "", false, ctx)
			require.NoError(t, err)
			body, err := router.encodeDispatchRequest(ctx)
			require.NoError(t, err)
			if tc.hosted {
				require.Contains(t, string(body), `{"type":"namespace","name":"mcp__claude","description":"Tools in the mcp__claude namespace.","tools":[{"type":"function","name":"Agent","description":"Launch a new agent."`)
				require.Contains(t, string(body), `"name":"Agent","namespace":"mcp__claude"`)
				require.NotContains(t, string(body), "mcp__claude__Agent")
				return
			}
			require.NotContains(t, string(body), `"namespace"`)
			require.Contains(t, string(body), `"name":"mcp__claude__Agent"`)
		})
	}
}

// ChatGPT's Codex backend answers a request that carries namespace tools with
// an empty output list in response.completed, as recorded with Codex 0.166.0
// on 2026-11-17. The streamed items stand, so the client still gets the
// terminal event and the record keeps the answer.
func TestEmptyTerminalOutputFromAHostedBackendCompletesTheStream(t *testing.T) {
	item := `"type":"message","id":"msg_1","role":"assistant"`
	stream := []byte(`event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"response_1","object":"response","created_at":1,"model":"source-model","status":"in_progress","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{` + item + `,"status":"in_progress","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":2,"output_index":0,"item_id":"msg_1","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":3,"output_index":0,"item_id":"msg_1","content_index":0,"delta":"Hi!"}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":4,"output_index":0,"item_id":"msg_1","content_index":0,"text":"Hi!"}

event: response.content_part.done
data: {"type":"response.content_part.done","sequence_number":5,"output_index":0,"item_id":"msg_1","content_index":0,"part":{"type":"output_text","text":"Hi!","annotations":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":6,"output_index":0,"item":{` + item + `,"status":"completed","content":[{"type":"output_text","text":"Hi!","annotations":[]}]}}

event: response.completed
data: {"type":"response.completed","sequence_number":7,"response":{"id":"response_1","object":"response","created_at":1,"model":"source-model","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}

`)
	ctx := namespacedToolContext(t, llmprotocol.OpenAIResponsesV1)
	response := (&OpenAIRouter{}).handleSemanticStreamingResponseBody(stream, true, ctx)
	body := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.False(t, ctx.StreamingAborted, body)
	require.Contains(t, body, "event: response.completed")
	require.NotContains(t, body, "response.failed")
	require.NotNil(t, ctx.SemanticResponse)
	require.Len(t, ctx.SemanticResponse.Output, 1)
}
