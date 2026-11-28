package extproc

import (
	"bytes"
	"encoding/json"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// responsesObjectRequest sets every member a Responses object repeats.
const responsesObjectRequest = `{"model":"public-model","input":"What time is it in Paris?",
	"instructions":"Answer in one sentence.",
	"tools":[{"type":"function","name":"get_time","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],
	"tool_choice":{"type":"function","name":"get_time"},"parallel_tool_calls":false,
	"max_output_tokens":64,"temperature":0.2,"top_p":0.9,"text":{"verbosity":"low"},
	"metadata":{"session":"s-1"},"user":"user-1","prompt_cache_key":"cache-1"}`

var responsesObjectEcho = map[string]string{
	"instructions":        `"Answer in one sentence."`,
	"tools":               `[{"type":"function","name":"get_time","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]`,
	"tool_choice":         `{"type":"function","name":"get_time"}`,
	"parallel_tool_calls": `false`, "max_output_tokens": `64`, "temperature": `0.2`, "top_p": `0.9`,
	"text": `{"verbosity":"low"}`, "metadata": `{"session":"s-1"}`, "user": `"user-1"`, "prompt_cache_key": `"cache-1"`,
}

// responsesObjectContext prepares a Responses request as ingress does, then
// changes the neutral request as routing may: the backend receives a decision
// prompt, no tools and another limit, which the client's object never shows.
func responsesObjectContext(t *testing.T, backend llmprotocol.WireFormat) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: backend,
		RequestModel: "public-model", RequestID: "responses-object", TraceContext: t.Context(),
	}
	request, immediate := (&OpenAIRouter{}).prepareProtocolRequest([]byte(responsesObjectRequest), ctx)
	require.Nil(t, immediate)
	require.NotNil(t, request)
	request.Instructions = append(request.Instructions, llmprotocol.InstructionBlock{
		Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "decision prompt"}},
	})
	request.Tools, request.ToolChoice = nil, llmprotocol.ToolChoice{}
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(4096)
	request.Generation++
	return ctx
}

func responsesObjectCutReply(backend llmprotocol.WireFormat) []byte {
	if backend == llmprotocol.AnthropicMessagesV1 {
		return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"It is noon"}],"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":3}}`)
	}
	return []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1792418136,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"It is noon"},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`)
}

func responsesObjectCutStream(backend llmprotocol.WireFormat) []byte {
	if backend == llmprotocol.AnthropicMessagesV1 {
		return []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"It is noon\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}
	chunk := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1792418136,\"model\":\"provider-model\","
	return []byte(chunk + "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"It is noon\"},\"finish_reason\":null}]}\n\n" +
		chunk + "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
		chunk + "\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n" +
		"data: [DONE]\n\n")
}

// assertResponsesObject checks what one client response object repeats and
// returns its created_at.
func assertResponsesObject(t *testing.T, label string, object map[string]json.RawMessage, ctx *RequestContext) int64 {
	t.Helper()
	for name, value := range responsesObjectEcho {
		require.JSONEq(t, value, string(object[name]), "%s %s", label, name)
	}
	require.JSONEq(t, strconv.Quote(ctx.ResponseObjectState.GeneratedResponseID), string(object["id"]), label)
	createdAt, err := strconv.ParseInt(string(object["created_at"]), 10, 64)
	require.NoError(t, err, "%s created_at %s", label, object["created_at"])
	return createdAt
}

func requireMessageStatuses(t *testing.T, label string, output json.RawMessage, want ...string) {
	t.Helper()
	var items []struct{ Type, Status string }
	require.NoError(t, json.Unmarshal(output, &items), label)
	got := make([]string, 0, len(items))
	for _, item := range items {
		got = append(got, item.Type+":"+item.Status)
	}
	require.Equal(t, want, got, label)
}

func expectedResponsesCreatedAt(t *testing.T, backend llmprotocol.WireFormat, createdAt int64, before time.Time) {
	t.Helper()
	if backend == llmprotocol.AnthropicMessagesV1 {
		// Messages replies carry no creation time, so the Router's applies.
		require.GreaterOrEqual(t, createdAt, before.Unix())
		require.LessOrEqual(t, createdAt, time.Now().Unix())
		return
	}
	require.Equal(t, int64(1792418136), createdAt)
}

func TestResponsesObjectRepeatsTheClientRequestBuffered(t *testing.T) {
	for _, backend := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(backend), func(t *testing.T) {
			before := time.Now()
			ctx := responsesObjectContext(t, backend)
			response := (&OpenAIRouter{}).handleNonStreamingResponseBody(responsesObjectCutReply(backend), ctx, 0, false)
			body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			var object map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(body, &object), string(body))
			require.JSONEq(t, `"incomplete"`, string(object["status"]))
			requireMessageStatuses(t, "buffered", object["output"], "message:incomplete")
			expectedResponsesCreatedAt(t, backend, assertResponsesObject(t, "buffered", object, ctx), before)
		})
	}
}

func TestResponsesObjectRepeatsTheClientRequestStreamed(t *testing.T) {
	for _, backend := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(backend), func(t *testing.T) {
			before := time.Now()
			ctx := responsesObjectContext(t, backend)
			response := (&OpenAIRouter{}).handleSemanticStreamingResponseBody(responsesObjectCutStream(backend), true, ctx)
			body := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			require.False(t, ctx.StreamingAborted, string(body))
			createdAt := map[string]int64{}
			var itemsDone []json.RawMessage
			for _, frame := range bytes.Split(body, []byte("\n\n")) {
				_, data, found := bytes.Cut(frame, []byte("data: "))
				if !found {
					continue
				}
				var event struct {
					Type     string                     `json:"type"`
					Response map[string]json.RawMessage `json:"response"`
					Item     json.RawMessage            `json:"item"`
				}
				require.NoError(t, json.Unmarshal(data, &event), string(data))
				if event.Response != nil {
					createdAt[event.Type] = assertResponsesObject(t, event.Type, event.Response, ctx)
				}
				if event.Type == "response.output_item.done" {
					itemsDone = append(itemsDone, event.Item)
				}
				if event.Type == "response.incomplete" {
					requireMessageStatuses(t, event.Type, event.Response["output"], "message:incomplete")
				}
			}
			require.Len(t, createdAt, 3, string(body))
			require.Equal(t, createdAt["response.created"], createdAt["response.in_progress"])
			require.Equal(t, createdAt["response.created"], createdAt["response.incomplete"])
			expectedResponsesCreatedAt(t, backend, createdAt["response.created"], before)
			require.Len(t, itemsDone, 1)
			requireMessageStatuses(t, "output_item.done", append(append([]byte("["), itemsDone[0]...), ']'), "message:incomplete")
			require.NotNil(t, ctx.SemanticResponse)
			require.Equal(t, createdAt["response.created"], ctx.SemanticResponse.CreatedAt.Unix(), "a stored object keeps the streamed creation time")
		})
	}
}
