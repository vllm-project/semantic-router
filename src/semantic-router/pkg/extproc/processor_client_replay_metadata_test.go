package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const replayTools = `[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]`

// A vLLM 0.33 reply with a tool call: refusal, annotations, audio and
// function_call are null, and reasoning carries the thinking or is null.
const replayVLLMToolCallReply = `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"backend-model",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":null,"annotations":null,"audio":null,` +
	`"function_call":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\": \"ls\"}"}}],` +
	`"reasoning":REASONING},"logprobs":null,"finish_reason":"tool_calls","stop_reason":null,"token_ids":null}],` +
	`"usage":{"prompt_tokens":20,"completion_tokens":8,"total_tokens":28}}`

const replayVLLMTextReply = `{"id":"chatcmpl-2","object":"chat.completion","created":1,"model":"backend-model",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":"README.md is there.","refusal":null,"annotations":null,` +
	`"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null}],` +
	`"usage":{"prompt_tokens":30,"completion_tokens":4,"total_tokens":34}}`

const replayVLLMTextStream = `data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"backend-model","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"backend-model","choices":[{"index":0,"delta":{"content":"README.md is there."},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"backend-model","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop","stop_reason":null}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"backend-model","choices":[],"usage":{"prompt_tokens":30,"completion_tokens":4,"total_tokens":34}}

data: [DONE]

`

// A tool loop's second turn from a litellm client, such as mini-SWE-agent
// 2.4.6, that replays each assistant message as litellm parsed it.
func TestChatTurnReplayingALiteLLMParsedReplyReachesTheBackend(t *testing.T) {
	styles := map[string]string{"reasoning": `"The user wants the files listed."`, "null reasoning": "null"}
	for style, reasoning := range styles {
		for _, stream := range []bool{false, true} {
			mode := map[bool]string{false: "buffered", true: "streamed"}[stream]
			t.Run(style+"/"+mode, func(t *testing.T) {
				router := &OpenAIRouter{}
				first := replayContext(t)
				_, immediate := router.prepareProtocolRequest([]byte(`{"model":"public-model",`+
					`"messages":[{"role":"user","content":"List the files."}],"tools":`+replayTools+`}`), first)
				require.Nil(t, immediate)
				upstream := strings.Replace(replayVLLMToolCallReply, "REASONING", reasoning, 1)
				received := replayClientBody(t, router.handleNonStreamingResponseBody([]byte(upstream), first, 0, false), upstream)

				second := replayContext(t)
				_, immediate = router.prepareProtocolRequest(litellmSecondTurn(t, received, stream), second)
				require.Nil(t, immediate, "the replayed turn was refused")
				dispatched, err := router.encodeDispatchRequest(second)
				require.NoError(t, err)
				require.NotContains(t, string(dispatched), "provider_specific_fields")
				require.Contains(t, string(dispatched), `"tool_call_id":"call_1"`)
				require.Contains(t, responseWarnings(second), `field="messages[].provider_specific_fields"`)

				if stream {
					response := router.handleSemanticStreamingResponseBody([]byte(replayVLLMTextStream), true, second)
					frames := string(replayClientBody(t, response, replayVLLMTextStream))
					require.True(t, second.StreamingComplete && !second.StreamingAborted)
					require.Contains(t, frames, "README.md is there.")
					require.Contains(t, frames, "data: [DONE]")
					return
				}
				response := router.handleNonStreamingResponseBody([]byte(replayVLLMTextReply), second, 0, false)
				require.Contains(t, string(replayClientBody(t, response, replayVLLMTextReply)), "README.md is there.")
			})
		}
	}
}

func replayContext(t *testing.T) *RequestContext {
	return &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model", TraceContext: t.Context(),
	}
}

// replayClientBody is what the client receives: the upstream bytes unless
// the Router rewrote them.
func replayClientBody(t *testing.T, reply *ext_proc.ProcessingResponse, upstream string) []byte {
	t.Helper()
	if immediate := reply.GetImmediateResponse(); immediate != nil {
		t.Fatalf("response was replaced: status=%v body=%s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	mutation := reply.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		return []byte(upstream)
	}
	if streamed := mutation.GetStreamedResponse(); streamed != nil {
		return streamed.GetBody()
	}
	return mutation.GetBody()
}

// litellmSecondTurn replays the assistant message the way litellm 1.82.6
// sends it back: the members of its Message type at the top level without
// the null ones, every other member in provider_specific_fields, and vLLM's
// reasoning also as reasoning_content.
func litellmSecondTurn(t *testing.T, reply []byte, stream bool) []byte {
	t.Helper()
	var wire struct {
		Choices []struct {
			Message map[string]any `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(reply, &wire))
	require.Len(t, wire.Choices, 1)
	replayed := map[string]any{}
	providerFields := map[string]any{}
	for name, value := range wire.Choices[0].Message {
		switch name {
		case "role", "content", "tool_calls", "function_call", "audio", "annotations":
			if value != nil {
				replayed[name] = value
			}
		default:
			providerFields[name] = value
		}
	}
	if reasoning, ok := wire.Choices[0].Message["reasoning"].(string); ok {
		replayed["reasoning_content"] = reasoning
		providerFields["reasoning_content"] = reasoning
	}
	replayed["provider_specific_fields"] = providerFields
	var tools []any
	require.NoError(t, json.Unmarshal([]byte(replayTools), &tools))
	body, err := json.Marshal(map[string]any{
		"model": "public-model", "stream": stream, "tools": tools,
		"messages": []any{
			map[string]any{"role": "user", "content": "List the files."},
			replayed,
			map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
		},
	})
	require.NoError(t, err)
	return body
}
