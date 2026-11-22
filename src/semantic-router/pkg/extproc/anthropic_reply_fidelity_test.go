package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// translatedAnthropicID is the id a Messages client receives for one a backend
// of another wire format assigned.
func translatedAnthropicID(prefix, id string) string {
	sum := sha256.Sum256([]byte(id))
	return prefix + hex.EncodeToString(sum[:12])
}

const weatherTool = `{"name":"lookup","description":"Look up the weather in a city.",` +
	`"input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}`

const chatWeatherReply = `{"id":"chatcmpl-weather","object":"chat.completion","created":1,"model":"worker","choices":[{"index":0,` +
	`"message":{"role":"assistant","reasoning_content":"Look it up.","content":null,"tool_calls":[{"id":"call_weather",` +
	`"type":"function","function":{"name":"lookup","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}],` +
	`"usage":{"prompt_tokens":40,"completion_tokens":8,"total_tokens":48}}`

func chatWeatherStream() []string {
	chunk := func(fields string) string {
		return `data: {"id":"chatcmpl-weather","object":"chat.completion.chunk","created":1,"model":"worker",` + fields + "}\n\n"
	}
	return []string{
		chunk(`"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Look it up."},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function",` +
			`"function":{"name":"lookup","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]`),
		chunk(`"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":8,"total_tokens":48}`),
		"data: [DONE]\n\n",
	}
}

// messagesStreamEvent is the part of a Messages stream event this test reads.
type messagesStreamEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message struct {
		ID    string `json:"id"`
		Usage struct {
			InputTokens int64 `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message"`
	ContentBlock struct {
		Type      string  `json:"type"`
		ID        string  `json:"id"`
		Signature *string `json:"signature"`
	} `json:"content_block"`
	Delta struct {
		Type      string  `json:"type"`
		Signature *string `json:"signature"`
	} `json:"delta"`
	Usage struct {
		InputTokens int64 `json:"input_tokens"`
	} `json:"usage"`
}

func messagesOfChatBackendRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: worker
  models:
    - name: worker
      provider_model_id: worker
      api_format: openai
      backend_refs: [{name: b, provider: openai, endpoint: "https://api.example.com/v1"}]
recipes:
  - name: default
    decisions:
      - name: tools
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: worker}]
`))
	require.NoError(t, err)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = router.Close() })
	router.CredentialResolver = newTestCredentialResolver(cfg)
	return router
}

// sendMessagesTurn runs a Messages request through the request phases and
// returns the body sent to the backend.
func sendMessagesTurn(t *testing.T, router *OpenAIRouter, messages string, stream bool) ([]byte, *RequestContext) {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}}
	require.NoError(t, router.processRequestHeaders(NewMockStream(nil), newRequestHeaders("POST", "/v1/messages"), ctx))
	body := fmt.Sprintf(`{"model":"vllm-sr/auto","max_tokens":256,"stream":%t,"tools":[%s],"messages":%s}`, stream, weatherTool, messages)
	replies := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(replies, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx))
	require.Len(t, replies.Responses, 1)
	require.Nil(t, replies.Responses[0].GetImmediateResponse(), "the turn was refused")
	return replies.Responses[0].GetRequestBody().GetResponse().GetBodyMutation().GetBody(), ctx
}

// answerMessagesTurn runs the backend's reply, read by read, through the
// response phases and returns what the client receives.
func answerMessagesTurn(t *testing.T, router *OpenAIRouter, ctx *RequestContext, contentType string, reads ...string) string {
	t.Helper()
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseHeaders(stream, &ext_proc.ProcessingRequest{}, &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"}, {Key: "content-type", Value: contentType},
		}}},
	}, ctx))
	var received strings.Builder
	for index, read := range reads {
		require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: []byte(read), EndOfStream: index == len(reads)-1},
		}, ctx))
		response := stream.Responses[len(stream.Responses)-1]
		require.Nil(t, response.GetImmediateResponse(), "the reply was refused")
		if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			read = string(mutation.GetBody())
		}
		received.WriteString(read)
	}
	return received.String()
}

func messagesStreamEvents(t *testing.T, stream string) []messagesStreamEvent {
	t.Helper()
	var events []messagesStreamEvent
	for _, frame := range strings.Split(strings.TrimSpace(stream), "\n\n") {
		_, data, found := strings.Cut(frame, "data: ")
		require.True(t, found, "frame %q", frame)
		var event messagesStreamEvent
		require.NoError(t, json.Unmarshal([]byte(data), &event), "frame %q", frame)
		events = append(events, event)
	}
	return events
}

// A Messages client of a Chat backend gets the reply as a Messages backend
// sends it, buffered and streamed alike: thinking with a signature, msg_ and
// toolu_ ids, and a message_start that reports the Router's estimate of the
// input until message_delta carries the backend's count. Sent back as the next
// turn's history, the reply reaches the Chat backend with the tool result
// paired with its call.
func TestMessagesClientOfAChatBackendGetsAMessagesReply(t *testing.T) {
	router := messagesOfChatBackendRouter(t)
	messageID, toolID := translatedAnthropicID("msg_", "chatcmpl-weather"), translatedAnthropicID("toolu_", "call_weather")
	question := `[{"role":"user","content":"What is the weather in Paris?"}]`

	_, ctx := sendMessagesTurn(t, router, question, false)
	buffered := answerMessagesTurn(t, router, ctx, "application/json", chatWeatherReply)
	var reply struct {
		ID      string            `json:"id"`
		Content []json.RawMessage `json:"content"`
	}
	require.NoError(t, json.Unmarshal([]byte(buffered), &reply), buffered)
	require.Equal(t, messageID, reply.ID)
	require.Len(t, reply.Content, 2, buffered)
	require.JSONEq(t, `{"type":"thinking","thinking":"Look it up.","signature":""}`, string(reply.Content[0]))
	require.JSONEq(t, fmt.Sprintf(`{"type":"tool_use","id":%q,"name":"lookup","input":{"city":"Paris"}}`, toolID), string(reply.Content[1]))

	_, ctx = sendMessagesTurn(t, router, question, true)
	estimate := int64(estimateCountTokens(ctx.SemanticRequest))
	require.Positive(t, estimate)
	streamed := answerMessagesTurn(t, router, ctx, "text/event-stream", chatWeatherStream()...)
	events := messagesStreamEvents(t, streamed)
	require.Equal(t, "message_start", events[0].Type, streamed)
	require.Equal(t, messageID, events[0].Message.ID)
	require.Equal(t, estimate, events[0].Message.Usage.InputTokens)
	var thinkingStarted, thinkingSigned, toolStarted, finalUsage bool
	for index, event := range events {
		switch {
		case event.Type == "content_block_start" && event.ContentBlock.Type == "thinking":
			require.NotNil(t, event.ContentBlock.Signature, streamed)
			require.Empty(t, *event.ContentBlock.Signature)
			thinkingStarted = true
		case event.Type == "content_block_stop" && event.Index == 0:
			previous := events[index-1]
			require.Equal(t, "signature_delta", previous.Delta.Type, streamed)
			require.NotNil(t, previous.Delta.Signature)
			require.Empty(t, *previous.Delta.Signature)
			thinkingSigned = true
		case event.Type == "content_block_start" && event.ContentBlock.Type == "tool_use":
			require.Equal(t, toolID, event.ContentBlock.ID)
			toolStarted = true
		case event.Type == "message_delta":
			require.EqualValues(t, 40, event.Usage.InputTokens)
			finalUsage = true
		}
	}
	require.True(t, thinkingStarted && thinkingSigned && toolStarted && finalUsage, streamed)
	require.Equal(t, messageID+" "+toolID, replyIdentifiers(*ctx.SemanticResponse), "the answer the Router keeps")

	history := fmt.Sprintf(`[{"role":"user","content":"What is the weather in Paris?"},{"role":"assistant","content":[%s,%s]},`+
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":%q,"content":"18C and sunny"}]}]`,
		reply.Content[0], reply.Content[1], toolID)
	sent, _ := sendMessagesTurn(t, router, history, false)
	var chat struct {
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID string `json:"id"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(sent, &chat), "%s", sent)
	var call, result string
	for _, message := range chat.Messages {
		if message.Role == "assistant" && len(message.ToolCalls) == 1 {
			call = message.ToolCalls[0].ID
		}
		if message.Role == "tool" {
			result = message.ToolCallID
		}
	}
	require.Equal(t, toolID, call, "%s", sent)
	require.Equal(t, toolID, result, "%s", sent)
}

// replyIdentifiers is a reply's message id followed by the ids of its tool
// calls.
func replyIdentifiers(response llmprotocol.Response) string {
	ids := []string{response.ID}
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.ToolCall != nil {
				ids = append(ids, content.ToolCall.ID)
			}
		}
	}
	return strings.Join(ids, " ")
}

func clientReplyIdentifiers(t *testing.T, body []byte, streamed bool) string {
	t.Helper()
	engine := protocolcodec.NewBuiltinEngine()
	if streamed {
		response, _, err := engine.DecodeResponseStream(llmprotocol.AnthropicMessagesV1, body, llmprotocol.StreamContext{Context: t.Context()})
		require.NoError(t, err, "%s", body)
		return replyIdentifiers(response)
	}
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	require.NoError(t, err, "%s", body)
	return replyIdentifiers(response)
}

// A fallback candidate's answer is rendered for a streaming client from the
// neutral response; a Messages client gets the ids of the translated buffered
// answer either way.
func TestFallbackAnswerGivesMessagesClientsMessagesIdentifiers(t *testing.T) {
	want := translatedAnthropicID("msg_", "chatcmpl-weather") + " " + translatedAnthropicID("toolu_", "call_weather")
	for _, streamed := range []bool{false, true} {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
		ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
		ctx.UpstreamStatusCode = http.StatusBadGateway
		ctx.ExpectStreamingResponse, ctx.SemanticRequest.Stream = streamed, streamed
		router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
			return []byte(chatWeatherReply), http.StatusOK, nil
		}
		answer := router.handleUpstreamTransportError([]byte("upstream 502"), ctx).GetImmediateResponse()
		require.EqualValues(t, http.StatusOK, answer.GetStatus().GetCode(), "streamed=%t: %s", streamed, answer.GetBody())
		require.Equal(t, want, clientReplyIdentifiers(t, answer.GetBody(), streamed), "streamed=%t", streamed)
		require.Equal(t, want, replyIdentifiers(*ctx.SemanticResponse), "streamed=%t: the answer the Router keeps", streamed)
	}
}

// An answer the response guard held is rendered again from the neutral
// response, with the guard's notes or as its refusal; a Messages client gets
// the ids a live stream gives.
func TestHeldStreamRenderGivesMessagesClientsMessagesIdentifiers(t *testing.T) {
	response, _, _, err := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(chatWeatherReply))
	require.NoError(t, err)
	ctx := &RequestContext{SourceFormat: llmprotocol.AnthropicMessagesV1, TargetFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()}
	body, err := (&OpenAIRouter{}).encodeClientStream(ctx, response)
	require.NoError(t, err)
	require.Equal(t, translatedAnthropicID("msg_", "chatcmpl-weather")+" "+translatedAnthropicID("toolu_", "call_weather"),
		clientReplyIdentifiers(t, body, true))
}
