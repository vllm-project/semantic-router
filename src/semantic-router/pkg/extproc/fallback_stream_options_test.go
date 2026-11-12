package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// candidateUsage is the usage model-fallback-1's backend reports.
var candidateUsage = map[string]interface{}{"prompt_tokens": 13.0, "completion_tokens": 7.0, "total_tokens": 20.0}

const candidateBufferedAnswer = `{"id":"chatcmpl-next","object":"chat.completion","created":1700000000,` +
	`"model":"model-fallback-1","choices":[{"index":0,"message":{"role":"assistant",` +
	`"content":"from the second model"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":13,"completion_tokens":7,"total_tokens":20}}`

// streamWithOptions makes the client stream with options, and takes the
// primary's fallback snapshot the way the body phase takes it.
func streamWithOptions(t *testing.T, ctx *RequestContext, options llmprotocol.StreamOptions) {
	t.Helper()
	streamRequested(ctx)
	ctx.SemanticRequest.StreamOptions = options
	snapshot, err := cloneSemanticRequestForReplay(ctx.SemanticRequest)
	require.NoError(t, err)
	ctx.FallbackRequest = snapshot
}

// answerBuffered serves model-fallback-1's buffered answer and keeps the
// request its call sent.
func answerBuffered(t *testing.T, router *OpenAIRouter) *map[string]interface{} {
	t.Helper()
	var sent map[string]interface{}
	router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
		require.Equal(t, "model-fallback-1", model)
		require.NoError(t, json.Unmarshal(body, &sent))
		return []byte(candidateBufferedAnswer), http.StatusOK, nil
	}
	return &sent
}

func status503() *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "503"}}},
	}}
}

// chatStreamChunks decodes a Chat stream's data frames; [DONE] decodes as nil.
func chatStreamChunks(t *testing.T, body []byte) []map[string]interface{} {
	t.Helper()
	var chunks []map[string]interface{}
	for _, line := range strings.Split(string(body), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			chunks = append(chunks, nil)
			continue
		}
		var chunk map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(data), &chunk), data)
		chunks = append(chunks, chunk)
	}
	return chunks
}

// requireChatStreamEnd checks how a Chat stream ends: its finish chunk, then
// the usage-only chunk with an empty choices list when usage is expected,
// then [DONE]. No other chunk carries usage.
func requireChatStreamEnd(t *testing.T, body []byte, usage map[string]interface{}) {
	t.Helper()
	chunks := chatStreamChunks(t, body)
	require.NotEmpty(t, chunks)
	require.Nil(t, chunks[len(chunks)-1], "the stream does not end with [DONE]:\n%s", body)
	chunks = chunks[:len(chunks)-1]
	if usage != nil {
		require.NotEmpty(t, chunks)
		last := chunks[len(chunks)-1]
		require.Equal(t, usage, last["usage"], "the usage-only chunk is not the last before [DONE]:\n%s", body)
		require.Equal(t, []interface{}{}, last["choices"], "the usage-only chunk has no empty choices list:\n%s", body)
		chunks = chunks[:len(chunks)-1]
	}
	require.NotEmpty(t, chunks)
	choices, _ := chunks[len(chunks)-1]["choices"].([]interface{})
	require.Len(t, choices, 1, "no finish chunk before the end:\n%s", body)
	require.Equal(t, "stop", choices[0].(map[string]interface{})["finish_reason"])
	for _, chunk := range chunks {
		require.Nil(t, chunk["usage"], "usage before the usage-only chunk:\n%s", body)
	}
}

// A streamed Chat request with stream_options falls back behind Envoy as any
// other: the candidate's buffered call carries no stream options, and the
// client's stream keeps the client's, with the candidate's usage last when
// it asked for usage.
func TestResponsePhaseFallbackAnswersAStreamWithStreamOptions(t *testing.T) {
	yes, no := true, false
	for _, tc := range []struct {
		name    string
		options llmprotocol.StreamOptions
		usage   map[string]interface{}
	}{
		{name: "include_usage", options: llmprotocol.StreamOptions{IncludeUsage: &yes}, usage: candidateUsage},
		{name: "include_usage false", options: llmprotocol.StreamOptions{IncludeUsage: &no}},
		{name: "include_obfuscation false", options: llmprotocol.StreamOptions{IncludeObfuscation: &no}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
			ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
			streamWithOptions(t, ctx, tc.options)
			sent := answerBuffered(t, router)

			resp, err := router.handleResponseHeaders(status503(), ctx)
			require.NoError(t, err)
			require.NotNil(t, *sent, "the candidate was never called")
			require.NotEqual(t, true, (*sent)["stream"])
			require.NotContains(t, *sent, "stream_options")

			immediate := resp.GetImmediateResponse()
			require.NotNil(t, immediate)
			require.Equal(t, typev3.StatusCode_OK, immediate.GetStatus().GetCode())
			require.Equal(t, "text/event-stream", immediateHeaderValue(resp, "content-type"))
			require.NoError(t, protocolcodec.NewBuiltinEngine().ValidateEncodedResponse(immediate.GetBody(), true))
			require.Contains(t, string(immediate.GetBody()), "from the second model")
			requireChatStreamEnd(t, immediate.GetBody(), tc.usage)
			require.Equal(t, "succeeded", ctx.FallbackRecord.FinalStatus)
			require.Equal(t, tc.options, ctx.SemanticRequest.StreamOptions, "the client's stream options were lost")
		})
	}
}

// A Responses client's stream_options fall back the same way.
func TestResponsePhaseFallbackAnswersAResponsesStreamWithStreamOptions(t *testing.T) {
	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
	ctx.SemanticRequest.Trusted.SourceFormat = llmprotocol.OpenAIResponsesV1
	no := false
	streamWithOptions(t, ctx, llmprotocol.StreamOptions{IncludeObfuscation: &no})
	sent := answerBuffered(t, router)

	resp, err := router.handleResponseHeaders(status503(), ctx)
	require.NoError(t, err)
	require.NotNil(t, *sent, "the candidate was never called")
	require.NotContains(t, *sent, "stream_options")
	immediate := resp.GetImmediateResponse()
	require.NotNil(t, immediate)
	require.Equal(t, typev3.StatusCode_OK, immediate.GetStatus().GetCode())
	body := string(immediate.GetBody())
	require.Contains(t, body, "from the second model")
	require.Contains(t, body, "event: response.completed")
	require.Contains(t, body, `"output_tokens":7`)
	assertClientStreamDecodes(t, protocolcodec.NewBuiltinEngine(), llmprotocol.OpenAIResponsesV1, immediate.GetBody())
}

// A response policy that rewrites the fallback answer renders the client's
// stream again, and the usage-only chunk stays last.
func TestResponsePhaseFallbackKeepsTheUsageChunkLastWhenThePolicyRewritesTheAnswer(t *testing.T) {
	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.VSRSelectedDecision = responseGuardDecision(groundedGuard(config.ResponseGuardAnnotate, true, ""))
	yes := true
	streamWithOptions(t, ctx, llmprotocol.StreamOptions{IncludeUsage: &yes})
	sent := answerBuffered(t, router)

	resp, err := router.handleResponseHeaders(status503(), ctx)
	require.NoError(t, err)
	require.NotNil(t, *sent, "the candidate was never called")
	body := resp.GetImmediateResponse().GetBody()
	require.Contains(t, string(body), "[Unverified Response]", "the policy did not rewrite the answer")
	requireChatStreamEnd(t, body, candidateUsage)
}

// candidateStreamReply is what a Chat backend streams for the candidate's
// call: like vLLM, the usage-only chunk only when the call asks for usage.
func candidateStreamReply(t *testing.T, call []byte) []byte {
	t.Helper()
	var request struct {
		StreamOptions *struct {
			IncludeUsage *bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	require.NoError(t, json.Unmarshal(call, &request))
	chunk := func(rest string) string {
		return `data: {"id":"chatcmpl-next","object":"chat.completion.chunk","created":1700000000,"model":"model-fallback-1",` + rest + "}\n\n"
	}
	reply := chunk(`"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]`) +
		chunk(`"choices":[{"index":0,"delta":{"content":"from the second model"},"finish_reason":null}]`) +
		chunk(`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]`)
	if request.StreamOptions != nil && request.StreamOptions.IncludeUsage != nil && *request.StreamOptions.IncludeUsage {
		reply += chunk(`"choices":[],"usage":{"prompt_tokens":13,"completion_tokens":7,"total_tokens":20}`)
	}
	return []byte(reply + "data: [DONE]\n\n")
}

// streamTheCandidate answers the native chain's candidate call with its
// backend's stream and returns what the client receives.
func streamTheCandidate(t *testing.T, session *routingSession, call *routing.Call) []byte {
	t.Helper()
	reply := candidateStreamReply(t, call.Request.Body)
	_, err := session.ResponseHeaders(routing.Header{
		{Name: ":status", Value: "200"}, {Name: "content-type", Value: "text/event-stream"},
	}, false)
	require.NoError(t, err)
	effect, err := session.ResponseBody(reply, true)
	require.NoError(t, err)
	return routing.ApplyBodyMutation(reply, effect.Body)
}

// The native gateway's candidate streams, and asks a Chat backend for its
// usage as the primary's dispatch does, whatever the client asked; the
// client's own stream options go along, and the client's stream keeps them.
func TestNativeFallbackAsksAStreamedChatCandidateForUsage(t *testing.T) {
	yes, no := true, false
	for _, tc := range []struct {
		name    string
		client  func(*RequestContext)
		options llmprotocol.StreamOptions
		want    map[string]interface{}
	}{
		{name: "chat", want: map[string]interface{}{"include_usage": true}},
		{name: "chat include_usage", options: llmprotocol.StreamOptions{IncludeUsage: &yes}, want: map[string]interface{}{"include_usage": true}},
		{name: "chat include_usage false", options: llmprotocol.StreamOptions{IncludeUsage: &no}, want: map[string]interface{}{"include_usage": true}},
		{name: "chat include_obfuscation false", options: llmprotocol.StreamOptions{IncludeObfuscation: &no}, want: map[string]interface{}{"include_usage": true, "include_obfuscation": false}},
		{name: "messages", client: messagesClient, want: map[string]interface{}{"include_usage": true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
			ctx := session.ctx
			if tc.client != nil {
				tc.client(ctx)
			}
			streamWithOptions(t, ctx, tc.options)

			step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
			require.NoError(t, err)
			require.NotNil(t, step.Call)
			var call map[string]interface{}
			require.NoError(t, json.Unmarshal(step.Call.Request.Body, &call))
			require.Equal(t, true, call["stream"])
			require.Equal(t, tc.want, call["stream_options"])

			session.fallback.succeeded(http.StatusOK)
			require.Equal(t, tc.options, ctx.SemanticRequest.StreamOptions, "the client's stream options changed")
		})
	}

	t.Run("buffered", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		require.NotContains(t, string(step.Call.Request.Body), "stream_options")
	})
}

// After a native fallback the client's stream carries the candidate's usage
// where its protocol puts it: the usage-only chunk last for a Chat client
// that asked for it, the final message_delta for a Messages client.
func TestNativeFallbackStreamCarriesTheCandidatesUsage(t *testing.T) {
	yes := true
	t.Run("chat include_usage", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		streamWithOptions(t, session.ctx, llmprotocol.StreamOptions{IncludeUsage: &yes})
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)

		client := streamTheCandidate(t, session, step.Call)
		require.Contains(t, string(client), "from the second model")
		requireChatStreamEnd(t, client, candidateUsage)
	})

	t.Run("chat", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		streamWithOptions(t, session.ctx, llmprotocol.StreamOptions{})
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)

		client := streamTheCandidate(t, session, step.Call)
		require.Contains(t, string(client), "from the second model")
		requireChatStreamEnd(t, client, nil)
	})

	t.Run("messages", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		messagesClient(session.ctx)
		streamWithOptions(t, session.ctx, llmprotocol.StreamOptions{})
		step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)

		client := streamTheCandidate(t, session, step.Call)
		require.Contains(t, string(client), "from the second model")
		usage := messagesStreamUsage(t, client)
		require.Equal(t, 13.0, usage["input_tokens"], "message_delta usage:\n%s", client)
		require.Equal(t, 7.0, usage["output_tokens"], "message_delta usage:\n%s", client)
	})
}

// messagesStreamUsage is the usage a Messages stream's message_delta reports.
func messagesStreamUsage(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	for _, line := range strings.Split(string(body), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event struct {
			Type  string                 `json:"type"`
			Usage map[string]interface{} `json:"usage"`
		}
		require.NoError(t, json.Unmarshal([]byte(data), &event), data)
		if event.Type == "message_delta" {
			return event.Usage
		}
	}
	require.Fail(t, fmt.Sprintf("no message_delta in the stream:\n%s", body))
	return nil
}
