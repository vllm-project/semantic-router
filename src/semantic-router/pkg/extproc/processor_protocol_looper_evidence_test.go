package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestConfidenceLogprobsReachBackendAfterNeutralReencoding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		require.NoError(t, err)
		router := &OpenAIRouter{}
		ctx := &RequestContext{LooperRequest: true, SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: llmprotocol.OpenAIChatV1}
		neutral, immediate := router.prepareProtocolRequest(body, ctx)
		require.Nil(t, immediate)
		require.NotNil(t, neutral)
		neutral.Model = "provider-model"
		neutral.Generation++ // Force a real encode instead of source replay.
		encoded, err := router.encodeDispatchRequest(ctx)
		require.NoError(t, err)
		var dispatched map[string]interface{}
		require.NoError(t, json.Unmarshal(encoded, &dispatched))
		require.Equal(t, "provider-model", dispatched["model"])
		require.Equal(t, true, dispatched["logprobs"])
		require.EqualValues(t, 2, dispatched["top_logprobs"])
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1}]}}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))
	}))
	defer server.Close()
	client := looper.NewClient(&config.LooperConfig{Endpoint: server.URL})
	defer client.Close()
	request := openai.ChatCompletionNewParams{Model: "public-model", Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")}}
	response, err := client.CallModel(context.Background(), &request, "public-model", false, 1, &looper.LogprobsConfig{Enabled: true, TopLogprobs: 2}, "")
	require.NoError(t, err)
	require.InDelta(t, -0.1, response.AverageLogprob, 1e-10)
}

func TestLooperLogprobsKeepPublicAndCrossProtocolContractsClosed(t *testing.T) {
	body := []byte(`{"model":"model","messages":[{"role":"user","content":"hello"}],"logprobs":true,"top_logprobs":2}`)
	router := &OpenAIRouter{}
	public := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1}
	_, immediate := router.prepareProtocolRequest(body, public)
	require.NotNil(t, immediate, "public logprobs contract remains unsupported")
	require.NotNil(t, public.ImmediateProtocolError)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		ctx := &RequestContext{LooperRequest: true, SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: target}
		_, immediate = router.prepareProtocolRequest(body, ctx)
		require.Nil(t, immediate)
		_, err := router.encodeDispatchRequest(ctx)
		var protocolErr *llmprotocol.ProtocolError
		require.ErrorAs(t, err, &protocolErr)
		require.Equal(t, "unsupported_logprobs", protocolErr.Code)
	}
}

func TestLooperEvidenceDoesNotBypassOrdinaryRequestValidation(t *testing.T) {
	for _, body := range []string{
		`{"model":"m","messages":[],"logprobs":true,"top_logprobs":2}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":true,"logprobs":false,"top_logprobs":2}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":"true","top_logprobs":2}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":99}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":2,"audio":{"voice":"alloy"}}`,
	} {
		ctx := &RequestContext{LooperRequest: true, SourceFormat: llmprotocol.OpenAIChatV1}
		request, immediate := (&OpenAIRouter{}).prepareProtocolRequest([]byte(body), ctx)
		require.Nil(t, request, body)
		require.NotNil(t, immediate, body)
		require.Nil(t, ctx.LooperLogprobs, body)
	}
}
