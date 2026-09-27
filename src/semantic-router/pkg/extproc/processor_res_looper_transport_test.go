package extproc

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// Exercise the actual Looper HTTP client against ExtProc's internal response
// boundary. Envoy transport is covered separately by the deployment profile.
func TestLooperInternalResponseProtocolOverHTTP(t *testing.T) {
	for _, backendFormat := range extProcMatrixFormats {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s_stream_%t", backendFormat, streaming), func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
					require.Equal(t, "true", request.Header.Get(headers.VSRLooperRequest))
					ctx := looperTransportContext(backendFormat, streaming)
					body := extProcResponseFixture(backendFormat)
					contentType := "application/json"
					if streaming {
						body = extProcStreamFixture(backendFormat)
						contentType = "text/event-stream"
					}
					router := &OpenAIRouter{}
					headerResponse := router.handleLooperResponseHeaders(looperTransportHeaders(200, contentType), ctx)
					require.Equal(t, streaming, ctx.IsStreamingResponse)
					if streaming {
						require.NotNil(t, headerResponse.ModeOverride)
					}
					if backendFormat != llmprotocol.OpenAIChatV1 {
						require.Contains(t, headerResponse.GetResponseHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders(), "content-length")
					}
					w.Header().Set("Content-Type", contentType)
					step := len(body)
					if streaming {
						step = 7
					} // Split inside both SSE frames and JSON tokens.
					for offset := 0; offset < len(body); offset += step {
						end := min(offset+step, len(body))
						chunk := body[offset:end]
						response := router.handleLooperResponseBody(chunk, end == len(body), ctx)
						require.Nil(t, response.GetImmediateResponse())
						if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
							chunk = mutation.GetBody()
						}
						_, _ = w.Write(chunk)
					}
				}))
				defer server.Close()
				client := looper.NewClient(&config.LooperConfig{Endpoint: server.URL})
				defer client.Close()
				request := openai.ChatCompletionNewParams{Model: "public-model", Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")}}
				response, err := client.CallModel(context.Background(), &request, "public-model", streaming, 1, nil, "")
				require.NoError(t, err)
				require.Equal(t, "hello", response.Content)
				require.Equal(t, looper.TokenUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}, response.Usage)
			})
		}
	}
}

func TestLooperInternalResponsePreservesHTTPFailure(t *testing.T) {
	for _, backendFormat := range extProcMatrixFormats {
		t.Run(string(backendFormat), func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := looperTransportContext(backendFormat, true)
			router.handleLooperResponseHeaders(looperTransportHeaders(429, "application/json"), ctx)
			require.Equal(t, 429, ctx.UpstreamStatusCode)
			require.False(t, ctx.IsStreamingResponse)
			response := router.handleLooperResponseBody(extProcTransportErrorFixture(backendFormat), true, ctx)
			require.Nil(t, response.GetImmediateResponse())
			require.Contains(t, string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()), `"error"`)
			require.Nil(t, ctx.SemanticResponse, "HTTP errors cannot become successful completions")
		})
	}
}

func TestLooperInternalResponsePreservesStreamingFailure(t *testing.T) {
	for _, backendFormat := range extProcMatrixFormats {
		t.Run(string(backendFormat), func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := looperTransportContext(backendFormat, true)
			router.handleLooperResponseHeaders(looperTransportHeaders(200, "text/event-stream"), ctx)
			// A normal EOF after the first native frame is an incomplete response.
			body := extProcStreamFixture(backendFormat)
			response := router.handleLooperResponseBody(body[:len(body)/2], true, ctx)
			require.Nil(t, response.GetImmediateResponse())
			translated := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
			require.Contains(t, translated, `"error"`)
			require.NotContains(t, translated, "[DONE]")
			require.Nil(t, ctx.SemanticResponse)
		})
	}
}

func looperTransportContext(backend llmprotocol.WireFormat, streaming bool) *RequestContext {
	includeUsage := true
	return &RequestContext{
		LooperRequest: true, SourceFormat: llmprotocol.OpenAIChatV1, TargetFormat: backend,
		RequestModel: "public-model", TraceContext: context.Background(),
		SemanticRequest: &llmprotocol.Request{
			Generation: 1, Model: "public-model", Stream: streaming,
			StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
		},
	}
}

func looperTransportHeaders(status int, contentType string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: fmt.Sprint(status)},
			{Key: "content-type", Value: contentType},
		}}},
	}
}

func TestLooperInternalEmptyChatRetainsAccountingOverHTTP(t *testing.T) {
	body := []byte(`{"object":"chat.completion","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		router := &OpenAIRouter{}
		ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
		router.handleLooperResponseHeaders(looperTransportHeaders(200, "application/json"), ctx)
		response := router.handleLooperResponseBody(body, true, ctx)
		require.Nil(t, response.GetImmediateResponse())
		require.Nil(t, response.GetResponseBody().GetResponse().GetBodyMutation())
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	defer server.Close()
	client := looper.NewClient(&config.LooperConfig{Endpoint: server.URL})
	defer client.Close()
	request := openai.ChatCompletionNewParams{Model: "public-model", Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")}}
	response, err := client.CallModel(context.Background(), &request, "public-model", false, 1, nil, "")
	require.NoError(t, err)
	require.Empty(t, response.Content)
	require.Equal(t, looper.TokenUsage{PromptTokens: 20, CompletionTokens: 2, TotalTokens: 22}, response.Usage)
}
