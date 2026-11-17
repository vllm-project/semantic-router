package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// vLLM 0.33's answer to a prompt longer than its window.
const vllmContextLengthError = `{"error":{"message":"This model's maximum context length is 1048576 tokens. ` +
	`However, your prompt contains at least 1048577 input tokens. Please reduce the length of the input. ` +
	`(parameter=input_tokens, value=1048577)","type":"BadRequestError","param":"input_tokens","code":400}}`

// Envoy's local reply, and pkg/upstream's, when a backend refuses the
// connection.
const connectRefusedReply = "upstream connect error or disconnect/reset before headers. reset reason: " +
	"remote connection failure, transport failure reason: delayed connect error: 111"

type clientError struct {
	Type  string `json:"type"`
	Error struct {
		Type    string  `json:"type"`
		Code    *string `json:"code"`
		Message string  `json:"message"`
		Param   *string `json:"param"`
	} `json:"error"`
}

func decodeClientError(t *testing.T, body []byte) clientError {
	t.Helper()
	var decoded clientError
	require.NoError(t, json.Unmarshal(body, &decoded), "%s", body)
	return decoded
}

func requireContextLengthExceeded(t *testing.T, status int, body []byte) {
	t.Helper()
	require.Equal(t, http.StatusBadRequest, status, "%s", body)
	decoded := decodeClientError(t, body)
	require.Equal(t, "invalid_request_error", decoded.Error.Type)
	require.NotNil(t, decoded.Error.Code, "%s", body)
	require.Equal(t, "context_length_exceeded", *decoded.Error.Code)
	require.Contains(t, decoded.Error.Message, "maximum context length is 1048576 tokens")
}

// A backend's refusal of a prompt over its window reaches the client with
// OpenAI's code for it, not the backend's "400" (#6320).
func TestBackendContextLengthRefusalIsContextLengthExceeded(t *testing.T) {
	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary"})
	ctx.UpstreamStatusCode = http.StatusBadRequest

	resp := router.handleUpstreamTransportError([]byte(vllmContextLengthError), ctx)
	require.Nil(t, resp.GetImmediateResponse())
	requireContextLengthExceeded(t, http.StatusBadRequest, resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
}

// A fallback chain that ends without a success answers with its last
// attempt: a candidate's 400 stays a 400, and the first model's transport
// failure stays out of the client's message (#6322).
func TestFallbackAnswersWithTheLastAttempt(t *testing.T) {
	eligible := []string{"model-primary", "model-fallback-1"}

	t.Run("response phase, the candidate refuses the prompt's length", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
			return []byte(vllmContextLengthError), http.StatusBadRequest, nil
		}
		resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"The server is temporarily unavailable","type":"server_error"}}`), ctx)
		immediate := resp.GetImmediateResponse()
		require.NotNil(t, immediate, "the client got the first model's 503")
		requireContextLengthExceeded(t, int(immediate.GetStatus().GetCode()), immediate.GetBody())
		require.Equal(t, "candidates_exhausted", ctx.FallbackRecord.FinalStatus, "a refusal for length goes on to wider windows only")
	})

	t.Run("response phase, no candidate backend answers", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
			return nil, 0, errors.New("dial tcp 127.0.0.1:8002: connect: connection refused")
		}
		resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"The server is temporarily unavailable","type":"server_error"}}`), ctx)
		immediate := resp.GetImmediateResponse()
		require.NotNil(t, immediate)
		require.EqualValues(t, http.StatusServiceUnavailable, immediate.GetStatus().GetCode())
		decoded := decodeClientError(t, immediate.GetBody())
		require.Equal(t, "upstream_unavailable", *decoded.Error.Code)
		require.NotContains(t, string(immediate.GetBody()), "connection refused")
	})

	t.Run("native gateway, the candidate refuses the prompt's length", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), eligible...)
		chain := session.Fallback()
		step, err := chain.Next(context.Background(), routing.Outcome{
			Route: "model-primary", Status: http.StatusServiceUnavailable, Local: true,
			Failure: "connect_failure", Body: []byte(connectRefusedReply), Duration: time.Millisecond,
		})
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		step, err = chain.Next(context.Background(), routing.Outcome{
			Route: "model-fallback-1", Status: http.StatusBadRequest,
			Header: routing.Header{{Name: "content-type", Value: "application/json"}},
			Body:   []byte(vllmContextLengthError), Duration: time.Millisecond,
		})
		require.NoError(t, err)
		require.Nil(t, step.Call)
		require.NotNil(t, step.Immediate, "the client got the first model's local reply")
		requireContextLengthExceeded(t, step.Immediate.Status, step.Immediate.Body)
		require.NotContains(t, string(step.Immediate.Body), "delayed connect error")
		require.Len(t, session.ctx.FallbackRecord.Attempts, 2)
	})

	t.Run("native gateway, no candidate backend answers", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), eligible...)
		chain := session.Fallback()
		local := routing.Outcome{
			Route: "model-primary", Status: http.StatusServiceUnavailable, Local: true,
			Failure: "connect_failure", Body: []byte(connectRefusedReply), Duration: time.Millisecond,
		}
		step, err := chain.Next(context.Background(), local)
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		local.Route = "model-fallback-1"
		step, err = chain.Next(context.Background(), local)
		require.NoError(t, err)
		require.NotNil(t, step.Immediate)
		require.Equal(t, http.StatusServiceUnavailable, step.Immediate.Status)
		require.Equal(t, "upstream_unavailable", *decodeClientError(t, step.Immediate.Body).Error.Code)
		require.NotContains(t, string(step.Immediate.Body), "delayed connect error")
	})

	t.Run("an Anthropic client gets its own error shape", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
			return []byte(vllmContextLengthError), http.StatusBadRequest, nil
		}
		resp := router.handleUpstreamTransportError([]byte(`{"error":{"message":"unavailable","type":"server_error"}}`), ctx)
		immediate := resp.GetImmediateResponse()
		require.NotNil(t, immediate)
		require.EqualValues(t, http.StatusBadRequest, immediate.GetStatus().GetCode())
		decoded := decodeClientError(t, immediate.GetBody())
		require.Equal(t, "error", decoded.Type)
		require.Equal(t, "invalid_request_error", decoded.Error.Type)
		require.Contains(t, decoded.Error.Message, "maximum context length is 1048576 tokens")
	})
}
