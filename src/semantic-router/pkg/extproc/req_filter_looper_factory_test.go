package extproc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleLooperExecutionRejectsUnknownAlgorithmBeforeUpstream(t *testing.T) {
	var upstreamCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		http.Error(w, "unexpected upstream request", http.StatusInternalServerError)
	}))
	defer server.Close()

	router := &OpenAIRouter{
		Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: server.URL}},
	}
	decision := &config.Decision{
		Name: "unknown-looper",
		ModelRefs: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
		Algorithm: &config.AlgorithmConfig{Type: "unregistered"},
	}
	request := &openai.ChatCompletionNewParams{
		Model:    "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
	}

	response, err := router.handleLooperExecution(
		context.Background(),
		request,
		decision,
		&RequestContext{RequestID: "unknown-looper-test"},
	)

	require.NoError(t, err)
	require.NotNil(t, response.GetImmediateResponse())
	assert.Equal(t, 500, int(response.GetImmediateResponse().GetStatus().GetCode()))
	assert.Contains(t, string(response.GetImmediateResponse().GetBody()), "unsupported Looper algorithm")
	assert.Zero(t, upstreamCalls.Load(), "unknown algorithm must fail before any upstream request")
}
