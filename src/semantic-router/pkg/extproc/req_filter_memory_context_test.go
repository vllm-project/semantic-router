package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/openai/openai-go"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type receiptMemoryStore struct {
	noopMemoryStore
	results []*memory.RetrieveResult
	err     error
}

func (s receiptMemoryStore) Retrieve(context.Context, memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return s.results, s.err
}

func TestMemoryRuntimeReceiptRecordsFailOpenRetrievalError(t *testing.T) {
	router := &OpenAIRouter{
		Config:      &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Backend: "milvus"}},
		MemoryStore: &receiptMemoryStore{err: errors.New("backend unavailable")},
	}
	ctx := &RequestContext{
		Headers:                 map[string]string{headers.AuthzUserID: "user-1"},
		TraceContext:            context.Background(),
		VSRSelectedDecisionName: "balance",
	}
	body := []byte(`{"messages":[{"role":"user","content":"What did I say?"}]}`)
	before := testutil.ToFloat64(metrics.PluginExecutionTotal.WithLabelValues("memory", "balance", "unavailable"))

	got, err := router.handleMemoryRetrieval(ctx, "What did I say?", body, &openai.ChatCompletionNewParams{})

	require.ErrorContains(t, err, "memory retrieval failed")
	assert.Equal(t, body, got)
	assert.Equal(t, before+1, testutil.ToFloat64(metrics.PluginExecutionTotal.WithLabelValues("memory", "balance", "unavailable")))
	diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "balance", 0, 0)
	assert.Equal(t, "milvus", diagnostics.MemoryBackend)
	assert.Equal(t, "unavailable", diagnostics.MemoryStatus)
	assert.Equal(t, "retrieval_error", diagnostics.MemoryReason)
	assert.True(t, diagnostics.MemoryFailOpen)
}

func TestMemoryRuntimeReceiptRecordsUsedMemory(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Backend: "milvus"}},
		MemoryStore: &receiptMemoryStore{results: []*memory.RetrieveResult{{
			Memory: &memory.Memory{Content: "The user's deadline is Friday."},
			Score:  0.9,
		}}},
	}
	ctx := &RequestContext{
		Headers:      map[string]string{headers.AuthzUserID: "user-1"},
		TraceContext: context.Background(),
	}
	body := []byte(`{"messages":[{"role":"user","content":"What is my deadline?"}]}`)

	got, err := router.handleMemoryRetrieval(ctx, "What is my deadline?", body, &openai.ChatCompletionNewParams{})

	require.NoError(t, err)
	assert.NotEqual(t, body, got)
	diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "balance", 0, 0)
	assert.Equal(t, "used", diagnostics.MemoryStatus)
	assert.Equal(t, "injected", diagnostics.MemoryReason)
	assert.Equal(t, 1, diagnostics.MemoryResultCount)
	assert.False(t, diagnostics.MemoryFailOpen)
}

func TestInjectMemoryMessages_InsertsAfterSystemAndDeveloperMessages(t *testing.T) {
	requestBody := []byte(`{
		"messages": [
			{"role": "system", "content": "system instructions"},
			{"role": "developer", "content": "developer instructions"},
			{"role": "user", "content": "hello"}
		]
	}`)

	modified, err := injectMemoryMessages(requestBody, "memory context")
	require.NoError(t, err)

	var request struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(modified, &request))
	require.Len(t, request.Messages, 4)
	assert.Equal(t, "system", request.Messages[0].Role)
	assert.Equal(t, "developer", request.Messages[1].Role)
	assert.Equal(t, "user", request.Messages[2].Role)
	assert.Equal(t, "memory context", request.Messages[2].Content)
	assert.Equal(t, "user", request.Messages[3].Role)
	assert.Equal(t, "hello", request.Messages[3].Content)
}

func TestInjectMemoryMessages_InitializesMissingMessages(t *testing.T) {
	modified, err := injectMemoryMessages([]byte(`{"model":"test-model"}`), "memory context")
	require.NoError(t, err)

	var request struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(modified, &request))
	require.Len(t, request.Messages, 1)
	assert.Equal(t, "user", request.Messages[0].Role)
	assert.Equal(t, "memory context", request.Messages[0].Content)
}
