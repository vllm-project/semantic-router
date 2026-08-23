package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type receiptMemoryStore struct {
	noopMemoryStore
	results []*memory.RetrieveResult
	err     error
}

func (store receiptMemoryStore) Retrieve(context.Context, memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return store.results, store.err
}

func TestConcreteModelBypassesMemoryPlugin(t *testing.T) {
	router := &OpenAIRouter{
		Config:      &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Backend: "milvus"}},
		MemoryStore: &receiptMemoryStore{err: errors.New("must not be called")},
	}
	request := testNeutralRequest("concrete-model", "hello")
	ctx := &RequestContext{SemanticRequest: request}
	ctx.Routing.SelectPassthrough()

	require.NoError(t, router.handleMemoryRetrieval(ctx, "hello", request))
	assert.Len(t, request.Messages, 1)
	assert.Empty(t, ctx.MemoryBackend)
}

func TestExtractConversationHistoryUsesNeutralMessages(t *testing.T) {
	router := &OpenAIRouter{}
	request := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "Hello"},
			{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"},
		}},
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Hi there!"}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult}}},
	}}

	history := router.extractConversationHistory(request)
	if len(history) != 2 {
		t.Fatalf("conversation history = %#v", history)
	}
	if history[0].Role != "user" || history[0].Content != "Hello" ||
		history[1].Role != "assistant" || history[1].Content != "Hi there!" {
		t.Fatalf("conversation history = %#v", history)
	}
}

func TestMemoryRuntimeReceiptRecordsFailOpenRetrievalError(t *testing.T) {
	router := &OpenAIRouter{
		Config:      &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Backend: "milvus"}},
		MemoryStore: &receiptMemoryStore{err: errors.New("backend unavailable secret-canary")},
	}
	request := testNeutralRequest("entrypoint", "What did I say?")
	ctx := &RequestContext{
		InferenceAccess:         testInferenceRequestAccess("user-1", ""),
		TraceContext:            context.Background(),
		VSRSelectedDecisionName: "balance",
		SemanticRequest:         request,
	}
	before := testutil.ToFloat64(metrics.PluginExecutionTotal.WithLabelValues("memory", "balance", "unavailable"))

	err := router.handleMemoryRetrieval(ctx, "What did I say?", request)
	require.ErrorContains(t, err, "memory retrieval failed")
	require.NotContains(t, err.Error(), "secret-canary")
	assert.Len(t, request.Messages, 1)
	assert.Equal(t, before+1, testutil.ToFloat64(metrics.PluginExecutionTotal.WithLabelValues("memory", "balance", "unavailable")))
	diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "balance", 0, 0)
	assert.Equal(t, "milvus", diagnostics.MemoryBackend)
	assert.Equal(t, "unavailable", diagnostics.MemoryStatus)
	assert.Equal(t, "retrieval_error", diagnostics.MemoryReason)
	assert.True(t, diagnostics.MemoryFailOpen)
}

func TestMemoryRuntimeInjectsNeutralMessage(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Backend: "milvus"}},
		MemoryStore: &receiptMemoryStore{results: []*memory.RetrieveResult{{
			Memory: &memory.Memory{Content: "The user's deadline is Friday."}, Score: 0.9,
		}}},
	}
	request := testNeutralRequest("entrypoint", "What is my deadline?")
	request.Instructions = []llmprotocol.InstructionBlock{{
		Role:    llmprotocol.RoleDeveloper,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Answer concisely."}},
	}}
	ctx := &RequestContext{
		InferenceAccess: testInferenceRequestAccess("user-1", ""),
		TraceContext:    context.Background(),
		SemanticRequest: request,
	}

	require.NoError(t, router.handleMemoryRetrieval(ctx, "What is my deadline?", request))
	require.Len(t, request.Messages, 2)
	assert.Equal(t, llmprotocol.RoleUser, request.Messages[0].Role)
	assert.Contains(t, request.Messages[0].Content[0].Text, "deadline is Friday")
	assert.Equal(t, "What is my deadline?", request.Messages[1].Content[0].Text)
	assert.Equal(t, llmprotocol.RoleDeveloper, request.Instructions[0].Role)
	assert.Equal(t, uint64(2), request.Generation)
	assert.Contains(t, ctx.MemoryMessageIndexes, 0)
	diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "model-a", "balance", 0, 0)
	assert.Equal(t, "used", diagnostics.MemoryStatus)
	assert.Equal(t, "injected", diagnostics.MemoryReason)
	assert.Equal(t, 1, diagnostics.MemoryResultCount)
	assert.False(t, diagnostics.MemoryFailOpen)
}
