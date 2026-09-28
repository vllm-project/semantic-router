package extproc

import (
	"context"
	"errors"
	"path/filepath"
	"runtime"
	"testing"
	"time"

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
		Headers:                 map[string]string{"x-authz-user-id": "user-1"},
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
		Headers:         map[string]string{"x-authz-user-id": "user-1"},
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

type thresholdRecordingMemoryStore struct {
	noopMemoryStore
	threshold float32
}

func (store *thresholdRecordingMemoryStore) Retrieve(_ context.Context, opts memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	store.threshold = opts.Threshold
	return nil, nil
}

func TestMemoryRetrievalThresholdFallsBackToConfigDefault(t *testing.T) {
	configDefault := config.DefaultCanonicalGlobal().Stores.Memory.DefaultSimilarityThreshold
	cases := []struct {
		name   string
		memory string
		want   float32
	}{
		{name: "omitted", memory: "{enabled: true}", want: configDefault},
		{name: "zero means unset", memory: "{enabled: true, default_similarity_threshold: 0}", want: configDefault},
		{name: "configured", memory: "{enabled: true, default_similarity_threshold: 0.74}", want: 0.74},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte("version: v0.3\nglobal:\n  stores:\n    memory: " + tc.memory + "\n"))
			require.NoError(t, err)
			store := &thresholdRecordingMemoryStore{}
			router := &OpenAIRouter{Config: cfg, MemoryStore: store}
			query := "What is my sister's name?"
			request := testNeutralRequest("entrypoint", query)
			ctx := &RequestContext{
				Headers:         map[string]string{"x-authz-user-id": "user-1"},
				TraceContext:    context.Background(),
				SemanticRequest: request,
			}

			require.NoError(t, router.handleMemoryRetrieval(ctx, query, request))
			assert.Equal(t, tc.want, store.threshold)
		})
	}
}

// The reference configuration is copied into the Router image. A low-scoring
// answer must survive its calibrated threshold, while an unrelated query must
// not inject a fact. Scores match the weighted-hybrid cold-start replay in
// #4228 for the pinned mom-embedding-light model.
type referenceScoredMemoryStore struct {
	noopMemoryStore
	options []memory.RetrieveOptions
}

func (store *referenceScoredMemoryStore) Retrieve(_ context.Context, opts memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	store.options = append(store.options, opts)
	var score float32
	var content string
	switch opts.Query {
	case "What is my dog's name?":
		score, content = 0.576, "My dog is a beagle named Biscuit."
	case "Which programming language am I learning?":
		score, content = 0.424, "I'm learning Rust for a side project."
	case "What is my sister's name?":
		score = 0.380
		content = "My dog is a beagle named Biscuit."
	default:
		return nil, errors.New("unexpected reference memory query")
	}
	if score < opts.Threshold {
		return nil, nil
	}
	return []*memory.RetrieveResult{{
		Memory: &memory.Memory{Content: content, CreatedAt: time.Now()},
		Score:  score,
	}}, nil
}

func TestReferenceMemoryCalibratedRequestPath(t *testing.T) {
	_, sourceFile, _, ok := runtime.Caller(0)
	require.True(t, ok)
	configPath := filepath.Join(filepath.Dir(sourceFile), "../../../../config/config.yaml")
	cfg, err := config.Load(configPath)
	require.NoError(t, err)
	require.Equal(t, "bert", cfg.Memory.EmbeddingModel)
	require.Equal(t, float32(0.40), cfg.Memory.DefaultSimilarityThreshold)
	require.Equal(t, "weighted", cfg.Memory.HybridMode)
	require.Equal(t, "heuristic", cfg.Memory.Reflection.Algorithm)

	decision := cfg.GetDecisionByName("computer-science-remom-route")
	require.NotNil(t, decision)
	plugin := decision.GetMemoryConfig()
	require.NotNil(t, plugin)
	require.NotNil(t, plugin.SimilarityThreshold)
	require.Equal(t, float32(0.40), *plugin.SimilarityThreshold)
	require.Equal(t, "weighted", plugin.HybridMode)
	require.NotNil(t, plugin.Reflection)
	require.Equal(t, "heuristic", plugin.Reflection.Algorithm)

	for _, tc := range []struct {
		name     string
		decision *config.Decision
	}{
		{name: "global"},
		{name: "decision override", decision: decision},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := &referenceScoredMemoryStore{}
			router := &OpenAIRouter{Config: cfg, MemoryStore: store}
			for _, query := range []struct {
				text       string
				wantStatus string
				wantFact   string
			}{
				{text: "What is my dog's name?", wantStatus: "used", wantFact: "Biscuit"},
				{text: "Which programming language am I learning?", wantStatus: "used", wantFact: "Rust"},
				{text: "What is my sister's name?", wantStatus: "missing"},
			} {
				request := testNeutralRequest("entrypoint", query.text)
				ctx := &RequestContext{
					Headers:             map[string]string{"x-authz-user-id": "memory-calibration-user"},
					TraceContext:        context.Background(),
					VSRSelectedDecision: tc.decision,
					SemanticRequest:     request,
				}
				require.NoError(t, router.handleMemoryRetrieval(ctx, query.text, request))
				assert.Equal(t, query.wantStatus, ctx.MemoryStatus)
				if query.wantStatus == "used" {
					assert.Contains(t, ctx.MemoryContext, query.wantFact)
				} else {
					assert.Empty(t, ctx.MemoryContext)
				}
			}
			require.Len(t, store.options, 3)
			for _, opts := range store.options {
				assert.Equal(t, float32(0.40), opts.Threshold)
				assert.True(t, opts.HybridSearch)
				assert.Equal(t, "weighted", opts.HybridMode)
				assert.True(t, opts.AdaptiveThreshold)
			}
		})
	}
}
