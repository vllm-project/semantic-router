package extproc

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

const contextQuery = "how long do refunds take?"

// contextRouter answers RAG from the vector store fixture (the refund chunk
// is the only one at least 0.3 similar to every query) and memory from one
// stored memory.
func contextRouter(t *testing.T) (*OpenAIRouter, string) {
	t.Helper()
	router, storeID := ragVectorStoreFixture(t, &plainRAGEmbedder{})
	router.Config = &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Backend: "milvus"}}
	router.MemoryStore = &receiptMemoryStore{results: []*memory.RetrieveResult{{
		Memory: &memory.Memory{Content: "The user prefers short answers."}, Score: 0.9,
	}}}
	return router, storeID
}

// withDecisionRuntime makes a fake model runtime the decision model.
func withDecisionRuntime(t *testing.T, router *OpenAIRouter) *runtimetest.Runtime {
	t.Helper()
	runtime := runtimetest.New(runtimetest.Model{ID: "vela"})
	server := httptest.NewServer(runtime.Handler())
	t.Cleanup(server.Close)
	deployments := map[string]config.ModelDeployment{"vela": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL}}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(deployments)
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	card, err := lease.Card(ctx, "vela")
	require.NoError(t, err)
	router.Config.DecisionModel, router.Config.ModelDeployments = "vela", deployments
	router.decisionDecider = lease
	router.decisionCards = map[string]modelservice.ModelCard{"vela": card}
	return runtime
}

func contextRequestContext(plugin config.ContextPluginConfig) (*RequestContext, *llmprotocol.Request) {
	request := testNeutralRequest("entrypoint", contextQuery)
	decision := &config.Decision{Name: "docs", Plugins: []config.DecisionPlugin{
		{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(plugin)},
	}}
	return &RequestContext{
		Headers:                 map[string]string{"x-authz-user-id": "user-1"},
		TraceContext:            context.Background(),
		UserContent:             contextQuery,
		SemanticRequest:         request,
		VSRSelectedDecision:     decision,
		VSRSelectedDecisionName: decision.Name,
	}, request
}

func ragSource(storeID string, match *config.MatchConfig) config.ContextSource {
	return config.ContextSource{
		Type:          config.ContextSourceRAG,
		Backend:       "vectorstore",
		BackendConfig: config.MustStructuredPayload(&config.VectorStoreRAGConfig{VectorStoreID: storeID}),
		Match:         match,
	}
}

func recallOnly(topK int) *config.MatchConfig {
	minSimilarity := 0.3
	return &config.MatchConfig{Recall: &config.MatchRecall{Method: config.MatchRecallEmbedding, TopK: topK, MinSimilarity: &minSimilarity}}
}

// toolExchanges are the names of the tool calls injected into the
// conversation, in order.
func toolExchanges(request *llmprotocol.Request) []string {
	var names []string
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				names = append(names, content.ToolCall.Name)
			}
		}
	}
	return names
}

// A stage's relevance questions, one per candidate, go to the decision model
// in one decisions call: a RAG source's default match asks about its recalled
// documents, compression about its segments, and a memory source's default
// match recalls without asking.
func TestContextPluginAsksOneDecisionsCallPerStage(t *testing.T) {
	router, storeID := contextRouter(t)
	runtime := withDecisionRuntime(t, router)
	plugin := config.ContextPluginConfig{Enabled: true, Sources: []config.ContextSource{
		ragSource(storeID, nil), {Type: config.ContextSourceMemory},
	}}
	ctx, request := contextRequestContext(plugin)

	require.NoError(t, router.retrieveContextSources(ctx, contextQuery, request))
	calls := runtime.Decisions()
	require.Len(t, calls, 1, "the RAG judge asks about its one recalled document; memory recalls only")
	require.Equal(t, map[string]any{"request": contextQuery, "candidate": "refunds take 14 business days"}, calls[0].State)

	segments := []string{"the refund window is 30 days", "status updates every 30 minutes", "the office is closed on Sunday"}
	scores, err := router.contextCompressionScorer(&config.ContextCompressConfig{}).Score(context.Background(), contextQuery, segments)
	require.NoError(t, err)
	require.Len(t, scores, len(segments))
	require.Len(t, runtime.Decisions(), 2, "compression asks about its segments in one call")

	memoryJudged := config.ContextPluginConfig{Enabled: true, Sources: []config.ContextSource{{
		Type:  config.ContextSourceMemory,
		Match: &config.MatchConfig{Recall: recallOnly(5).Recall, Judge: &config.MatchJudge{Preset: "relevance"}},
	}}}
	ctx, request = contextRequestContext(memoryJudged)
	require.NoError(t, router.retrieveContextSources(ctx, contextQuery, request))
	require.Len(t, runtime.Decisions(), 3, "a memory source with a judge asks once")
}

// Each source injects one tool exchange after the last user message, in the
// order the plugin lists the sources.
func TestContextSourcesInjectInSourceOrder(t *testing.T) {
	for _, tt := range []struct {
		name    string
		sources func(storeID string) []config.ContextSource
		want    []string
	}{
		{name: "rag first", sources: func(storeID string) []config.ContextSource {
			return []config.ContextSource{ragSource(storeID, recallOnly(1)), {Type: config.ContextSourceMemory}}
		}, want: []string{ragContextToolName, memoryContextToolName}},
		{name: "memory first", sources: func(storeID string) []config.ContextSource {
			return []config.ContextSource{{Type: config.ContextSourceMemory}, ragSource(storeID, recallOnly(1))}
		}, want: []string{memoryContextToolName, ragContextToolName}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			router, storeID := contextRouter(t)
			ctx, request := contextRequestContext(config.ContextPluginConfig{Enabled: true, Sources: tt.sources(storeID)})
			require.NoError(t, router.retrieveContextSources(ctx, contextQuery, request))
			require.Equal(t, tt.want, toolExchanges(request))
			require.Equal(t, contextQuery, request.Messages[0].Content[0].Text)
			require.Len(t, ctx.RAGToolCallIDs, 1)
			require.Len(t, ctx.MemoryToolCallIDs, 1)
		})
	}
}

// system_prompt injection leaves the conversation alone.
func TestContextSystemPromptInjection(t *testing.T) {
	router, storeID := contextRouter(t)
	ctx, request := contextRequestContext(config.ContextPluginConfig{
		Enabled:   true,
		Sources:   []config.ContextSource{ragSource(storeID, recallOnly(1))},
		Injection: &config.ContextInjectionConfig{Mode: config.ContextInjectionSystemPrompt},
	})
	require.NoError(t, router.retrieveContextSources(ctx, contextQuery, request))
	require.Empty(t, toolExchanges(request))
	require.Len(t, request.Messages, 1)
	require.NotEmpty(t, request.Instructions)
	require.Contains(t, request.Instructions[0].Content[0].Text, "refunds take 14 business days")
}

// A failed RAG retrieval refuses the request only with on_failure block.
func TestContextRAGOnFailure(t *testing.T) {
	for _, tt := range []struct {
		onFailure string
		blocks    bool
	}{
		{onFailure: "", blocks: false},
		{onFailure: config.RAGOnFailureSkip, blocks: false},
		{onFailure: config.RAGOnFailureWarn, blocks: false},
		{onFailure: config.RAGOnFailureBlock, blocks: true},
	} {
		t.Run("on_failure="+tt.onFailure, func(t *testing.T) {
			router, _ := contextRouter(t)
			source := ragSource("vs-missing", recallOnly(1))
			source.OnFailure = tt.onFailure
			ctx, request := contextRequestContext(config.ContextPluginConfig{Enabled: true, Sources: []config.ContextSource{source, {Type: config.ContextSourceMemory}}})
			err := router.retrieveContextSources(ctx, contextQuery, request)
			if tt.blocks {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, []string{memoryContextToolName}, toolExchanges(request), "memory still runs after a skipped RAG failure")
		})
	}
}
