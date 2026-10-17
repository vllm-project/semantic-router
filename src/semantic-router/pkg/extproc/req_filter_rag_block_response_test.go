package extproc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ragBlockSentinel stands in for the private detail a failing retrieval
// backend puts in its error response body: backend identity, transport
// detail, or any other upstream content. In on_failure=block mode that
// detail belongs in the server log only, never in the client 503 (#4717).
const ragBlockSentinel = "rag-block-sentinel-upstream-detail"

func ragBlockFailingBackend(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(ragBlockSentinel))
	}))
	t.Cleanup(server.Close)
	return server
}

func ragBlockDecision(endpoint string) config.Decision {
	return config.Decision{
		Name:     "rag-block-route",
		Priority: 1,
		Plugins: []config.DecisionPlugin{{
			Type: "rag",
			Configuration: config.MustStructuredPayload(config.RAGPluginConfig{
				Enabled:   true,
				Backend:   "external_api",
				OnFailure: "block",
				BackendConfig: config.MustStructuredPayload(&config.ExternalAPIRAGConfig{
					Endpoint:        endpoint,
					RequestFormat:   "custom",
					RequestTemplate: `{"query":"{{.Query}}"}`,
				}),
			}),
		}},
	}
}

func ragBlockEmbeddings() *embedding.Set {
	return embedding.NewSet(
		map[string]embedding.Provider{config.RAGQueryEmbeddingModel: &plainRAGEmbedder{}},
		config.RAGQueryEmbeddingModel,
	)
}

func requireGenericRAGBlockResponse(t *testing.T, body string) {
	t.Helper()
	require.Contains(t, body, "RAG retrieval failed")
	require.NotContains(t, body, ragBlockSentinel)
	require.NotContains(t, body, "API returned status")
}

func TestRAGBlockModePreRoutingResponseOmitsRetrievalErrorChain(t *testing.T) {
	server := ragBlockFailingBackend(t)

	router, chatModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Embeddings = ragBlockEmbeddings()
	decision := ragBlockDecision(server.URL)
	decision.ModelRefs = []config.ModelRef{{Model: chatModel}}
	router.Config.Decisions = []config.Decision{decision}

	classifier, err := classification.NewClassifier(router.Config, nil, nil, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = classifier.Close() })
	router.Classifier = classifier

	request := testNeutralRequest("auto", "how long does a refund take")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.RequestModel = "auto"

	_, response := router.runRequestPreRoutingStages("auto", extractSemanticRequestSignals(request), ctx)
	require.NotNil(t, response)
	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate)
	require.EqualValues(t, 503, immediate.GetStatus().GetCode())
	requireGenericRAGBlockResponse(t, string(immediate.GetBody()))
}

func TestRAGBlockModeLooperResponseOmitsRetrievalErrorChain(t *testing.T) {
	server := ragBlockFailingBackend(t)

	decision := ragBlockDecision(server.URL)
	router := &OpenAIRouter{
		Config:     &config.RouterConfig{},
		Embeddings: ragBlockEmbeddings(),
	}
	ctx := &RequestContext{
		Headers:             map[string]string{},
		TraceContext:        context.Background(),
		UserContent:         "how long does a refund take",
		VSRSelectedDecision: &decision,
	}

	response := router.runLooperInternalPlugins(ctx, decision.Name)
	require.NotNil(t, response)
	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate)
	require.EqualValues(t, 503, immediate.GetStatus().GetCode())
	requireGenericRAGBlockResponse(t, string(immediate.GetBody()))
}
