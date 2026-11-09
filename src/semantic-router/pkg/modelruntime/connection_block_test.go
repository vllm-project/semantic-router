package modelruntime

import (
	"context"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestRerankDeploymentConnectsThroughItsConnectionBlock(t *testing.T) {
	gateway := connectortest.New(t, map[string]http.HandlerFunc{
		"/rerank": func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"results":[{"index":1,"relevance_score":0.2},{"index":0,"relevance_score":0.9}]}`))
		},
	})
	spec := config.ResolvedModelBinding{Binding: config.ModelBinding{Deployment: "ranker"}, Deployment: gateway.Deployment(config.RerankProvider, "bge-reranker-v2-m3")}
	reranker, err := newRemoteReranker(spec)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reranker.Close() })
	scores, err := reranker.ScorePairs(context.Background(), "", []tasks.QueryDocument{
		{Query: "router config", Document: "How to bind a signal"},
		{Query: "router config", Document: "A recipe for bread"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores.Scores) != 2 || scores.Scores[0] != 0.9 || scores.Scores[1] != 0.2 {
		t.Fatalf("scores follow the documents' order: %+v", scores)
	}
	if calls := gateway.Calls(); len(calls) != 1 || calls[0].Path != connectortest.Prefix+"/rerank" {
		t.Fatalf("calls: %+v", calls)
	}
}
