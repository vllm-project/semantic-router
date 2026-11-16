package embedding

import (
	"context"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

func TestOpenAIEmbeddingsDeploymentConnectsThroughItsConnectionBlock(t *testing.T) {
	gateway := connectortest.New(t, map[string]http.HandlerFunc{
		"/embeddings": func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","model":"bge-m3","data":[{"object":"embedding","index":0,"embedding":[0.6,0.8]}]}`))
		},
	})
	deployment := gateway.Deployment(config.OpenAIEmbeddingsProvider, "bge-m3")
	provider, err := NewOpenAICompatibleProvider(OpenAICompatibleConfig{BaseURL: deployment.Endpoint, Model: deployment.Model, Deployment: &deployment})
	if err != nil {
		t.Fatal(err)
	}
	vector, err := provider.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(vector) != 2 {
		t.Fatalf("vector: %v", vector)
	}
	if calls := gateway.Calls(); len(calls) != 1 || calls[0].Path != connectortest.Prefix+"/embeddings" {
		t.Fatalf("calls: %+v", calls)
	}
}
