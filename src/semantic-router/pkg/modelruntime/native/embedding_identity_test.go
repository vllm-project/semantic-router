package native

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestOwnedRemoteEmbeddingIdentityIncludesClientAndVectorContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Dimensions int `json:"dimensions"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		vector := make([]float32, request.Dimensions)
		vector[0] = 1
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"index": 0, "embedding": vector}}})
	}))
	defer server.Close()
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return errors.New("redirect forbidden") }}
	cfg := embedding.OpenAICompatibleConfig{BaseURL: server.URL, Model: "embedding", Dimensions: 2, ExpectedDimension: 2, HTTPClient: client}
	spec := config.ResolvedModelBinding{Recipe: "first", Name: "embedding", Binding: config.ModelBinding{Deployment: "embedding", Contract: "embedding.v1", Adapter: "openai_compatible"}, Deployment: config.ModelDeployment{Provider: "http"}}
	runtime := New(binding.NewPool())
	first, err := runtime.RemoteEmbedding(context.Background(), spec, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	spec.Recipe = "alias"
	alias, err := runtime.RemoteEmbedding(context.Background(), spec, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer alias.Close()
	if first.CacheIdentity() != alias.CacheIdentity() {
		t.Fatal("identical custom-client execution did not share")
	}
	cfg.Dimensions, cfg.ExpectedDimension = 3, 3
	changed, err := runtime.RemoteEmbedding(context.Background(), spec, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer changed.Close()
	if first.CacheIdentity() == changed.CacheIdentity() || changed.Dimension() != 3 {
		t.Fatal("vector contract change reused previous HTTP connector")
	}
	cfg.HTTPClient = &http.Client{CheckRedirect: client.CheckRedirect}
	independent, err := runtime.RemoteEmbedding(context.Background(), spec, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer independent.Close()
	if independent.CacheIdentity() == changed.CacheIdentity() {
		t.Fatal("independent HTTP clients collapsed")
	}
	if err = first.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = alias.Embed(context.Background(), "still owned"); err != nil {
		t.Fatal(err)
	}
}
