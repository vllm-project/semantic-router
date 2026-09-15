//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestOwnedStandaloneEmbeddingLeaseDrainsAcrossHTTPForwards(t *testing.T) {
	entered, unblock := make(chan struct{}), make(chan struct{})
	var unblockOnce sync.Once
	releaseHTTP := func() { unblockOnce.Do(func() { close(unblock) }) }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var input struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
			t.Error(err)
			return
		}
		if len(input.Input) == 1 && input.Input[0] == "first request" {
			close(entered)
			<-unblock
		}
		data := make([]map[string]any, len(input.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": []float32{1, 0}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	t.Cleanup(func() { releaseHTTP(); server.Close() })
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "remote"
	cfg.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
	cfg.EmbeddingConfig.TargetDimension = 2
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "active", Candidates: []string{"candidate"}}}
	cfg.Decisions = []config.Decision{{Name: "active", Rules: config.RuleNode{Type: "embedding", Name: "active"}}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "service", Contract: "embedding.v1", Adapter: "openai_compatible"}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"service": {Provider: "http", ExternalModel: "embedder"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "embedder", ModelName: "embedder", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: server.URL}}}
	service := services.NewClassificationService(nil, cfg)
	t.Cleanup(func() { _ = service.Close() })
	if err := service.TryRefreshRuntimeConfig(cfg); err != nil {
		t.Fatal(err)
	}
	api := &ClassificationAPIServer{classificationSvc: service, config: cfg}
	snapshotConfig, prepared, release, err := api.acquireEmbeddingRuntime()
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if snapshotConfig != service.GetConfig() {
		t.Fatal("configuration and classifier were not acquired from one snapshot")
	}
	provider, err := prepared.Default()
	if err != nil {
		t.Fatal(err)
	}
	called := make(chan error, 1)
	go func() {
		_, callErr := provider.Embed(context.Background(), "first request")
		called <- callErr
	}()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("real HTTP embedding did not begin")
	}
	refreshed := make(chan error, 1)
	go func() { refreshed <- service.TryRefreshRuntimeConfig(&config.RouterConfig{}) }()
	releaseHTTP()
	if callErr := <-called; callErr != nil {
		t.Fatal(callErr)
	}
	// The first native/HTTP Resource.Use is now finished. The API operation's
	// lease must still protect the gap before its next forward.
	select {
	case refreshErr := <-refreshed:
		t.Fatalf("refresh retired a leased operation between forwards: %v", refreshErr)
	case <-time.After(40 * time.Millisecond):
	}
	if _, callErr := provider.Embed(context.Background(), "second request"); callErr != nil {
		t.Fatalf("second forward lost its prepared owner: %v", callErr)
	}
	release()
	select {
	case refreshErr := <-refreshed:
		if refreshErr != nil {
			t.Fatal(refreshErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("refresh did not complete after the API operation released")
	}
	if _, callErr := provider.Embed(context.Background(), "retired"); !errors.Is(callErr, binding.ErrClosed) {
		t.Fatalf("retired provider is still callable: %v", callErr)
	}
	newConfig, replacement, releaseReplacement, err := api.acquireEmbeddingRuntime()
	defer releaseReplacement()
	if err != nil || newConfig == snapshotConfig || replacement == prepared || replacement.Ready() {
		t.Fatalf("replacement did not expose its own empty snapshot: %v", err)
	}
}
