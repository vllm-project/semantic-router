package modelruntime

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func TestOwnedEmbeddingSkipsUnusedCatalogArtifacts(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Qwen3ModelPath = "/does-not-exist/qwen3"
	cfg.MmBertModelPath = "/does-not-exist/mmbert"
	cfg.KnowledgeBases = []config.KnowledgeBaseConfig{{Name: "unused catalog knowledge base"}}
	prepared, err := PrepareOwnedEmbeddings(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	if prepared.Ready() {
		t.Fatal("unused catalog was provisioned")
	}
}

func TestOwnedRemoteEmbeddingIndependentReferencesAndFailedCandidate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		if request.Model == "invalid" {
			_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[]}]}`))
			return
		}
		vector := []float64{1, 0}
		if request.Model == "second" {
			vector = []float64{0, 1}
		}
		data := make([]map[string]interface{}, len(request.Input))
		for i := range data {
			data[i] = map[string]interface{}{"index": i, "embedding": vector}
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"data": data})
	}))
	defer server.Close()
	runtime := native.New(binding.NewPool())
	makeConfig := func(model string) *config.RouterConfig {
		cfg := &config.RouterConfig{}
		cfg.EmbeddingConfig.ModelType = "remote"
		cfg.EmbeddingConfig.Backend = config.EmbeddingBackendOpenAICompatible
		cfg.EmbeddingConfig.TargetDimension = 2
		cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "active", Candidates: []string{"candidate"}}}
		cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "service", Contract: "embedding.v1", Adapter: "openai_compatible"}}
		cfg.ModelDeployments = map[string]config.ModelDeployment{"service": {Provider: "http", ExternalModel: "embedder"}}
		cfg.ExternalModels = []config.ExternalModelConfig{{Name: "embedder", ModelName: model, ModelEndpoint: config.ClassifierVLLMEndpoint{Address: server.URL}}}
		return cfg
	}
	first, err := PrepareOwnedEmbeddings(context.Background(), makeConfig("first"), runtime)
	if err != nil {
		t.Fatal(err)
	}
	peer, err := PrepareOwnedEmbeddings(context.Background(), makeConfig("first"), runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer peer.Close()
	second, err := PrepareOwnedEmbeddings(context.Background(), makeConfig("second"), runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	firstProvider, _ := first.Default()
	peerProvider, _ := peer.Default()
	secondProvider, _ := second.Default()
	if _, err = PrepareOwnedEmbeddings(context.Background(), makeConfig("invalid"), runtime); err == nil {
		t.Fatal("candidate with empty vector accepted")
	}
	if err = first.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = firstProvider.Embed(context.Background(), "query"); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed binding error = %v", err)
	}
	vector, err := peerProvider.Embed(context.Background(), "query")
	if err != nil || len(vector) != 2 || vector[0] != 1 {
		t.Fatalf("peer = %v, %v", vector, err)
	}
	vector, err = secondProvider.Embed(context.Background(), "query")
	if err != nil || len(vector) != 2 || vector[1] != 1 {
		t.Fatalf("second = %v, %v", vector, err)
	}
}

func TestOwnedRecipeEmbeddingDoesNotProvisionSharedCache(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.EmbeddingModel = "mmbert"
	cfg.MmBertModelPath = "/does-not-exist/shared-cache"
	prepared, err := PrepareOwnedRecipeEmbeddings(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	if prepared.Ready() {
		t.Fatal("standalone recipe provisioned shared cache model")
	}
	if _, err := PrepareOwnedEmbeddings(context.Background(), cfg, nil); err == nil {
		t.Fatal("service runtime ignored required cache artifact")
	}
}
