package classification

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExplicitEmbeddingBindingOverridesLegacyOpenVINOSelector(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		data := make([]map[string]any, len(request.Input))
		for i, text := range request.Input {
			vector := []float32{0, 1}
			if strings.Contains(text, "billing") {
				vector = []float32{1, 0}
			}
			data[i] = map[string]any{"index": i, "embedding": vector}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer server.Close()
	cfg := &config.RouterConfig{}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"test"}, Recipe: config.DefaultRecipeName}}
	cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendOpenVINO, ModelType: "mmbert", TargetDimension: 2, PreloadEmbeddings: true}
	cfg.MmBertModelPath = "/unavailable-legacy-openvino/model.xml"
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "billing", Candidates: []string{"billing invoice"}, SimilarityThreshold: 0.9, AggregationMethodConfiged: config.AggregationMethodMax}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "selected", Contract: "embedding.v1", Adapter: "openai_compatible"}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"selected": {Provider: "http", ExternalModel: "embedder"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "embedder", ModelName: "selected-embedding", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: server.URL}}}
	classifier, err := NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer classifier.Close()
	if classifier.keywordEmbeddingClassifier == nil || classifier.keywordEmbeddingClassifier.provider == nil {
		t.Fatal("explicit prepared binding was replaced by the legacy nil-provider fallback")
	}
	before := calls.Load()
	result, err := classifier.keywordEmbeddingClassifier.ClassifyDetailed("billing support")
	if err != nil || len(result.Matches) != 1 || result.Matches[0].RuleName != "billing" {
		t.Fatalf("explicit binding inference: %+v %v", result, err)
	}
	if calls.Load() <= before {
		t.Fatal("classification did not execute the explicitly bound HTTP provider")
	}
}
