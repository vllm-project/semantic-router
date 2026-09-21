//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestEmbeddingAPIGlobalAndExplicitRecipeHTTP(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			return
		}
		vector := make([]float32, 64)
		if request.Model == "global" {
			vector[0] = 1
		} else {
			vector[1] = 1
		}
		data := make([]map[string]any, len(request.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": vector}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer backend.Close()
	cfg := &config.RouterConfig{}
	cfg.API.Embeddings.Enabled = true
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", TargetDimension: 64}
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "topic", Candidates: []string{"hello"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"global": {Provider: "http", ExternalModel: "global"},
		"local":  {Provider: "http", ExternalModel: "local"},
	}
	for _, model := range []string{"global", "local"} {
		cfg.ExternalModels = append(cfg.ExternalModels, config.ExternalModelConfig{Name: model, ModelName: model, ModelEndpoint: config.ClassifierVLLMEndpoint{Address: backend.URL}})
	}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "local", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	service, err := services.NewClassificationServiceFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer service.Close()
	api := &ClassificationAPIServer{classificationSvc: service, config: cfg}
	for _, tc := range []struct {
		name, body, scope string
		handler           http.HandlerFunc
	}{
		{"global", `{"texts":["hello"],"dimension":64}`, "@global", api.handleEmbeddings},
		{"recipe", `{"texts":["hello"],"dimension":64,"recipe":"default"}`, "default", api.handleEmbeddings},
		{"similarity", `{"text1":"hello","text2":"hello","dimension":64}`, "@global", api.handleSimilarity},
		{"batch", `{"query":"hello","candidates":["hello"],"dimension":64}`, "@global", api.handleBatchSimilarity},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			tc.handler(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(tc.body)))
			if recorder.Code != http.StatusOK {
				t.Fatalf("HTTP %d: %s", recorder.Code, recorder.Body.String())
			}
			var response struct {
				Recipe     string `json:"recipe"`
				Embeddings []struct {
					Embedding []float32 `json:"embedding"`
				} `json:"embeddings"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if response.Recipe != tc.scope {
				t.Fatalf("response scope=%q, want %q", response.Recipe, tc.scope)
			}
			if len(response.Embeddings) > 0 {
				index := 0
				if tc.scope == "default" {
					index = 1
				}
				if response.Embeddings[0].Embedding[index] != 1 {
					t.Fatalf("request crossed model scope: %+v", response.Embeddings)
				}
			}
		})
	}
}

func TestEmbeddingAPIUnavailableReleasesSimilarityLease(t *testing.T) {
	for _, batch := range []bool{false, true} {
		cfg := &config.RouterConfig{}
		cfg.API.Embeddings.Enabled = true
		service := services.NewClassificationService(nil, cfg)
		api := &ClassificationAPIServer{classificationSvc: service, config: cfg}
		recorder := httptest.NewRecorder()
		if batch {
			api.handleBatchSimilarity(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"query":"hello","candidates":["hi"]}`)))
		} else {
			api.handleSimilarity(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"text1":"hello","text2":"hi"}`)))
		}
		if recorder.Code != http.StatusServiceUnavailable {
			t.Fatalf("unavailable runtime HTTP %d: %s", recorder.Code, recorder.Body.String())
		}
		closed := make(chan error, 1)
		go func() { closed <- service.Close() }()
		select {
		case err := <-closed:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(time.Second):
			t.Fatal("error response leaked a runtime lease")
		}
	}
}
