package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func newReadinessEmbeddingConfig(t *testing.T) (*config.RouterConfig, func(string) int) {
	t.Helper()
	var mu sync.Mutex
	inputs := map[string]int{}
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		data := make([]map[string]any, len(request.Input))
		mu.Lock()
		for i, input := range request.Input {
			inputs[input]++
			data[i] = map[string]any{"index": i, "embedding": []float32{1, 0}}
		}
		mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	t.Cleanup(endpoint.Close)
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", TargetDimension: 2}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"embedding": {Provider: "http", ExternalModel: "test"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "test", ModelName: "test", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: endpoint.URL}}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "embedding", Contract: "embedding.v1", Adapter: "openai_compatible"}}
	return cfg, func(input string) int { mu.Lock(); defer mu.Unlock(); return inputs[input] }
}

// Prepare actual owned providers and consumers, then exercise the same component
// publication and startup/reload warmup used by the server.
func prepareReadinessRouter(t *testing.T, cfg *config.RouterConfig) *OpenAIRouter {
	t.Helper()
	runtime := native.New(nil)
	components := &routerComponents{cfg: cfg, resources: newResourceScope()}
	var err error
	components.embeddings, err = modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	components.resources.add(components.embeddings.Close)
	components.serviceEmbeddings, err = modelruntime.PrepareOwnedGlobalServiceEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	components.resources.add(components.serviceEmbeddings.Close)
	components.cacheEmbeddings, err = modelruntime.PrepareOwnedResponseCacheEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	components.resources.add(components.cacheEmbeddings.Close)
	components.toolsDatabase, components.toolEmbedder, err = buildToolsRuntime(cfg, components.serviceEmbeddings)
	if err != nil {
		t.Fatal(err)
	}
	components.recipeClassifiers, err = classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Runtime: runtime, Embeddings: components.embeddings})
	if err != nil {
		t.Fatal(err)
	}
	components.resources.add(components.recipeClassifiers.Close)
	components.classifier = components.recipeClassifiers.Default()
	router := components.buildRouter()
	t.Cleanup(func() { _ = router.Close() })
	return router
}

func TestStartupWarmupUsesGlobalToolsOwner(t *testing.T) {
	cfg, count := newReadinessEmbeddingConfig(t)
	cfg.Tools.Enabled = true
	cfg.Tools.ToolsDBPath = filepath.Join(t.TempDir(), "tools.json")
	if err := os.WriteFile(cfg.Tools.ToolsDBPath, []byte(`[{"tool":{"type":"function","function":{"name":"lookup"}},"description":"global tool description"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	router := prepareReadinessRouter(t, cfg)
	if router.Embeddings.Ready() {
		t.Fatal("fixture unexpectedly has recipe embeddings")
	}
	state := router.embeddingRuntimeState()
	if !state.AnyReady || !state.ToolsReady || state.KnowledgeBasesReady {
		t.Fatalf("wrong tools-only readiness: %+v", state)
	}
	server := &Server{service: NewRouterService(router)}
	if err := server.WarmupRouter(context.Background(), modelruntime.WarmupRouterOptions{}); err != nil {
		t.Fatal(err)
	}
	if router.ToolsDatabase.GetToolCount() != 1 || count("global tool description") != 1 {
		t.Fatal("prepared global tools were skipped during startup")
	}
}

func TestStartupReadinessIncludesCacheWithoutEnablingOtherConsumers(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires real ONNX Runtime")
	}
	artifact, err := filepath.Abs("../../../../onnx-binding/instance/testdata/omni")
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "multimodal"
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.EmbeddingModel = "multimodal"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"omni": {Artifact: artifact, Provider: "ort", Device: "cpu", Input: config.ModelInputBudget{MaxTokens: 512, Overflow: "reject"}}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "omni", Adapter: "vela_omni", Contract: "embedding.v1"}}
	router := prepareReadinessRouter(t, cfg)
	if router.Embeddings.Ready() || router.serviceEmbeddings.Ready() || !router.cacheEmbeddings.Ready() {
		t.Fatal("fixture must prepare only the cache owner")
	}
	state := router.embeddingRuntimeState()
	if !state.AnyReady || state.ToolsReady || state.KnowledgeBasesReady {
		t.Fatalf("cache readiness leaked to tools/KB: %+v", state)
	}
}

func TestReloadWarmupUsesNewNamedKnowledgeBaseOwner(t *testing.T) {
	previousCfg, previousCount := newReadinessEmbeddingConfig(t)
	previousCfg.Tools.Enabled = true
	previousCfg.Tools.ToolsDBPath = filepath.Join(t.TempDir(), "tools.json")
	if err := os.WriteFile(previousCfg.Tools.ToolsDBPath, []byte(`[{"tool":{"type":"function","function":{"name":"lookup"}},"description":"retired tool"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	previous := prepareReadinessRouter(t, previousCfg)
	priorState := previous.embeddingRuntimeState()
	if !priorState.ToolsReady {
		t.Fatal("previous generation tools were not prepared")
	}
	if err := previous.Close(); err != nil {
		t.Fatal(err)
	}

	cfg, count := newReadinessEmbeddingConfig(t)
	kbDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(kbDir, "labels.json"), []byte(`{"version":"1.0.0","labels":{"topic":{"description":"topic","exemplars":["new generation KB exemplar"]}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg.KnowledgeBases = []config.KnowledgeBaseConfig{{Name: "named-kb", Source: config.KnowledgeBaseSource{Path: kbDir}}}
	cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "named", Profile: config.RoutingProfile{Signals: config.Signals{KBRules: []config.KBSignalRule{{Name: "topic", KB: "named-kb", Target: config.KBSignalTarget{Kind: config.KBTargetKindLabel, Value: "topic"}}}}}}}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"named-entry"}, Recipe: "named"}}
	current := prepareReadinessRouter(t, cfg)
	if current.Embeddings.Ready() || current.serviceEmbeddings.Ready() {
		t.Fatal("fixture unexpectedly prepares a default/global embedding")
	}
	state := current.embeddingRuntimeState()
	if !state.AnyReady || state.ToolsReady || !state.KnowledgeBasesReady {
		t.Fatalf("wrong named KB readiness: %+v", state)
	}
	if count("new generation KB exemplar") != 0 {
		t.Fatal("KB should still need startup warmup")
	}
	if err := warmupReloadRouter(current); err != nil {
		t.Fatal(err)
	}
	if count("new generation KB exemplar") != 1 || previousCount("retired tool") != 0 {
		t.Fatal("reload borrowed previous readiness or skipped new KB owner")
	}
	// An empty following generation cannot inherit KB/tools availability either.
	emptyCfg, _ := newReadinessEmbeddingConfig(t)
	empty := prepareReadinessRouter(t, emptyCfg)
	if err := warmupReloadRouter(empty); err != nil {
		t.Fatal(err)
	}
	emptyState := empty.embeddingRuntimeState()
	if emptyState.AnyReady || emptyState.ToolsReady || emptyState.KnowledgeBasesReady {
		t.Fatalf("empty generation retained prior readiness: %+v", emptyState)
	}
}
