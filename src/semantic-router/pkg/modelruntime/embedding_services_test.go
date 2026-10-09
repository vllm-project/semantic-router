package modelruntime

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// embeddingRuntime serves embedding cards for named deployments. A vector is
// [deployment-specific marker, dimension, layer, 1...] at the requested view.
type embeddingRuntime struct {
	mu     sync.Mutex
	cards  map[string]modelservice.ModelCard
	served map[string]int
}

func newEmbeddingRuntime(deployments ...string) *embeddingRuntime {
	runtime := &embeddingRuntime{cards: make(map[string]modelservice.ModelCard), served: make(map[string]int)}
	for i, deployment := range deployments {
		runtime.cards[deployment] = modelservice.ModelCard{
			ID: deployment, ModelSHA256: strings.Repeat(string(rune('a'+i)), 64), Surfaces: []string{"embeddings"},
			MaxInputTokens: 512, Device: "cpu", Dtype: "float32", Profile: "exact",
			Embedding: &modelservice.EmbeddingCard{Dimensions: []int{8, 4, 3}, Layers: []int{2, 6}, Modalities: []string{"text"}, Normalized: true, Pooling: "mean"},
		}
	}
	return runtime
}

func (r *embeddingRuntime) Card(_ context.Context, deployment string) (modelservice.ModelCard, error) {
	card, ok := r.cards[deployment]
	if !ok {
		return modelservice.ModelCard{}, modelservice.ErrUnknownDeployment
	}
	return card, nil
}

func (r *embeddingRuntime) Classify(context.Context, string, modelservice.ClassifyRequest) (modelservice.ClassifyResponse, error) {
	return modelservice.ClassifyResponse{}, errors.New("not served")
}

func (r *embeddingRuntime) Embed(_ context.Context, deployment string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	r.mu.Lock()
	r.served[deployment] += len(request.Inputs)
	r.mu.Unlock()
	dimension := r.cards[deployment].Embedding.Dimensions[0]
	if request.Dimensions > 0 {
		dimension = request.Dimensions
	}
	response := modelservice.EmbedResponse{Embeddings: make([][]float32, len(request.Inputs)), Inputs: make([]*modelservice.InputUsage, len(request.Inputs)), Errors: make([]string, len(request.Inputs))}
	for i := range request.Inputs {
		vector := make([]float32, dimension)
		for j := range vector {
			vector[j] = 1
		}
		vector[0], vector[1], vector[2] = float32(len(deployment)), float32(dimension), float32(request.Layer)
		response.Embeddings[i] = vector
		response.Inputs[i] = &modelservice.InputUsage{Tokens: 3, ProcessedTokens: 3}
	}
	return response, nil
}

func (r *embeddingRuntime) Rerank(context.Context, string, modelservice.RerankRequest) (modelservice.RerankResponse, error) {
	return modelservice.RerankResponse{}, errors.New("not served")
}

func (r *embeddingRuntime) inputs(deployment string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.served[deployment]
}

func TestImplicitEmbeddingDeploymentsServeModuleDefaults(t *testing.T) {
	services := newEmbeddingRuntime("@embedding.mmbert", "@embedding.qwen3")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	cfg.Qwen3ModelPath = t.TempDir()
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "route", Candidates: []string{"hello"}}}
	cfg.ModelSelection.ML.ModelsPath, cfg.ModelSelection.ML.ModelType = "selectors", "qwen3"
	cfg.Decisions = []config.Decision{{Algorithm: &config.AlgorithmConfig{Type: "knn"}}}
	runtime := serving.New(services, binding.NewPool())
	prepared, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	for model, deployment := range map[string]string{"mmbert": "@embedding.mmbert", "qwen3": "@embedding.qwen3"} {
		provider, err := prepared.Get(model, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		vector, err := provider.Embed(context.Background(), model+" query")
		if err != nil || len(vector) != 8 || vector[0] != float32(len(deployment)) {
			t.Fatalf("%s served by the wrong deployment: %v, %v", model, vector, err)
		}
	}
	entries := runtime.PreparedBindings()
	if len(entries) != 2 || entries[0].Capability.Provider != config.ModelRuntimeProvider {
		t.Fatalf("prepared bindings %+v", entries)
	}
}

func TestImplicitEmbeddingSpecResolvesRegistryPackages(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	omni := t.TempDir()
	cfg.MultiModalModelPath = omni
	cfg.EmbeddingModels.UseCPU = true
	spec, err := cfg.ImplicitEmbeddingBinding(config.DefaultRecipeName, "mmbert")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Binding.Deployment != "@embedding.mmbert" || spec.Deployment.Artifact != "vllm-sr/Vela-1.0-Encoder-307M-Embedding" ||
		len(spec.Deployment.Revision) != 40 || spec.Deployment.Device != "cpu" || spec.Deployment.Input.Overflow != "truncate" ||
		spec.Deployment.Profile != "exact" {
		t.Fatalf("mmbert spec %+v", spec)
	}
	cfg.EmbeddingModels.UseCPU = false
	if spec, err = cfg.ImplicitEmbeddingBinding(config.DefaultRecipeName, "multimodal"); err != nil || spec.Deployment.Artifact != omni || spec.Deployment.Device != "auto" ||
		spec.Deployment.Profile != "exact" {
		t.Fatalf("multimodal spec %+v, %v", spec, err)
	}
	cfg.Qwen3ModelPath = "models/mom-embedding-light"
	for _, model := range []string{"bert", "gemma", "qwen3"} {
		if _, err := cfg.ImplicitEmbeddingBinding(config.DefaultRecipeName, model); err == nil {
			t.Fatalf("%s must not resolve (unserved model, or a path that is neither registered nor local)", model)
		}
	}
}

func TestConfiguredViewDoesNotChangeGlobalConsumers(t *testing.T) {
	services := newEmbeddingRuntime("primary")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType, cfg.EmbeddingConfig.TargetLayer, cfg.EmbeddingConfig.TargetDimension = "mmbert", 6, 3
	cfg.Tools.Enabled = true
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "primary", Candidates: []string{"hello"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Endpoint: "http://runtime:8100", Input: config.ModelInputBudget{Overflow: "truncate"}}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "primary", Contract: "embedding.v1"}}
	runtime := serving.New(services, nil)
	prepared, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	global, err := PrepareOwnedGlobalServiceEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer global.Close()
	primary, err := prepared.Get("mmbert", 3, 6)
	if err != nil {
		t.Fatal(err)
	}
	vector, err := primary.Embed(context.Background(), "hello world")
	if err != nil || len(vector) != 3 || vector[2] != 6 {
		t.Fatalf("configured view: %v, %v", vector, err)
	}
	tools, err := global.Default()
	if err != nil {
		t.Fatal(err)
	}
	if vector, err = tools.Embed(context.Background(), "hello world"); err != nil || len(vector) != 8 {
		t.Fatalf("the global tools consumer keeps the full view: %v, %v", vector, err)
	}
	resources := map[string]string{}
	for _, entry := range runtime.PreparedBindings() {
		resources[entry.Identity.Recipe] = entry.ResourceID
	}
	if resources[string(config.GlobalModelScope)] == "" || resources[string(config.GlobalModelScope)] != resources[string(config.DefaultRecipeName)] {
		t.Fatalf("one deployment must share one admission resource: %v", resources)
	}
}

func TestPreparedEmbeddingsShareTheContentCache(t *testing.T) {
	services := newEmbeddingRuntime("@embedding.mmbert")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = t.TempDir()
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "route", Candidates: []string{"hello"}}}
	runtime := serving.New(services, nil)
	first, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	a, _ := first.Default()
	b, _ := second.Default()
	before := services.inputs("@embedding.mmbert")
	text := "one request, two consumers, one runtime input"
	if _, err := a.Embed(context.Background(), text); err != nil {
		t.Fatal(err)
	}
	if _, err := embedding.Embed(context.Background(), b, text, embedding.Options{}); err != nil {
		t.Fatal(err)
	}
	if got := services.inputs("@embedding.mmbert") - before; got != 1 {
		t.Fatalf("%d runtime inputs for one text, want 1", got)
	}
}
