package modelruntime

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// embeddingRuntime serves embedding cards for named deployments. A vector is
// [deployment-specific marker, dimension, layer, 1...] at the requested view.
type embeddingRuntime struct {
	mu     sync.Mutex
	cards  map[string]modelservice.ModelCard
	served map[string]int
}

// embeddingDeployment is the deployment cfg serves an embedding model
// keyword on.
func embeddingDeployment(t *testing.T, cfg *config.RouterConfig, model string) string {
	t.Helper()
	name, _, err := cfg.EmbeddingModelDeployment(model)
	if err != nil {
		t.Fatal(err)
	}
	return name
}

func newEmbeddingRuntime(deployments ...string) *embeddingRuntime {
	runtime := &embeddingRuntime{cards: make(map[string]modelservice.ModelCard), served: make(map[string]int)}
	for i, deployment := range deployments {
		runtime.cards[deployment] = modelservice.ModelCard{
			ID: deployment, ModelSHA256: strings.Repeat(string(rune('a'+i)), 64), Surfaces: []string{"embeddings"},
			MaxInputTokens: 512, Device: "cpu", Dtype: "float32",
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

func (r *embeddingRuntime) CurrentCard(deployment string) (modelservice.ModelCard, bool) {
	card, ok := r.cards[deployment]
	return card, ok
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

func (r *embeddingRuntime) inputs(deployment string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.served[deployment]
}

func TestImplicitEmbeddingSpecResolvesRegistryPackages(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	omni := t.TempDir()
	cfg.MultiModalModelPath = omni
	cfg.EmbeddingModels.UseCPU = true
	spec, err := implicitEmbeddingSpec(cfg, config.DefaultRecipeName, "mmbert")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Binding.Deployment != "Vela-1.0-Encoder-307M-Embedding" || spec.Deployment.Artifact != "vllm-sr/Vela-1.0-Encoder-307M-Embedding" ||
		len(spec.Deployment.Revision) != 40 || spec.Deployment.Device != "cpu" || spec.Deployment.Input.Overflow != "truncate" ||
		spec.Deployment.HasEngineArgs() {
		t.Fatalf("mmbert spec %+v", spec)
	}
	cfg.EmbeddingModels.UseCPU = false
	if spec, err = implicitEmbeddingSpec(cfg, config.DefaultRecipeName, "multimodal"); err != nil || spec.Deployment.Artifact != omni || spec.Deployment.Device != "auto" ||
		spec.Deployment.HasEngineArgs() {
		t.Fatalf("multimodal spec %+v, %v", spec, err)
	}
	cfg.Qwen3ModelPath = "models/mom-embedding-light"
	for _, model := range []string{"bert", "gemma", "qwen3"} {
		if _, err := implicitEmbeddingSpec(cfg, config.DefaultRecipeName, model); err == nil {
			t.Fatalf("%s must not resolve (unserved model, or a path that is neither registered nor local)", model)
		}
	}
}
