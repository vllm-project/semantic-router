package classification

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type countedEmbeddingProvider struct {
	calls atomic.Int32
}

func (p *countedEmbeddingProvider) Embed(context.Context, string) ([]float32, error) {
	p.calls.Add(1)
	return []float32{1, 0}, nil
}

func (p *countedEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		embedding, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = embedding
	}
	return result, nil
}

func (*countedEmbeddingProvider) Dimension() int { return 2 }
func (*countedEmbeddingProvider) Backend() string {
	return config.EmbeddingBackendOpenAICompatible
}

type embeddingFamilyCase struct {
	name      string
	callCount int32
	run       func(embedding.Provider, string) error
}

type embeddingBackendCase struct {
	name         string
	override     string
	backend      string
	withProvider bool
	wantProvider bool
	wantCandle   bool
	wantOpenVINO bool
}

type nativeEmbeddingCallCounters struct {
	candle   atomic.Int32
	openVINO atomic.Int32
}

func installCountedNativeEmbeddingBackends(t *testing.T, counters *nativeEmbeddingCallCounters) {
	t.Helper()
	originalCandle := getEmbeddingWithModelType
	originalOpenVINO := openVINOTextEmbedding
	getEmbeddingWithModelType = func(string, string, int) (*candle_binding.EmbeddingOutput, error) {
		counters.candle.Add(1)
		return &candle_binding.EmbeddingOutput{Embedding: []float32{1, 0}}, nil
	}
	openVINOTextEmbedding = func(string, string, int) ([]float32, error) {
		counters.openVINO.Add(1)
		return []float32{1, 0}, nil
	}
	t.Cleanup(func() {
		getEmbeddingWithModelType = originalCandle
		openVINOTextEmbedding = originalOpenVINO
	})
}

func providerBackedEmbeddingFamilies() []embeddingFamilyCase {
	return []embeddingFamilyCase{
		{
			name:      "embedding rules",
			callCount: 1,
			run: func(provider embedding.Provider, backend string) error {
				classifier := &EmbeddingClassifier{
					// HNSW defaults materialize Candle even when model_type=remote
					// selected the provider at the parent embedding_models layer.
					backend:  backend,
					provider: provider,
				}
				_, err := classifier.computeEmbedding("query", config.EmbeddingModelTypeQwen3)
				return err
			},
		},
		{
			name:      "reask",
			callCount: 1,
			run: func(provider embedding.Provider, backend string) error {
				_, err := (&ReaskClassifier{modelType: config.EmbeddingModelTypeQwen3, backend: backend, provider: provider}).embedText("query")
				return err
			},
		},
		{
			name:      "complexity request and preload",
			callCount: 2,
			run: func(provider embedding.Provider, backend string) error {
				classifier := &ComplexityClassifier{modelType: config.EmbeddingModelTypeQwen3, backend: backend, provider: provider}
				if _, err := classifier.loadQueryEmbeddingsCached("query", "", nil); err != nil {
					return err
				}
				_, err := classifier.computeCandidateEmbedding(complexityCandidateTask{candidate: "candidate"})
				return err
			},
		},
		{
			name:      "contrastive preference",
			callCount: 1,
			run: func(provider embedding.Provider, backend string) error {
				_, err := (&ContrastivePreferenceClassifier{modelType: config.EmbeddingModelTypeQwen3, backend: backend, provider: provider}).embedText("query")
				return err
			},
		},
		{
			name:      "contrastive jailbreak",
			callCount: 1,
			run: func(provider embedding.Provider, backend string) error {
				_, err := (&ContrastiveJailbreakClassifier{modelType: config.EmbeddingModelTypeQwen3, backend: backend, provider: provider}).embedText("query")
				return err
			},
		},
		{
			name:      "knowledge base request and preload",
			callCount: 2,
			run: func(provider embedding.Provider, backend string) error {
				classifier := &KnowledgeBaseClassifier{modelType: config.EmbeddingModelTypeQwen3, backend: backend, provider: provider}
				if _, err := classifier.embedText("query"); err != nil {
					return err
				}
				result := classifier.embedOneExemplar(
					classifier.currentBackend(),
					config.EmbeddingModelTypeQwen3,
					0,
					exemplarRef{text: "candidate"},
				)
				return result.err
			},
		},
	}
}

func effectiveEmbeddingBackendCases() []embeddingBackendCase {
	return []embeddingBackendCase{
		{name: "configured remote with HNSW candle default", backend: config.EmbeddingBackendCandle, withProvider: true, wantProvider: true},
		{name: "remote override", override: config.EmbeddingBackendOpenAICompatible, backend: config.EmbeddingBackendCandle, withProvider: true, wantProvider: true},
		{name: "candle override", override: config.EmbeddingBackendCandle, backend: config.EmbeddingBackendOpenAICompatible, withProvider: true, wantCandle: true},
		{name: "openvino override", override: config.EmbeddingBackendOpenVINO, backend: config.EmbeddingBackendOpenAICompatible, withProvider: true, wantOpenVINO: true},
		{name: "configured candle without provider", backend: config.EmbeddingBackendCandle, wantCandle: true},
		{name: "configured openvino without provider", backend: config.EmbeddingBackendOpenVINO, wantOpenVINO: true},
	}
}

func runEmbeddingFamiliesForBackend(
	t *testing.T,
	backend embeddingBackendCase,
	families []embeddingFamilyCase,
	counters *nativeEmbeddingCallCounters,
) {
	t.Helper()
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", backend.override)
	for _, family := range families {
		t.Run(family.name, func(t *testing.T) {
			assertEmbeddingFamilyUsesBackend(t, backend, family, counters)
		})
	}
}

func assertEmbeddingFamilyUsesBackend(
	t *testing.T,
	backend embeddingBackendCase,
	family embeddingFamilyCase,
	counters *nativeEmbeddingCallCounters,
) {
	t.Helper()
	countedProvider := &countedEmbeddingProvider{}
	var provider embedding.Provider
	if backend.withProvider {
		provider = countedProvider
	}
	counters.candle.Store(0)
	counters.openVINO.Store(0)
	if err := family.run(provider, backend.backend); err != nil {
		t.Fatalf("execute family: %v", err)
	}

	assertEmbeddingCallCount(t, "provider", countedProvider.calls.Load(), expectedEmbeddingCalls(backend.wantProvider, family.callCount))
	assertEmbeddingCallCount(t, "candle", counters.candle.Load(), expectedEmbeddingCalls(backend.wantCandle, family.callCount))
	assertEmbeddingCallCount(t, "openvino", counters.openVINO.Load(), expectedEmbeddingCalls(backend.wantOpenVINO, family.callCount))
}

func expectedEmbeddingCalls(enabled bool, callCount int32) int32 {
	if enabled {
		return callCount
	}
	return 0
}

func assertEmbeddingCallCount(t *testing.T, backend string, got, want int32) {
	t.Helper()
	if got != want {
		t.Fatalf("%s calls = %d, want %d", backend, got, want)
	}
}

func TestProviderBackedClassifiersUseExactlyTheEffectiveBackend(t *testing.T) {
	counters := &nativeEmbeddingCallCounters{}
	installCountedNativeEmbeddingBackends(t, counters)
	families := providerBackedEmbeddingFamilies()
	for _, backend := range effectiveEmbeddingBackendCases() {
		t.Run(backend.name, func(t *testing.T) {
			runEmbeddingFamiliesForBackend(t, backend, families, counters)
		})
	}
}

func TestRemoteBackendWithoutProviderNeverFallsBackToNative(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenAICompatible)

	originalCandle := getEmbeddingWithModelType
	originalOpenVINO := openVINOTextEmbedding
	var nativeCalls atomic.Int32
	getEmbeddingWithModelType = func(string, string, int) (*candle_binding.EmbeddingOutput, error) {
		nativeCalls.Add(1)
		return nil, fmt.Errorf("must not be called")
	}
	openVINOTextEmbedding = func(string, string, int) ([]float32, error) {
		nativeCalls.Add(1)
		return nil, fmt.Errorf("must not be called")
	}
	t.Cleanup(func() {
		getEmbeddingWithModelType = originalCandle
		openVINOTextEmbedding = originalOpenVINO
	})

	if _, _, err := executeTextEmbedding(context.Background(), "", nil, "query", config.EmbeddingModelTypeQwen3, 0); err == nil {
		t.Fatal("remote backend without provider succeeded")
	}
	if got := nativeCalls.Load(); got != 0 {
		t.Fatalf("remote backend fell back to native %d times", got)
	}
}
