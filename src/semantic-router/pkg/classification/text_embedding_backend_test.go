package classification

import (
	"context"
	"fmt"
	"strings"
	"sync"
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

func TestBuildClassifierInitializesConfiguredOpenVINOForReaskOnly(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "")

	originalInit := initializeOpenVINOTextEmbedding
	originalEmbed := openVINOTextEmbedding
	var initCalls atomic.Int32
	var embedCalls atomic.Int32
	var initializedModelType string
	initializeOpenVINOTextEmbedding = func(modelType, _, fallbackPath string, _ bool) error {
		initCalls.Add(1)
		initializedModelType = modelType
		if fallbackPath == "" {
			return fmt.Errorf("missing fallback path")
		}
		return nil
	}
	openVINOTextEmbedding = func(modelType, _ string, _ int) ([]float32, error) {
		embedCalls.Add(1)
		if modelType != config.EmbeddingModelTypeQwen3 {
			return nil, fmt.Errorf("unexpected model type %q", modelType)
		}
		return []float32{1, 0}, nil
	}
	t.Cleanup(func() {
		initializeOpenVINOTextEmbedding = originalInit
		openVINOTextEmbedding = originalEmbed
	})

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
			Qwen3ModelPath: "models/explicit-qwen3",
			EmbeddingConfig: config.HNSWConfig{
				Backend:   config.EmbeddingBackendOpenVINO,
				ModelType: config.EmbeddingModelTypeQwen3,
			},
		}},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			ReaskRules: []config.ReaskRule{{Name: "repeat", Threshold: 0.8, LookbackTurns: 1}},
		}},
	}
	classifier, err := BuildClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildClassifier: %v", err)
	}
	if classifier.keywordEmbeddingClassifier != nil {
		t.Fatal("reask-only config unexpectedly built embedding-rules classifier")
	}
	if classifier.reaskClassifier == nil {
		t.Fatal("reask classifier was not built")
	}
	if classifier.reaskClassifier.backend != config.EmbeddingBackendOpenVINO || classifier.reaskClassifier.modelType != config.EmbeddingModelTypeQwen3 {
		t.Fatalf("reask runtime = backend %q model %q", classifier.reaskClassifier.backend, classifier.reaskClassifier.modelType)
	}
	if initCalls.Load() != 1 || initializedModelType != config.EmbeddingModelTypeQwen3 {
		t.Fatalf("OpenVINO init = calls %d model %q", initCalls.Load(), initializedModelType)
	}
	if _, err := classifier.reaskClassifier.embedText("repeat"); err != nil {
		t.Fatalf("reask embed: %v", err)
	}
	if embedCalls.Load() != 1 {
		t.Fatalf("OpenVINO embed calls = %d, want 1", embedCalls.Load())
	}
}

type candleOverrideCallCounters struct {
	init  atomic.Int32
	embed atomic.Int32
}

func installCountedCandleOverrideBackend(t *testing.T, counters *candleOverrideCallCounters) {
	t.Helper()
	originalInit := initializeCandleTextEmbedding
	originalEmbed := getEmbeddingWithModelType
	initializeCandleTextEmbedding = func(qwen3Path, gemmaPath, mmBertPath string, _ bool) error {
		counters.init.Add(1)
		if qwen3Path != "" || gemmaPath != "" || mmBertPath == "" {
			return fmt.Errorf("unexpected local paths qwen=%q gemma=%q mmbert=%q", qwen3Path, gemmaPath, mmBertPath)
		}
		return nil
	}
	getEmbeddingWithModelType = func(_ string, modelType string, _ int) (*candle_binding.EmbeddingOutput, error) {
		counters.embed.Add(1)
		if modelType != "mmbert" {
			return nil, fmt.Errorf("unexpected local model type %q", modelType)
		}
		return &candle_binding.EmbeddingOutput{Embedding: []float32{1, 0}}, nil
	}
	t.Cleanup(func() {
		initializeCandleTextEmbedding = originalInit
		getEmbeddingWithModelType = originalEmbed
	})
}

func TestBuildClassifierRemoteConfigWithCandleOverridePreparesExplicitLocalModel(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendCandle)
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "mmbert")

	counters := &candleOverrideCallCounters{}
	installCountedCandleOverrideBackend(t, counters)

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
			MmBertModelPath: "models/explicit-mmbert",
			EmbeddingConfig: config.HNSWConfig{
				Backend:   config.EmbeddingBackendOpenAICompatible,
				ModelType: config.EmbeddingModelTypeRemote,
			},
			Endpoint: config.EmbeddingEndpointConfig{BaseURL: "https://must-not-be-used.invalid/v1", Model: "remote"},
		}},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			ReaskRules: []config.ReaskRule{{Name: "repeat", Threshold: 0.8, LookbackTurns: 1}},
		}},
	}
	classifier, err := BuildClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildClassifier: %v", err)
	}
	if classifier.reaskClassifier == nil || classifier.reaskClassifier.provider != nil {
		t.Fatalf("local override retained remote provider: %+v", classifier.reaskClassifier)
	}
	if classifier.reaskClassifier.modelType != "mmbert" || classifier.reaskClassifier.backend != config.EmbeddingBackendOpenAICompatible {
		t.Fatalf("runtime classifier = %+v", classifier.reaskClassifier)
	}
	if counters.init.Load() != 1 {
		t.Fatalf("Candle init calls = %d, want 1", counters.init.Load())
	}
	if _, err := classifier.reaskClassifier.embedText("repeat"); err != nil {
		t.Fatalf("local reask embed: %v", err)
	}
	if counters.embed.Load() != 1 {
		t.Fatalf("Candle embed calls = %d, want 1", counters.embed.Load())
	}
}

func TestNonContrastiveJailbreakRulesDoNotInitializeTextBackend(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	originalInit := initializeOpenVINOTextEmbedding
	var initCalls atomic.Int32
	initializeOpenVINOTextEmbedding = func(string, string, string, bool) error {
		initCalls.Add(1)
		return nil
	}
	t.Cleanup(func() { initializeOpenVINOTextEmbedding = originalInit })

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
			Qwen3ModelPath:  "models/qwen3",
			EmbeddingConfig: config.HNSWConfig{Backend: config.EmbeddingBackendOpenVINO, ModelType: config.EmbeddingModelTypeQwen3},
		}},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			JailbreakRules: []config.JailbreakRule{{Name: "bert-only", Method: "classifier"}},
		}},
	}
	option, err := newClassifierOptionBuilder(cfg, nil).buildContrastiveJailbreakClassifiersOption()
	if err != nil || option != nil {
		t.Fatalf("non-contrastive builder = option %v err %v", option, err)
	}
	if initCalls.Load() != 0 {
		t.Fatalf("non-contrastive rules initialized text backend %d times", initCalls.Load())
	}
}

func TestKnowledgeBaseBuilderFailsClosedOnInvalidDefinition(t *testing.T) {
	cfg := &config.RouterConfig{
		KnowledgeBases: []config.KnowledgeBaseConfig{{
			Name:   "required-kb",
			Source: config.KnowledgeBaseSource{Path: "missing-required-kb.json"},
		}},
	}
	option, err := newClassifierOptionBuilder(cfg, nil).buildKBClassifiersOption()
	if err == nil || option != nil {
		t.Fatalf("invalid KB builder = option %v err %v, want fail-closed error", option, err)
	}
}

func TestBuildClassifierRejectsConflictingOpenVINOFamilyModels(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "")
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				Qwen3ModelPath:  "models/qwen3",
				MmBertModelPath: "models/mmbert",
				EmbeddingConfig: config.HNSWConfig{Backend: config.EmbeddingBackendOpenVINO, ModelType: "qwen3"},
			},
			Classifier: config.Classifier{PreferenceModel: config.PreferenceModelConfig{EmbeddingModel: "mmbert"}},
		},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			ReaskRules:      []config.ReaskRule{{Name: "repeat", LookbackTurns: 1}},
			PreferenceRules: []config.PreferenceRule{{Name: "secure", Description: "secure preference"}},
		}},
	}
	classifier, err := BuildClassifier(cfg, nil, nil, nil)
	if classifier != nil || err == nil || !strings.Contains(err.Error(), "OpenVINO text embedding model conflict") {
		t.Fatalf("conflicting OpenVINO plans = classifier %v err %v", classifier, err)
	}
}

func TestTextEmbeddingRuntimeInitializesSameOpenVINOPlanOnce(t *testing.T) {
	originalInit := initializeOpenVINOTextEmbedding
	var initCalls atomic.Int32
	initializeOpenVINOTextEmbedding = func(string, string, string, bool) error {
		initCalls.Add(1)
		return nil
	}
	t.Cleanup(func() { initializeOpenVINOTextEmbedding = originalInit })

	cfg := &config.RouterConfig{InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
		Qwen3ModelPath: "models/qwen3",
	}}}
	plan := embedding.RuntimePlan{
		Backend:   config.EmbeddingBackendOpenVINO,
		ModelType: config.EmbeddingModelTypeQwen3,
		ModelPath: cfg.Qwen3ModelPath,
	}
	runtime := &textEmbeddingRuntime{}
	errs := make(chan error, 2)
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- runtime.ensureInitialized(cfg, plan)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("ensureInitialized: %v", err)
		}
	}
	if initCalls.Load() != 1 {
		t.Fatalf("OpenVINO init calls = %d, want 1", initCalls.Load())
	}
}

func TestExternalPreferenceDoesNotDependOnEmbeddingRuntime(t *testing.T) {
	useContrastive := false
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{EmbeddingConfig: config.HNSWConfig{
				Backend:   "invalid-unused-backend",
				ModelType: "invalid-unused-model",
			}},
			Classifier: config.Classifier{PreferenceModel: config.PreferenceModelConfig{UseContrastive: &useContrastive}},
		},
		ExternalModels: []config.ExternalModelConfig{{
			ModelRole:     config.ModelRolePreference,
			ModelName:     "external-preference",
			ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "http://preference.example"},
		}},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			PreferenceRules: []config.PreferenceRule{{Name: "secure", Description: "secure preference"}},
		}},
	}
	classifier := &Classifier{Config: cfg, textBackendRuntime: &textEmbeddingRuntime{}}
	if err := classifier.initializePreferenceClassifier(); err != nil {
		t.Fatalf("external preference inherited unused embedding failure: %v", err)
	}
	if classifier.preferenceClassifier == nil || classifier.preferenceClassifier.useContrastive || !classifier.preferenceClassifier.IsInitialized() {
		t.Fatalf("external preference was not initialized: %+v", classifier.preferenceClassifier)
	}
}
