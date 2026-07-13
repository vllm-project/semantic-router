package classification

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

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
	if classifier.reaskClassifier.modelType != "mmbert" || classifier.reaskClassifier.backend != config.EmbeddingBackendCandle {
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

func TestBuildClassifierLocalConfigWithRemoteOverrideUsesEndpointProvider(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenAICompatible)
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "")

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{{"index": 0, "embedding": []float64{1, 0}}},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	defer server.Close()

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
			Qwen3ModelPath: "models/local-qwen3",
			EmbeddingConfig: config.HNSWConfig{
				Backend:   config.EmbeddingBackendCandle,
				ModelType: config.EmbeddingModelTypeQwen3,
			},
			Endpoint: config.EmbeddingEndpointConfig{BaseURL: server.URL + "/v1", Model: "remote-embedding"},
		}},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			ReaskRules: []config.ReaskRule{{Name: "repeat", Threshold: 0.8, LookbackTurns: 1}},
		}},
	}
	classifier, err := BuildClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildClassifier: %v", err)
	}
	if classifier.reaskClassifier == nil || classifier.reaskClassifier.provider == nil {
		t.Fatalf("remote override provider = %+v", classifier.reaskClassifier)
	}
	if classifier.reaskClassifier.modelType != config.EmbeddingModelTypeRemote {
		t.Fatalf("remote override model type = %q", classifier.reaskClassifier.modelType)
	}
	if _, err := classifier.reaskClassifier.embedText("repeat"); err != nil {
		t.Fatalf("remote reask embed: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("remote embedding HTTP calls = %d, want 1", got)
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

func concurrentCandleUnionConfig() *config.RouterConfig {
	useContrastive := true
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				Qwen3ModelPath:  "models/qwen3",
				GemmaModelPath:  "models/gemma",
				MmBertModelPath: "models/mmbert",
				EmbeddingConfig: config.HNSWConfig{
					Backend:   config.EmbeddingBackendCandle,
					ModelType: config.EmbeddingModelTypeQwen3,
				},
			},
			Classifier: config.Classifier{PreferenceModel: config.PreferenceModelConfig{
				UseContrastive: &useContrastive,
				EmbeddingModel: "mmbert",
			}},
		},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			ReaskRules:      []config.ReaskRule{{Name: "repeat", LookbackTurns: 1}},
			PreferenceRules: []config.PreferenceRule{{Name: "secure"}},
		}},
	}
}

func mustResolveTextEmbeddingRuntimePlan(t *testing.T, cfg *config.RouterConfig, label, modelType string) embedding.RuntimePlan {
	t.Helper()
	plan, err := resolveTextEmbeddingRuntimePlan(cfg, modelType)
	if err != nil {
		t.Fatalf("resolve %s plan: %v", label, err)
	}
	return plan
}

func installExpectedCandleUnionInitializer(t *testing.T, cfg *config.RouterConfig, initCalls *atomic.Int32) {
	t.Helper()
	originalInit := initializeCandleTextEmbedding
	initializeCandleTextEmbedding = func(qwen3Path, gemmaPath, mmBertPath string, _ bool) error {
		initCalls.Add(1)
		if qwen3Path != config.ResolveModelPath(cfg.Qwen3ModelPath) || gemmaPath != "" || mmBertPath != config.ResolveModelPath(cfg.MmBertModelPath) {
			return fmt.Errorf("unexpected Candle union qwen=%q gemma=%q mmbert=%q", qwen3Path, gemmaPath, mmBertPath)
		}
		return nil
	}
	t.Cleanup(func() { initializeCandleTextEmbedding = originalInit })
}

func ensureTextEmbeddingPlansConcurrently(
	t *testing.T,
	runtime *textEmbeddingRuntime,
	cfg *config.RouterConfig,
	plans []embedding.RuntimePlan,
) {
	t.Helper()
	errs := make(chan error, len(plans))
	var wg sync.WaitGroup
	for _, plan := range plans {
		wg.Add(1)
		go func(plan embedding.RuntimePlan) {
			defer wg.Done()
			errs <- runtime.ensureInitialized(cfg, plan)
		}(plan)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("ensureInitialized: %v", err)
		}
	}
}

func TestTextEmbeddingRuntimeInitializesCandleModelUnionOnceUnderConcurrentFamilies(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "")
	t.Setenv("EMBEDDING_MODEL_OVERRIDE", "")
	cfg := concurrentCandleUnionConfig()
	basePlan := mustResolveTextEmbeddingRuntimePlan(t, cfg, "base", config.EmbeddingModelTypeQwen3)
	preferencePlan := mustResolveTextEmbeddingRuntimePlan(t, cfg, "preference", "mmbert")

	var initCalls atomic.Int32
	installExpectedCandleUnionInitializer(t, cfg, &initCalls)
	runtime := &textEmbeddingRuntime{}
	ensureTextEmbeddingPlansConcurrently(t, runtime, cfg, []embedding.RuntimePlan{basePlan, preferencePlan})

	if initCalls.Load() != 1 {
		t.Fatalf("Candle init calls = %d, want 1", initCalls.Load())
	}

	unexpectedPlan := mustResolveTextEmbeddingRuntimePlan(t, cfg, "unexpected", "gemma")
	if err := runtime.ensureInitialized(cfg, unexpectedPlan); err == nil || !strings.Contains(err.Error(), "runtime conflict") {
		t.Fatalf("unprepared Candle hot reload error = %v, want runtime conflict", err)
	}
	if initCalls.Load() != 1 {
		t.Fatalf("conflicting hot reload called native initializer %d times", initCalls.Load())
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
