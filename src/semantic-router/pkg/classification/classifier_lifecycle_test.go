package classification

import (
	"context"
	"errors"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type countingEmbeddingInitializer struct {
	calls  int
	onInit func()
}

func (i *countingEmbeddingInitializer) Init(string, string, string, bool, string, string) error {
	i.calls++
	if i.onInit != nil {
		i.onInit()
	}
	return nil
}

type countingCoreClassifierInitializer struct {
	calls int
}

func (i *countingCoreClassifierInitializer) Init(string, bool, ...int) error {
	i.calls++
	return nil
}

type countingPIIInitializer struct {
	calls int
}

func (i *countingPIIInitializer) Init(string, bool, int) error {
	i.calls++
	return nil
}

func TestNewClassifierWithOptionsDefersRuntimeInitialization(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				EmbeddingRules: []config.EmbeddingRule{
					{
						Name:       "support",
						Candidates: []string{"hello"},
					},
				},
			},
		},
	}
	initializer := &countingEmbeddingInitializer{}

	classifier, err := newClassifierWithOptions(
		cfg,
		withKeywordEmbeddingClassifier(initializer, &EmbeddingClassifier{}),
	)
	if err != nil {
		t.Fatalf("newClassifierWithOptions() error = %v", err)
	}
	if initializer.calls != 0 {
		t.Fatalf("initializer called during build: got %d, want 0", initializer.calls)
	}

	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	if initializer.calls != 1 {
		t.Fatalf("initializer calls = %d, want 1", initializer.calls)
	}
}

func TestClassifierBuildParallelismSerializesDefaultCandleRuntime(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")

	if got := classifierBuildParallelism(8); got != 1 {
		t.Fatalf("classifierBuildParallelism() = %d, want 1 for default candle runtime", got)
	}
}

func TestClassifierBuildParallelismSerializesExplicitCandleRuntime(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "candle")

	if got := classifierBuildParallelism(8); got != 1 {
		t.Fatalf("classifierBuildParallelism() = %d, want 1 for explicit candle runtime", got)
	}
}

func TestInitializeRuntimeWarmsEmbeddingCandidatesAfterBackendInit(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				EmbeddingRules: []config.EmbeddingRule{{
					Name:       "support",
					Candidates: []string{"hello"},
				}},
			},
		},
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					ModelType:         "mmbert",
					PreloadEmbeddings: true,
				},
			},
		},
	}

	backendInitialized := false
	initializer := &countingEmbeddingInitializer{onInit: func() {
		backendInitialized = true
	}}
	originalFunc := getEmbedding2DMatryoshka
	getEmbedding2DMatryoshka = func(text string, modelType string, targetLayer int, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		if !backendInitialized {
			return nil, errors.New("embedding preload ran before backend initialization")
		}
		return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(1.0, 0.0, 0.0)}, nil
	}
	t.Cleanup(func() {
		getEmbedding2DMatryoshka = originalFunc
	})

	embeddingClassifier, err := NewEmbeddingClassifier(cfg.EmbeddingRules, cfg.EmbeddingConfig)
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier() error = %v", err)
	}
	classifier, err := newClassifierWithOptions(
		cfg,
		withKeywordEmbeddingClassifier(initializer, embeddingClassifier),
	)
	if err != nil {
		t.Fatalf("newClassifierWithOptions() error = %v", err)
	}

	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	if initializer.calls != 1 {
		t.Fatalf("initializer calls = %d, want 1", initializer.calls)
	}
	if got := embeddingClassifier.GetPreloadStats(); got != 1 {
		t.Fatalf("preloaded candidates = %d, want 1", got)
	}
}

func TestInitializeRuntimeSkipsUnusedCoreSignalClassifiers(t *testing.T) {
	categoryInitializer := &countingCoreClassifierInitializer{}
	piiInitializer := &countingPIIInitializer{}
	jailbreakInitializer := &countingCoreClassifierInitializer{}
	classifier := &Classifier{
		Config: &config.RouterConfig{
			InlineModels: config.InlineModels{
				Classifier: config.Classifier{
					CategoryModel: config.CategoryModel{
						ModelID:             "models/mmbert32k-intent-classifier-merged",
						CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
					},
					PIIModel: config.PIIModel{
						ModelID:        "models/mmbert32k-pii-detector-merged",
						PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
					},
				},
				PromptGuard: config.PromptGuardConfig{
					Enabled:              true,
					ModelID:              "models/mmbert32k-jailbreak-detector-merged",
					JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
				},
			},
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{{
					Name:  "default",
					Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
				}},
			},
		},
		CategoryMapping:      &CategoryMapping{CategoryToIdx: map[string]int{"billing": 0, "support": 1}},
		PIIMapping:           &PIIMapping{LabelToIdx: map[string]int{"EMAIL_ADDRESS": 0, "PHONE_NUMBER": 1}},
		JailbreakMapping:     &JailbreakMapping{LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1}},
		categoryInitializer:  categoryInitializer,
		piiInitializer:       piiInitializer,
		jailbreakInitializer: jailbreakInitializer,
	}

	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	if categoryInitializer.calls != 0 || piiInitializer.calls != 0 || jailbreakInitializer.calls != 0 {
		t.Fatalf("expected unused signal initializers to be skipped, got category=%d pii=%d jailbreak=%d", categoryInitializer.calls, piiInitializer.calls, jailbreakInitializer.calls)
	}
}

func TestInitializeRuntimeInitializesCoreSignalClassifiersWhenUsed(t *testing.T) {
	categoryInitializer := &countingCoreClassifierInitializer{}
	piiInitializer := &countingPIIInitializer{}
	jailbreakInitializer := &countingCoreClassifierInitializer{}
	classifier := &Classifier{
		Config: &config.RouterConfig{
			InlineModels: config.InlineModels{
				Classifier: config.Classifier{
					CategoryModel: config.CategoryModel{
						ModelID:             "models/mmbert32k-intent-classifier-merged",
						CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
					},
					PIIModel: config.PIIModel{
						ModelID:        "models/mmbert32k-pii-detector-merged",
						PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
					},
				},
				PromptGuard: config.PromptGuardConfig{
					Enabled:              true,
					ModelID:              "models/mmbert32k-jailbreak-detector-merged",
					JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
				},
			},
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{{
					Name: "guarded-route",
					Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
						{Type: config.SignalTypeDomain, Name: "billing"},
						{Type: config.SignalTypePII, Name: "contains_pii"},
						{Type: config.SignalTypeJailbreak, Name: "detector"},
					}},
				}},
			},
		},
		CategoryMapping:      &CategoryMapping{CategoryToIdx: map[string]int{"billing": 0, "support": 1}},
		PIIMapping:           &PIIMapping{LabelToIdx: map[string]int{"EMAIL_ADDRESS": 0, "PHONE_NUMBER": 1}},
		JailbreakMapping:     &JailbreakMapping{LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1}},
		categoryInitializer:  categoryInitializer,
		piiInitializer:       piiInitializer,
		jailbreakInitializer: jailbreakInitializer,
	}

	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	if categoryInitializer.calls != 1 || piiInitializer.calls != 1 || jailbreakInitializer.calls != 1 {
		t.Fatalf("expected used signal initializers to run once, got category=%d pii=%d jailbreak=%d", categoryInitializer.calls, piiInitializer.calls, jailbreakInitializer.calls)
	}
}

func TestUnsupportedLocalHallucinationBackendIsExplicitlyDegraded(t *testing.T) {
	original := nativeBackendCapabilities
	t.Cleanup(func() { nativeBackendCapabilities = original })
	nativeBackendCapabilities = NativeBackendCapabilities{Name: "test-backend"}

	classifier := &Classifier{Config: newHallucinationLifecycleConfig(config.HallucinationBackendCandle)}
	if classifier.IsHallucinationDetectionEnabled() {
		t.Fatal("unsupported local hallucination backend must not advertise enabled capability")
	}

	var hallucinationTaskFound bool
	for _, task := range classifier.runtimeTasks() {
		if task.Name != "classifier.hallucination" {
			continue
		}
		hallucinationTaskFound = true
		if !task.BestEffort {
			t.Fatal("unsupported local hallucination task must degrade without aborting unrelated startup")
		}
		err := task.Run(context.Background())
		if err == nil || !strings.Contains(err.Error(), "does not support local hallucination detection") {
			t.Fatalf("unsupported local hallucination task error = %v", err)
		}
	}
	if !hallucinationTaskFound {
		t.Fatal("configured local hallucination task was silently omitted")
	}
	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("best-effort unsupported hallucination backend aborted startup: %v", err)
	}
	if classifier.IsHallucinationDetectorReady() {
		t.Fatal("unsupported local hallucination backend must not report ready")
	}
}

func TestEndpointHallucinationDoesNotDependOnNativeCapability(t *testing.T) {
	original := nativeBackendCapabilities
	t.Cleanup(func() { nativeBackendCapabilities = original })
	nativeBackendCapabilities = NativeBackendCapabilities{Name: "test-backend"}

	classifier := &Classifier{Config: newHallucinationLifecycleConfig(config.HallucinationBackendEndpoint)}
	if !classifier.IsHallucinationDetectionEnabled() {
		t.Fatal("endpoint hallucination backend should not depend on local binding capability")
	}
}

func TestPublicAuxiliaryConsumersAreOwnedOnlyByDefaultClassifier(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				Enabled: true,
				FactCheckModel: config.FactCheckModelConfig{
					ModelID: "models/test-fact-check",
				},
				HallucinationModel: config.HallucinationModelConfig{
					Backend: config.HallucinationBackendCandle,
					ModelID: "models/test-hallucination-detector",
				},
				NLIModel: config.NLIModelConfig{
					ModelID: "models/test-hallucination-explainer",
				},
			},
			FeedbackDetector: config.FeedbackDetectorConfig{
				Enabled: true,
				ModelID: "models/test-feedback",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				FactCheckRules:    []config.FactCheckRule{{Name: "verification-needed"}},
				UserFeedbackRules: []config.UserFeedbackRule{{Name: "correction-needed"}},
			},
		},
	}
	defaultClassifier := &Classifier{Config: cfg}
	if !defaultClassifier.needsHallucinationDetectorForRuntime() ||
		!defaultClassifier.needsLocalHallucinationNLIForRuntime() {
		t.Fatal("default API-only NLI configuration was omitted from runtime lifecycle")
	}
	defaultTasks := make(map[string]bool)
	for _, task := range defaultClassifier.runtimeTasks() {
		defaultTasks[task.Name] = true
	}
	for _, name := range []string{"classifier.fact_check", "classifier.hallucination", "classifier.feedback"} {
		if !defaultTasks[name] {
			t.Fatalf("default API-only runtime task %q was omitted: %v", name, defaultTasks)
		}
	}

	namedConfig := *cfg
	namedConfig.RoutingScope = "named"
	namedClassifier := &Classifier{Config: &namedConfig}
	if namedClassifier.needsHallucinationDetectorForRuntime() ||
		namedClassifier.needsLocalHallucinationNLIForRuntime() {
		t.Fatal("named classifier inherited the default public NLI API consumer")
	}
	for _, task := range namedClassifier.runtimeTasks() {
		if task.Name == "classifier.fact_check" ||
			task.Name == "classifier.hallucination" ||
			task.Name == "classifier.feedback" {
			t.Fatalf("named classifier inherited default API task %q", task.Name)
		}
	}
}

func TestSignalReadinessRequiresInitializedFactCheckAndFeedbackModels(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				FactCheckModel: config.FactCheckModelConfig{ModelID: "models/test-fact-check"},
			},
			FeedbackDetector: config.FeedbackDetectorConfig{
				Enabled: true,
				ModelID: "models/test-feedback",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				FactCheckRules:    []config.FactCheckRule{{Name: "verification-needed"}},
				UserFeedbackRules: []config.UserFeedbackRule{{Name: "correction-needed"}},
			},
			Decisions: []config.Decision{{
				Name: "model-backed-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					{Type: config.SignalTypeFactCheck, Name: "verification-needed"},
					{Type: config.SignalTypeUserFeedback, Name: "correction-needed"},
				}},
			}},
		},
	}
	classifier := &Classifier{Config: cfg}

	ready := classifier.signalReadiness()
	if ready[config.SignalTypeFactCheck] || ready[config.SignalTypeUserFeedback] {
		t.Fatal("configured but uninitialized model signals must not report ready")
	}

	classifier.factCheckClassifier = &FactCheckClassifier{initialized: true}
	classifier.feedbackDetector = &FeedbackDetector{initialized: true}
	ready = classifier.signalReadiness()
	if !ready[config.SignalTypeFactCheck] || !ready[config.SignalTypeUserFeedback] {
		t.Fatal("initialized model signals should report ready")
	}
}

func TestSignalReadinessAllowsContrastiveJailbreakWithoutPromptGuard(t *testing.T) {
	rule := config.JailbreakRule{Name: "contrastive-guard", Method: "contrastive"}
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{JailbreakRules: []config.JailbreakRule{rule}},
			},
		},
		contrastiveJailbreakClassifiers: map[string]*ContrastiveJailbreakClassifier{
			rule.Name: {},
		},
	}

	if !classifier.signalReadiness()[config.SignalTypeJailbreak] {
		t.Fatal("initialized contrastive jailbreak rules must not require Prompt Guard")
	}
	delete(classifier.contrastiveJailbreakClassifiers, rule.Name)
	if classifier.signalReadiness()[config.SignalTypeJailbreak] {
		t.Fatal("contrastive jailbreak rules without an initialized classifier must not report ready")
	}
}

func newHallucinationLifecycleConfig(backend string) *config.RouterConfig {
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				HallucinationModel: config.HallucinationModelConfig{
					Backend: backend,
					ModelID: "models/test-hallucination-detector",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name: "response-verification-route",
				Plugins: []config.DecisionPlugin{{
					Type: config.DecisionPluginHallucination,
					Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled": true,
					}),
				}},
			}},
		},
	}
}
