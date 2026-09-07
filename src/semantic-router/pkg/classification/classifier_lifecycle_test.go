package classification

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

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

// The response-stage consumers of the jailbreak classifier are never named by
// a decision rule, so the signal-type walk alone would leave the model
// uninitialized and every response-stage detection unavailable.
func TestInitializeRuntimeInitializesJailbreakClassifierForResponseStageConsumers(t *testing.T) {
	keywordDecisions := func(plugins ...config.DecisionPlugin) []config.Decision {
		return []config.Decision{{
			Name:    "keyword-route",
			Rules:   config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{{Type: config.SignalTypeKeyword, Name: "probe"}}},
			Plugins: plugins,
		}}
	}
	tests := []struct {
		name      string
		rules     []config.JailbreakRule
		decisions []config.Decision
	}{
		{
			name:      "response-direction rule",
			rules:     []config.JailbreakRule{{Name: "unsafe_completion", Direction: config.SignalDirectionResponse}},
			decisions: keywordDecisions(),
		},
		{
			name: "response_jailbreak plugin owns detection",
			decisions: keywordDecisions(config.DecisionPlugin{
				Type: "response_jailbreak",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled": true,
				}),
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
						Signals:   config.Signals{JailbreakRules: tt.rules},
						Decisions: tt.decisions,
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
			if categoryInitializer.calls != 0 || piiInitializer.calls != 0 {
				t.Fatalf("expected the unused request-stage initializers to be skipped, got category=%d pii=%d", categoryInitializer.calls, piiInitializer.calls)
			}
			if jailbreakInitializer.calls != 1 {
				t.Fatalf("expected the jailbreak initializer to run once for the response-stage consumer, got %d", jailbreakInitializer.calls)
			}
		})
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

func TestClassifierCloseClosesMCPCategoryClient(t *testing.T) {
	mcpClassifier, mockClient, _ := newTestMCPCategoryClassifier()
	mcpClassifier.client = mockClient
	mockClient.connected = true

	classifier, err := newClassifierWithOptions(
		&config.RouterConfig{},
		withMCPCategory(mcpClassifier, mcpClassifier),
	)
	if err != nil {
		t.Fatalf("newClassifierWithOptions() error = %v", err)
	}

	if err := classifier.Close(); err != nil {
		t.Fatalf("Classifier.Close() error = %v", err)
	}
	if mockClient.connected {
		t.Fatal("Classifier.Close() did not close the MCP category classifier's client; it leaks on every router reload")
	}
}

type connectingMCPInitializer struct {
	connected bool
	closes    int
	initiated chan struct{}
}

func newConnectingMCPInitializer() *connectingMCPInitializer {
	return &connectingMCPInitializer{initiated: make(chan struct{})}
}

func (i *connectingMCPInitializer) Init(*config.RouterConfig) error {
	i.connected = true
	close(i.initiated)
	return nil
}

func (i *connectingMCPInitializer) Close() error {
	i.connected = false
	i.closes++
	return nil
}

func (i *connectingMCPInitializer) state() (connected bool, closes int) {
	return i.connected, i.closes
}

type failAfterInitializer struct {
	waitFor <-chan struct{}
	raced   bool
}

func (i *failAfterInitializer) Init(string, bool, int) error {
	select {
	case <-i.waitFor:
	case <-time.After(5 * time.Second):
		i.raced = true
	}
	return errors.New("pii initializer failed")
}

func TestInitializeRuntimeReleasesAcquiredResourcesWhenALaterTaskFails(t *testing.T) {
	mcpInitializer := newConnectingMCPInitializer()
	piiInitializer := &failAfterInitializer{waitFor: mcpInitializer.initiated}

	classifier := &Classifier{
		Config: &config.RouterConfig{
			InlineModels: config.InlineModels{
				Classifier: config.Classifier{
					MCPCategoryModel: config.MCPCategoryModel{Enabled: true},
					PIIModel: config.PIIModel{
						ModelID:        "models/mmbert32k-pii-detector-merged",
						PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
					},
				},
			},
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{{
					Name: "guarded-route",
					Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
						{Type: config.SignalTypeDomain, Name: "billing"},
						{Type: config.SignalTypePII, Name: "contains_pii"},
					}},
				}},
			},
		},
		CategoryMapping:        &CategoryMapping{CategoryToIdx: map[string]int{"billing": 0, "support": 1}},
		PIIMapping:             &PIIMapping{LabelToIdx: map[string]int{"EMAIL_ADDRESS": 0, "PHONE_NUMBER": 1}},
		mcpCategoryInitializer: mcpInitializer,
		piiInitializer:         piiInitializer,
	}

	err := classifier.InitializeRuntime()
	if err == nil {
		t.Fatal("InitializeRuntime() = nil, want the PII initializer's error")
	}

	if piiInitializer.raced {
		t.Fatal("the PII task gave up waiting for the MCP task, so this run never produced a half-initialized classifier")
	}
	connected, closes := mcpInitializer.state()
	if closes == 0 {
		t.Fatal("InitializeRuntime() left the MCP category connection open after failing;" +
			" the caller discards the classifier, so it leaks on every failed build")
	}
	if connected {
		t.Fatal("MCP category initializer reports a live connection after rollback")
	}
	if closes != 1 {
		t.Fatalf("MCP category initializer closed %d times, want exactly 1", closes)
	}
}

func TestRecipeClassifiersInitializeRuntimeRollsBackEarlierRecipes(t *testing.T) {
	firstMCP := newConnectingMCPInitializer()
	secondMCP := newConnectingMCPInitializer()
	failing := &failAfterInitializer{waitFor: firstMCP.initiated}

	first := recipeClassifierWithMCP(firstMCP, nil)
	second := recipeClassifierWithMCP(secondMCP, failing)
	set := &RecipeClassifiers{
		byRecipe: map[config.RecipeName]*Classifier{"first": first, "second": second},
		order:    []config.RecipeName{"first", "second"},
	}

	if err := set.InitializeRuntime(); err == nil {
		t.Fatal("InitializeRuntime() = nil, want the second recipe's error")
	}

	if failing.raced {
		t.Fatal("the failing recipe never observed the first recipe connecting, so no earlier recipe was left initialized")
	}
	connected, closes := firstMCP.state()
	if closes == 0 {
		t.Fatal("the first recipe's MCP connection survived the second recipe's failure;" +
			" the caller discards the whole set, so it leaks on every failed build")
	}
	if connected {
		t.Fatal("the first recipe reports a live MCP connection after rollback")
	}
	if closes != 1 {
		t.Fatalf("the first recipe's MCP initializer closed %d times, want exactly 1", closes)
	}
	if _, closes := secondMCP.state(); closes != 1 {
		t.Fatalf("the failing recipe's MCP initializer closed %d times, want exactly 1", closes)
	}
}

func TestRecipeClassifiersCloseClosesEveryRecipe(t *testing.T) {
	defaultMCP := newConnectingMCPInitializer()
	otherMCP := newConnectingMCPInitializer()
	set := &RecipeClassifiers{
		byRecipe: map[config.RecipeName]*Classifier{
			config.DefaultRecipeName: recipeClassifierWithMCP(defaultMCP, nil),
			"other":                  recipeClassifierWithMCP(otherMCP, nil),
		},
		order: []config.RecipeName{config.DefaultRecipeName, "other"},
	}

	if err := set.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	if err := set.Close(); err != nil {
		t.Fatalf("RecipeClassifiers.Close() error = %v", err)
	}

	for name, initializer := range map[string]*connectingMCPInitializer{
		"default": defaultMCP,
		"other":   otherMCP,
	} {
		connected, closes := initializer.state()
		if closes != 1 {
			t.Errorf("%s recipe's MCP initializer closed %d times, want exactly 1", name, closes)
		}
		if connected {
			t.Errorf("%s recipe still reports a live MCP connection after Close", name)
		}
	}
}

func recipeClassifierWithMCP(mcp MCPCategoryInitializer, pii *failAfterInitializer) *Classifier {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				MCPCategoryModel: config.MCPCategoryModel{Enabled: true},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name: "guarded-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					{Type: config.SignalTypeDomain, Name: "billing"},
				}},
			}},
		},
	}
	classifier := &Classifier{
		Config:                 cfg,
		CategoryMapping:        &CategoryMapping{CategoryToIdx: map[string]int{"billing": 0, "support": 1}},
		mcpCategoryInitializer: mcp,
	}

	if pii != nil {
		cfg.PIIModel = config.PIIModel{
			ModelID:        "models/mmbert32k-pii-detector-merged",
			PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
		}
		cfg.Decisions[0].Rules.Conditions = append(cfg.Decisions[0].Rules.Conditions,
			config.RuleNode{Type: config.SignalTypePII, Name: "contains_pii"})
		classifier.PIIMapping = &PIIMapping{LabelToIdx: map[string]int{"EMAIL_ADDRESS": 0, "PHONE_NUMBER": 1}}
		classifier.piiInitializer = pii
	}
	return classifier
}
