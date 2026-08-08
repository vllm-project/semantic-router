package classification

import (
	"errors"
	"sync"
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
	originalFunc := getEmbeddingWithModelType
	getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		if !backendInitialized {
			return nil, errors.New("embedding preload ran before backend initialization")
		}
		return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(1.0, 0.0, 0.0)}, nil
	}
	t.Cleanup(func() {
		getEmbeddingWithModelType = originalFunc
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

func TestClassifierCloseClosesMCPCategoryClient(t *testing.T) {
	mcpClassifier, mockClient, _ := newTestMCPCategoryClassifier()
	mcpClassifier.client = mockClient
	mockClient.connected = true // simulate a prior successful Init()/Connect()

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

// connectingMCPInitializer stands in for the MCP category classifier: Init
// establishes a connection, Close releases it, and connected records which of
// the two ran last.
type connectingMCPInitializer struct {
	mu        sync.Mutex
	connected bool
	closes    int
	initiated chan struct{}
}

func newConnectingMCPInitializer() *connectingMCPInitializer {
	return &connectingMCPInitializer{initiated: make(chan struct{})}
}

func (i *connectingMCPInitializer) Init(*config.RouterConfig) error {
	i.mu.Lock()
	i.connected = true
	i.mu.Unlock()
	close(i.initiated)
	return nil
}

func (i *connectingMCPInitializer) Close() error {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.connected = false
	i.closes++
	return nil
}

func (i *connectingMCPInitializer) state() (connected bool, closes int) {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.connected, i.closes
}

// failAfterInitializer fails only once another initializer has reported
// success, so the classifier is genuinely half-initialized when the error
// surfaces. Without the handshake a fast failure could pre-empt the task whose
// resource this test is about, and the assertion would hold vacuously.
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
		// A pre-populated mapping keeps initializeMCPCategoryClassifier from
		// fetching one over MCP, which would need a live inference client. The
		// connection this test is about is established before that point.
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
