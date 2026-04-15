package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type countingEmbeddingInitializer struct {
	calls int
}

func (i *countingEmbeddingInitializer) Init(string, string, string, bool) error {
	i.calls++
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
