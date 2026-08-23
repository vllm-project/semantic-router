package modeldownload

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	testCategoryModelPath  = "models/mmbert32k-intent-classifier-merged"
	testPIIModelPath       = "models/mmbert32k-pii-detector-merged"
	testJailbreakModelPath = "models/mmbert32k-jailbreak-detector-merged"
)

// newMmBERT32KClassifierConfig mirrors the shipped defaults: all three classifiers on the
// mmBERT-32K backend, pointed at merged (non-LoRA) model directories, with routing that
// actually consumes each signal so the models survive the optional-model feature gates.
func newMmBERT32KClassifierConfig() *config.RouterConfig {
	return &config.RouterConfig{
		MoMRegistry: map[string]string{
			testCategoryModelPath:  "llm-semantic-router/mmbert32k-intent-classifier-merged",
			testPIIModelPath:       "llm-semantic-router/mmbert32k-pii-detector-merged",
			testJailbreakModelPath: "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		},
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             testCategoryModelPath,
					UseMmBERT32K:        true,
					CategoryMappingPath: testCategoryModelPath + "/category_mapping.json",
				},
				PIIModel: config.PIIModel{
					ModelID:        testPIIModelPath,
					UseMmBERT32K:   true,
					PIIMappingPath: testPIIModelPath + "/pii_type_mapping.json",
				},
			},
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              testJailbreakModelPath,
				Variant:              config.PromptGuardVariantMmBERT32K,
				JailbreakMappingPath: testJailbreakModelPath + "/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{Name: "domain-route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"}},
				{Name: "pii-route", Rules: config.RuleNode{Type: config.SignalTypePII, Name: "pii"}},
				{Name: "jailbreak-route", Rules: config.RuleNode{Type: config.SignalTypeJailbreak, Name: "jailbreak"}},
			},
		},
	}
}

// TestBuildModelSpecsRequiresClassifierRuntimeWeights guards #2669. On the mmBERT-32K
// backend all three classifiers load through TraditionalModernBertTokenClassifier, which
// hard-reads config.json, tokenizer.json, and model.safetensors from the model root. Those
// files are the completeness contract; without them a half-downloaded directory satisfies
// the nested-weight heuristic in IsModelComplete and is never re-fetched.
func TestBuildModelSpecsRequiresClassifierRuntimeWeights(t *testing.T) {
	specs, err := BuildModelSpecs(newMmBERT32KClassifierConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	for _, path := range []string{testCategoryModelPath, testPIIModelPath, testJailbreakModelPath} {
		spec, ok := findSpecByPath(specs, path)
		if !ok {
			t.Fatalf("BuildModelSpecs() produced no spec for %q; got %#v", path, specs)
		}
		for _, want := range []string{"config.json", "model.safetensors", "tokenizer.json"} {
			if !slices.Contains(spec.RequiredFiles, want) {
				t.Errorf("%s RequiredFiles = %#v, missing %q", path, spec.RequiredFiles, want)
			}
		}
	}
}

// TestPartialClassifierDirReportedIncomplete reproduces the #2669 symptom: an interrupted
// download leaves the companion mapping and a nested adapter blob behind, which satisfies
// the recursive *.safetensors heuristic even though the runtime weights never arrived. The
// directory must read as incomplete so the snapshot is fetched again.
func TestPartialClassifierDirReportedIncomplete(t *testing.T) {
	specs, err := BuildModelSpecs(newMmBERT32KClassifierConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	spec, ok := findSpecByPath(specs, testPIIModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() produced no spec for %q", testPIIModelPath)
	}

	dir := t.TempDir()
	writeModelFile(t, dir, "config.json", "{}")
	writeModelFile(t, dir, "pii_type_mapping.json", "{}")
	writeModelFile(t, filepath.Join(dir, "lora_adapter"), "adapter_model.safetensors", "adapter-bytes")

	complete, err := IsModelComplete(dir, spec.RequiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if complete {
		t.Fatalf("partial classifier dir reported complete; the runtime hard-loads model.safetensors and would fail at init")
	}
}

// TestCompleteClassifierDirReportedComplete is the control: a fully downloaded directory
// must not be re-fetched on every restart.
func TestCompleteClassifierDirReportedComplete(t *testing.T) {
	specs, err := BuildModelSpecs(newMmBERT32KClassifierConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	spec, _ := findSpecByPath(specs, testPIIModelPath)

	dir := t.TempDir()
	writeModelFile(t, dir, "config.json", "{}")
	writeModelFile(t, dir, "tokenizer.json", "{}")
	writeModelFile(t, dir, "model.safetensors", "weights")
	writeModelFile(t, dir, "pii_type_mapping.json", "{}")

	complete, err := IsModelComplete(dir, spec.RequiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if !complete {
		t.Fatalf("complete classifier dir reported incomplete; RequiredFiles = %#v", spec.RequiredFiles)
	}
}

// TestLoRAClassifierKeepsHeuristicCompleteness pins the carve-out. Off the mmBERT-32K path,
// PII and jailbreak initialisation auto-detects LoRA models, whose directories carry
// adapter weights instead of a root model.safetensors. Demanding the root weights there
// would put a valid model into a permanent re-download loop.
func TestLoRAClassifierKeepsHeuristicCompleteness(t *testing.T) {
	cfg := newMmBERT32KClassifierConfig()
	cfg.Classifier.CategoryModel.UseMmBERT32K = false
	cfg.Classifier.PIIModel.UseMmBERT32K = false
	cfg.InlineModels.PromptGuard.Variant = config.PromptGuardVariantCandle

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	for _, path := range []string{testCategoryModelPath, testPIIModelPath, testJailbreakModelPath} {
		spec, ok := findSpecByPath(specs, path)
		if !ok {
			t.Fatalf("BuildModelSpecs() produced no spec for %q", path)
		}
		if slices.Contains(spec.RequiredFiles, "model.safetensors") {
			t.Errorf("%s requires model.safetensors on the LoRA-capable backend; RequiredFiles = %#v", path, spec.RequiredFiles)
		}
	}
}
