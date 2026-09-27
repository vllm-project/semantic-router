package classification

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func nativeLabelFixture(t *testing.T, task string) string {
	t.Helper()
	artifact, err := filepath.Abs(filepath.Join("..", "..", "..", "..", "onnx-binding", "instance", "testdata", task))
	if err != nil {
		t.Fatal(err)
	}
	return artifact
}

// Uses the real native runtime and checked-in ONNX graphs; no network or model
// download is needed. Run with CORE_NATIVE_FIXTURES=1 and an ORT shared library.
func TestNativeMappingCandidateKeepsPreviousModel(t *testing.T) {
	if os.Getenv("CORE_NATIVE_FIXTURES") != "1" {
		t.Skip("set CORE_NATIVE_FIXTURES=1 with the ORT runtime installed")
	}
	runtime := native.New(binding.NewPool())
	spec := func(task, contract string) config.ResolvedModelBinding {
		artifact := nativeLabelFixture(t, task)
		return config.ResolvedModelBinding{
			Recipe: "primary", Name: task,
			Binding:    config.ModelBinding{Deployment: task, Adapter: "mmbert", Contract: contract},
			Deployment: config.ModelDeployment{Artifact: artifact, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{Overflow: "reject"}},
		}
	}
	sequenceSpec := spec("sequence", config.RemoteClassifierContractLabelDistribution)
	first := &ownedSequenceBackend{runtime: runtime, spec: sequenceSpec, labels: []string{"negative", "positive"}}
	if initErr := first.Init("", true, 2); initErr != nil {
		t.Fatal(initErr)
	}
	t.Cleanup(func() { _ = first.Close() })
	before, callErr := first.Classify(context.Background(), "hello world")
	if callErr != nil {
		t.Fatal(callErr)
	}
	candidate := &ownedSequenceBackend{runtime: runtime, spec: sequenceSpec, labels: []string{"positive", "negative"}}
	if initErr := candidate.Init("", true, 2); !errors.Is(initErr, binding.ErrCapability) || candidate.handle != nil {
		t.Fatalf("mismapped native candidate became callable: %v", initErr)
	}
	after, repeatErr := first.Classify(context.Background(), "hello world")
	if repeatErr != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("failed candidate changed previous model: before=%+v after=%+v err=%v", before, after, repeatErr)
	}
	tokenSpec := spec("token", config.RemoteClassifierContractTokenSpans)
	tokens := &ownedTokenBackend{runtime: runtime, spec: tokenSpec, labels: []string{"O", "B-SECRET"}}
	if initErr := tokens.Init("", true, 2); initErr != nil {
		t.Fatal(initErr)
	}
	t.Cleanup(func() { _ = tokens.Close() })
	wrongTokens := &ownedTokenBackend{runtime: runtime, spec: tokenSpec, labels: []string{"O", "B-PERSON"}}
	if initErr := wrongTokens.Init("", true, 2); !errors.Is(initErr, binding.ErrCapability) || wrongTokens.handle != nil {
		t.Fatalf("wrong PII label was accepted: %v", initErr)
	}
	result, tokenErr := tokens.ClassifyTokens(context.Background(), "hello world")
	if tokenErr != nil || len(result.Entities) == 0 || result.Entities[0].EntityType != "SECRET" {
		t.Fatalf("previous token model lost its labels: %+v %v", result, tokenErr)
	}
}

func TestTwoLocalRulesOwnNativeModelsInOneRecipe(t *testing.T) {
	if os.Getenv("CORE_NATIVE_FIXTURES") != "1" {
		t.Skip("set CORE_NATIVE_FIXTURES=1 with the ORT runtime installed")
	}
	firstPath, secondPath := nativeLabelFixture(t, "sequence"), t.TempDir()
	for _, name := range []string{"model.onnx", "config.json", "tokenizer.json"} {
		data, readErr := os.ReadFile(filepath.Join(firstPath, name))
		if readErr != nil {
			t.Fatal(readErr)
		}
		if name == "config.json" {
			var modelConfig map[string]any
			if decodeErr := json.Unmarshal(data, &modelConfig); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			modelConfig["id2label"] = map[string]string{"0": "allow", "1": "deny"}
			var encodeErr error
			data, encodeErr = json.Marshal(modelConfig)
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
		}
		if writeErr := os.WriteFile(filepath.Join(secondPath, name), data, 0o600); writeErr != nil {
			t.Fatal(writeErr)
		}
	}
	cfg := &config.RouterConfig{}
	cfg.ClassifierRules = []config.ClassifierSignalRule{
		{Name: "intent.local", Type: config.ClassifierSignalTypeLocal, ModelPath: firstPath, UseCPU: true, Labels: []string{"negative", "positive"}},
		{Name: "guard.local", Type: config.ClassifierSignalTypeLocal, ModelPath: secondPath, UseCPU: true, Labels: []string{"allow", "deny"}},
	}
	cfg.ModelDeployments = map[string]config.ModelDeployment{}
	cfg.ModelBindings = map[string]config.ModelBinding{}
	for _, rule := range cfg.ClassifierRules {
		cfg.ModelDeployments[rule.Name] = config.ModelDeployment{Artifact: rule.ModelPath, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{Overflow: "reject"}}
		cfg.ModelBindings["classifier."+rule.Name] = config.ModelBinding{Deployment: rule.Name, Adapter: "mmbert", Contract: config.RemoteClassifierContractLabelDistribution}
	}
	models, prepareErr := newClassifierModelRuntime(cfg, native.New(binding.NewPool()))
	if prepareErr != nil {
		t.Fatal(prepareErr)
	}
	builder := &classifierOptionBuilder{cfg: models.cfg, models: models}
	apply, buildErr := builder.buildGenericClassifiersOption()
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	classifier := &Classifier{}
	apply(classifier)
	t.Cleanup(func() { closeLabelClassifiers(classifier.genericClassifiers) })
	first := classifier.genericClassifiers["intent.local"]
	second := classifier.genericClassifiers["guard.local"]
	for _, probe := range []struct {
		task  labelClassifier
		label string
	}{{first, "positive"}, {second, "deny"}} {
		result, inferErr := probe.task.Classify(context.Background(), "hello world")
		if inferErr != nil || len(result.Scores) != 2 || result.Scores[probe.label] == 0 {
			t.Fatalf("independent native labels lost: %+v %v", result, inferErr)
		}
	}
	if closeErr := first.(interface{ Close() error }).Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if _, inferErr := first.Classify(context.Background(), "hello world"); !errors.Is(inferErr, binding.ErrClosed) {
		t.Fatalf("closed rule remained callable: %v", inferErr)
	}
	if result, inferErr := second.Classify(context.Background(), "hello world"); inferErr != nil || result.Scores["deny"] == 0 {
		t.Fatalf("closing one native rule affected the other: %+v %v", result, inferErr)
	}
}

func TestLegacyStartupUsesProjectedNativeMapping(t *testing.T) {
	if os.Getenv("CORE_NATIVE_FIXTURES") != "1" {
		t.Skip("set CORE_NATIVE_FIXTURES=1 with the ORT runtime installed")
	}
	root := t.TempDir()
	mapping := filepath.Join(root, "labels.json")
	if writeErr := os.WriteFile(mapping, []byte(`{"category_to_idx":{"negative":0,"positive":1},"idx_to_category":{"0":"negative","1":"positive"}}`), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	cfg := &config.RouterConfig{}
	cfg.CategoryModel.ModelID = filepath.Join(root, "obsolete-model")
	cfg.CategoryMappingPath = filepath.Join(root, "obsolete-mapping.json")
	cfg.Decisions = []config.Decision{{Name: "positive", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "positive"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"native": {Provider: "ort", Artifact: nativeLabelFixture(t, "sequence"), Device: "cpu", Precision: "native", Input: config.ModelInputBudget{Overflow: "reject"}}}
	cfg.ModelBindings = map[string]config.ModelBinding{"domain_classifier": {Deployment: "native", Adapter: "mmbert", Contract: config.RemoteClassifierContractLabelDistribution, MappingPath: mapping}}
	classifier, initErr := NewLegacyClassifierFromConfig(cfg)
	if initErr != nil {
		t.Fatalf("startup read obsolete mapping before its binding: %v", initErr)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	if classifier.CategoryMapping.IdxToCategory["1"] != "positive" {
		t.Fatal("startup did not preserve projected model labels")
	}
	if cfg.CategoryMappingPath == mapping {
		t.Fatal("startup overwrote canonical default mapping")
	}
}
