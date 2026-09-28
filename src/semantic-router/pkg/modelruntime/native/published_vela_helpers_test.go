package native

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type publishedVelaModel struct {
	name, path, override string
}

func publishedVelaModels(t *testing.T) []publishedVelaModel {
	t.Helper()
	defaults := config.DefaultGlobalConfig()
	system := config.DefaultSystemModels()
	models := []publishedVelaModel{
		{"Domain", defaults.CategoryModel.ModelID, "VLLM_SR_DOMAIN_MODEL"},
		{"Guard", defaults.PromptGuard.ModelID, "VLLM_SR_JAILBREAK_MODEL"},
		{"PII", defaults.PIIModel.ModelID, "VLLM_SR_PII_MODEL"},
		{"FactCheck", defaults.HallucinationMitigation.FactCheckModel.ModelID, "VLLM_SR_FACTCHECK_MODEL"},
		{"Feedback", defaults.FeedbackDetector.ModelID, "VLLM_SR_FEEDBACK_MODEL"},
		{"Safety", system.Safety, "VLLM_SR_SAFETY_MODEL"},
		{"Hazard", system.Hazard, "VLLM_SR_HAZARD_MODEL"},
		{"Embedding", defaults.MmBertModelPath, "VLLM_SR_EMBEDDING_MODEL"},
	}
	// These optional modules do not both have a path in DefaultSystemModels.
	for _, family := range []struct {
		name     string
		purpose  config.ModelPurpose
		override string
	}{
		{"Modality", config.PurposeModalityDetection, "VLLM_SR_MODALITY_MODEL"},
		{"Reranker", config.PurposeReranking, "VLLM_SR_RERANKER_MODEL"},
	} {
		var path string
		for _, model := range config.GetModelsByPurpose(family.purpose) {
			for _, tag := range model.Tags {
				if tag == "vela" {
					if path != "" {
						t.Fatalf("multiple published Vela defaults for %s", family.name)
					}
					path = model.LocalPath
				}
			}
		}
		if path == "" {
			t.Fatalf("missing published Vela registry entry for %s", family.name)
		}
		models = append(models, publishedVelaModel{family.name, path, family.override})
	}
	return models
}

func publishedVelaSpec(t *testing.T, model publishedVelaModel) (config.ResolvedModelBinding, *config.ModelSpec) {
	t.Helper()
	registered := config.GetModelByPath(model.path)
	if registered == nil || len(registered.Revision) != 40 {
		t.Fatalf("%s requires an immutable registry entry", model.name)
	}
	path := os.Getenv(model.override)
	if path == "" {
		if os.Getenv("VLLM_SR_REQUIRE_MODEL_TESTS") == "1" {
			t.Fatalf("required published %s model needs %s from the model manifest", model.name, model.override)
		}
		t.Skipf("published %s inference requires an explicit %s; use make test-models", model.name, model.override)
	}
	path, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("required published %s model at %q: %v", model.name, path, err)
	}
	if !info.IsDir() {
		t.Fatalf("published model path is not a directory: %s", path)
	}
	provider := os.Getenv("VLLM_SR_MODEL_TEST_PROVIDER")
	if provider == "" {
		provider, _ = config.DefaultModelExecution(true)
	}
	if provider != "candle" && provider != "ort" {
		t.Fatalf("unsupported VLLM_SR_MODEL_TEST_PROVIDER=%q", provider)
	}
	t.Logf("published family=%s repo=%s revision=%s provider=%s path=%s", model.name, registered.RepoID, registered.Revision, provider, path)
	return config.ResolvedModelBinding{
		Recipe: "published-vela", Name: model.name,
		Binding: config.ModelBinding{Deployment: model.name, Adapter: "mmbert32k", Contract: config.RemoteClassifierContractLabelDistribution},
		Deployment: config.ModelDeployment{
			Artifact: path, Revision: registered.Revision, Provider: provider, Device: "cpu", Precision: "native",
			Input: config.ModelInputBudget{MaxTokens: 2048, Overflow: "reject"},
		},
	}, registered
}

func publishedVelaLabels(t *testing.T, path string) []string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		Labels map[string]string `json:"id2label"`
	}
	if err := json.Unmarshal(data, &metadata); err != nil {
		t.Fatal(err)
	}
	labels := make([]string, len(metadata.Labels))
	for index := range labels {
		labels[index] = metadata.Labels[strconv.Itoa(index)]
		if labels[index] == "" {
			t.Fatalf("published mapping is missing class %d", index)
		}
	}
	return labels
}

func assertPublishedVelaCapability(t *testing.T, spec config.ResolvedModelBinding, capability binding.Capability, labels []string) {
	t.Helper()
	if capability.Provider != spec.Deployment.Provider || capability.Device != "cpu" || capability.Contract != spec.Binding.Contract || capability.Limits.EffectiveTokens() != spec.Deployment.Input.MaxTokens {
		t.Fatalf("actual capability differs from prepared CPU task: %+v", capability)
	}
	if labels != nil && !reflect.DeepEqual(capability.Labels, labels) {
		t.Fatalf("prepared label order=%v, published config=%v", capability.Labels, labels)
	}
	t.Logf("actual capability=%+v", capability)
}

func assertPublishedVelaUsage(t *testing.T, usage *tasks.InputUsage, long bool) {
	t.Helper()
	if usage == nil || usage.OriginalTokens <= 0 || usage.Truncated || usage.ProcessedTokens != usage.OriginalTokens {
		t.Fatalf("model did not process the complete input: %+v", usage)
	}
	if long && usage.ProcessedTokens <= 512 {
		t.Fatalf("long regression did not cross 512 tokens: %+v", usage)
	}
}

func assertPublishedVelaDistribution(t *testing.T, values []float32, classes int) {
	t.Helper()
	if len(values) != classes || classes < 2 {
		t.Fatalf("incomplete distribution: %v, want %d classes", values, classes)
	}
	if err := validateDistribution("", tasks.LabelDistribution{Probabilities: values}); err != nil {
		t.Fatalf("invalid real-model distribution: %v: %v", values, err)
	}
}

func publishedVelaLabel(values []float32, labels []string) string {
	best := 0
	for index, value := range values {
		if value > values[best] {
			best = index
		}
	}
	label := strings.ToLower(labels[best])
	if label == "sat" {
		label = "satisfied"
	}
	return label
}

func assertPublishedVelaVector(t *testing.T, vector []float32, dimension int) {
	t.Helper()
	if len(vector) != dimension {
		t.Fatalf("embedding has %d dimensions, want %d", len(vector), dimension)
	}
	var squaredNorm float64
	for _, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatal("embedding is not finite")
		}
		squaredNorm += float64(value) * float64(value)
	}
	if math.Abs(math.Sqrt(squaredNorm)-1) > 1e-3 {
		t.Fatalf("embedding is not normalized: norm=%g", math.Sqrt(squaredNorm))
	}
}

func publishedHazardReference(t *testing.T) *config.OperatingPointReference {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var source struct {
		Routing struct {
			Bindings map[string]config.ModelBinding `yaml:"model_bindings"`
		} `yaml:"routing"`
	}
	if err := yaml.Unmarshal(data, &source); err != nil {
		t.Fatal(err)
	}
	ref := source.Routing.Bindings["classifier.content-risk"].OperatingPoint
	if ref == nil || ref.Path == "" || len(ref.SHA256) != 64 {
		t.Fatal("canonical config has no pinned Hazard operating point")
	}
	return ref
}
