package controllers

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	kubeyaml "sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDefaultOperatorSamplesSelectVelaModels(t *testing.T) {
	for _, name := range []string{
		"vllm_v1alpha1_semanticrouter.yaml",
		"vllm.ai_v1alpha1_semanticrouter_openshift.yaml",
	} {
		t.Run(name, func(t *testing.T) {
			canonical := sampleModelConfig(t, name)
			data, err := yaml.Marshal(canonical)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := routerconfig.ParseYAMLBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			const prefix = "models/Vela-1.0-Encoder-307M-"
			if cfg.MmBertModelPath != prefix+"Embedding" {
				t.Fatalf("embedding artifact = %q", cfg.MmBertModelPath)
			}
			if cfg.PromptGuard.ModelID != prefix+"Guard" || cfg.PromptGuard.Variant != routerconfig.PromptGuardVariantMmBERT32K {
				t.Fatalf("guard model = %+v", cfg.PromptGuard)
			}
			if cfg.CategoryModel.ModelID != prefix+"Domain" || cfg.Variant != routerconfig.CategoryVariantMmBERT32K {
				t.Fatalf("domain model = %+v", cfg.CategoryModel)
			}
			if cfg.PIIModel.ModelID != prefix+"PII" || !cfg.PIIModel.UseMmBERT32K || cfg.PIIMappingPath != prefix+"PII/pii_mapping.json" {
				t.Fatalf("PII model = %+v", cfg.PIIModel)
			}
		})
	}
}

func TestTypedOperatorSampleUsesPublishedPIIMapping(t *testing.T) {
	canonical := sampleModelConfig(t, "vllm.ai_v1alpha1_semanticrouter_model_runtime.yaml")
	binding := canonical.Routing.ModelBindings["pii_classifier"]
	want := routerconfig.DefaultCanonicalGlobal().ModelCatalog.Modules.Classifier.PII.PIIMappingPath
	if binding.MappingPath != want {
		t.Fatalf("PII binding requests %q, published Vela mapping is %q", binding.MappingPath, want)
	}
}

func TestOperatorCIUsesCanonicalVelaModels(t *testing.T) {
	workflow, err := os.ReadFile("../../../.github/workflows/operator-ci.yml")
	if err != nil {
		t.Fatal(err)
	}
	_, resource, ok := strings.Cut(string(workflow), "          apiVersion: vllm.ai/v1alpha1\n")
	if !ok {
		t.Fatal("operator CI SemanticRouter resource is missing")
	}
	resource, _, ok = strings.Cut(resource, "\n          EOF")
	if !ok {
		t.Fatal("operator CI resource heredoc is not terminated")
	}
	lines := []string{"apiVersion: vllm.ai/v1alpha1"}
	for _, line := range strings.Split(resource, "\n") {
		// Cache backends vary across the matrix; model selection is shared.
		if strings.TrimSpace(line) != "${CACHE_CONFIG}" {
			lines = append(lines, strings.TrimPrefix(line, "          "))
		}
	}
	canonical := operatorModelConfig(t, []byte(strings.Join(lines, "\n")))
	data, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := routerconfig.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	defaults := routerconfig.DefaultGlobalConfig()
	system := routerconfig.DefaultSystemModels()
	if cfg.MmBertModelPath != defaults.MmBertModelPath || cfg.EmbeddingConfig.ModelType != "mmbert" || !cfg.UseCPU {
		t.Fatalf("CI embedding must use the canonical CPU model: %+v", cfg.EmbeddingModels)
	}
	if cfg.CategoryModel.ModelID != system.DomainClassifier || cfg.Variant != routerconfig.CategoryVariantMmBERT32K || !cfg.CategoryModel.UseCPU {
		t.Fatalf("CI domain classifier must inherit canonical CPU defaults: %+v", cfg.CategoryModel)
	}
	if cfg.PIIModel.ModelID != system.PIIClassifier || !cfg.PIIModel.UseMmBERT32K || !cfg.PIIModel.UseCPU {
		t.Fatalf("CI PII classifier must inherit canonical CPU defaults: %+v", cfg.PIIModel)
	}
}

func TestOperatorClassifierOverrideKeepsUnspecifiedModels(t *testing.T) {
	reconciler := &SemanticRouterReconciler{}
	defaults := routerconfig.DefaultCanonicalGlobal().ModelCatalog.Modules.Classifier
	withDomain, err := reconciler.convertClassifierModule(&vllmv1alpha1.ClassifierConfig{
		CategoryModel: &vllmv1alpha1.CategoryModelConfig{ModelID: "custom-domain"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if withDomain.Domain.ModelID != "custom-domain" || !reflect.DeepEqual(withDomain.PII, defaults.PII) {
		t.Fatalf("domain override replaced default PII: %+v", withDomain)
	}
	withPII, err := reconciler.convertClassifierModule(&vllmv1alpha1.ClassifierConfig{
		PIIModel: &vllmv1alpha1.PIIModelConfig{ModelID: "custom-pii"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if withPII.PII.ModelID != "custom-pii" || !reflect.DeepEqual(withPII.Domain, defaults.Domain) {
		t.Fatalf("PII override replaced default domain: %+v", withPII)
	}
}

func sampleModelConfig(t *testing.T, name string) *routerconfig.CanonicalConfig {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "config", "samples", name))
	if err != nil {
		t.Fatal(err)
	}
	return operatorModelConfig(t, data)
}

func operatorModelConfig(t *testing.T, data []byte) *routerconfig.CanonicalConfig {
	t.Helper()
	var sample vllmv1alpha1.SemanticRouter
	if err := kubeyaml.Unmarshal(data, &sample); err != nil {
		t.Fatal(err)
	}
	// Backend discovery is independent of model configuration and needs a
	// cluster. Exercise the actual sample's entire config without its endpoints.
	sample.Spec.VLLMEndpoints = nil
	reconciler := &SemanticRouterReconciler{}
	canonical, err := reconciler.buildCanonicalConfig(context.Background(), &sample)
	if err != nil {
		t.Fatal(err)
	}
	return canonical
}
