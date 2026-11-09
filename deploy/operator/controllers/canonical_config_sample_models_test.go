package controllers

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	"k8s.io/apimachinery/pkg/api/resource"
	kubeyaml "sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDefaultOperatorSamplesSelectTheDefaultModel(t *testing.T) {
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
			if cfg.MmBertModelPath != "models/Vela-1.0-Encoder-307M-Embedding" {
				t.Fatalf("embedding artifact = %q", cfg.MmBertModelPath)
			}
			for _, consumer := range []string{"prompt_guard", "domain_classifier", "pii_classifier"} {
				if deployment := resolvedDeployment(t, cfg, consumer); deployment.Artifact != vela3Fast.RepoID {
					t.Fatalf("%s runs %+v, want vela3-fast", consumer, deployment)
				}
			}
			if cfg.PromptGuard.Threshold != .88 || cfg.PIIMappingPath != "" {
				t.Fatalf("guard threshold = %v, PII mapping = %q", cfg.PromptGuard.Threshold, cfg.PIIMappingPath)
			}
		})
	}
}

var vela3Fast = routerconfig.GetModelByPath(routerconfig.Vela3SignalModel)

func resolvedDeployment(t *testing.T, cfg *routerconfig.RouterConfig, consumer string) routerconfig.ModelDeployment {
	t.Helper()
	resolved, ok, err := cfg.ResolveTask(consumer)
	if err != nil || !ok {
		t.Fatalf("%s resolves to no deployment: %v", consumer, err)
	}
	return resolved.Deployment
}

// vela3FastMemoryLimit is the least Router memory limit that holds vela3-fast
// while it loads; 8Gi is OOM-killed.
var vela3FastMemoryLimit = resource.MustParse("12Gi")

func TestOperatorSamplesRunTheGuardAtItsModelsOperatingPoint(t *testing.T) {
	names, err := filepath.Glob(filepath.Join("..", "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	guarded := 0
	for _, path := range names {
		name := filepath.Base(path)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var sample vllmv1alpha1.SemanticRouter
		if err := kubeyaml.Unmarshal(data, &sample); err != nil {
			t.Fatal(err)
		}
		if sample.Kind != "SemanticRouter" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			canonical := operatorModelConfig(t, data)
			raw, err := yaml.Marshal(canonical)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := routerconfig.ParseYAMLBytes(raw)
			if err != nil {
				t.Fatal(err)
			}
			if !cfg.PromptGuard.Enabled {
				return
			}
			guarded++
			guard, ok, err := cfg.ResolveTask("prompt_guard")
			if err != nil || !ok {
				t.Fatalf("the guard resolves to no deployment: %v", err)
			}
			want, _ := routerconfig.ThresholdsOfDeployment(guard.Name, guard.Deployment)
			if cfg.PromptGuard.Threshold != want.PromptGuard {
				t.Fatalf("guard threshold %v on %s, want its operating point %v",
					cfg.PromptGuard.Threshold, guard.Name, want.PromptGuard)
			}
			if guard.Deployment.Artifact != vela3Fast.RepoID {
				return
			}
			// No limit leaves the Router uncapped, which also holds the model.
			limit := sample.Spec.Resources.Limits.Memory()
			if !limit.IsZero() && limit.Cmp(vela3FastMemoryLimit) < 0 {
				t.Fatalf("memory limit %s cannot hold vela3-fast; want at least %s",
					limit, &vela3FastMemoryLimit)
			}
		})
	}
	if guarded == 0 {
		t.Fatal("no sample runs the prompt guard")
	}
}

func TestTypedOperatorSampleServesThePinnedVelaPII(t *testing.T) {
	canonical := sampleModelConfig(t, "vllm.ai_v1alpha1_semanticrouter_model_runtime.yaml")
	binding, bound := canonical.Routing.ModelBindings["pii"]
	if !bound {
		t.Fatalf("the sample binds no pii deployment: %+v", canonical.Routing.ModelBindings)
	}
	if binding.MappingPath != "" {
		t.Fatalf("PII binding names a mapping file instead of the served labels: %q", binding.MappingPath)
	}
	release := routerconfig.GetModelByPath("models/Vela-1.0-Encoder-307M-PII")
	deployment := canonical.Global.ModelCatalog.Deployments[binding.Deployment]
	if !deployment.IsModelRuntime() || deployment.Artifact != release.RepoID || deployment.Revision != release.Revision {
		t.Fatalf("PII deployment is not the pinned Vela release on the model runtime: %+v", deployment)
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
		lines = append(lines, strings.TrimPrefix(line, "          "))
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
	if cfg.MmBertModelPath != defaults.MmBertModelPath || cfg.EmbeddingConfig.ModelType != "mmbert" || !cfg.UseCPU {
		t.Fatalf("CI embedding must use the canonical CPU model: %+v", cfg.EmbeddingModels)
	}
	for _, consumer := range []string{"domain_classifier", "pii_classifier"} {
		resolved, _, err := cfg.ResolveTask(consumer)
		if err != nil || resolved.Name != routerconfig.DefaultDecisionDeployment || resolved.Deployment.Device != "cpu" {
			t.Fatalf("CI %s must run the default CPU decision model: %+v (%v)", consumer, resolved, err)
		}
	}
	if !cfg.CategoryModel.UseCPU || !cfg.PIIModel.UseCPU {
		t.Fatalf("CI classifiers must inherit canonical CPU defaults: %+v, %+v", cfg.CategoryModel, cfg.PIIModel)
	}
}

func TestOperatorClassifierOverrideKeepsUnspecifiedModels(t *testing.T) {
	reconciler := &SemanticRouterReconciler{}
	defaults := operatorModelModules().Classifier
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
