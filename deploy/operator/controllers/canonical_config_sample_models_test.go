package controllers

import (
	"context"
	"os"
	"path/filepath"
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
			embedding := cfg.ModelDeployments[cfg.EmbeddingDeployment]
			if cfg.EmbeddingDeployment != routerconfig.EmbeddingDefaultDeployment || embedding.Artifact != velaEmbedding.RepoID {
				t.Fatalf("embedding deployment = %s %+v", cfg.EmbeddingDeployment, embedding)
			}
			for _, signalType := range []string{routerconfig.SignalTypeJailbreak, routerconfig.SignalTypePII} {
				if deployment := applicationDeployment(t, cfg, signalType); deployment.Artifact != vela3Fast.RepoID {
					t.Fatalf("%s rules ask %+v, want vela3-fast", signalType, deployment)
				}
			}
		})
	}
}

var (
	vela3Fast     = routerconfig.GetModelByPath(routerconfig.Vela3SignalModel)
	velaEmbedding = routerconfig.GetModelByPath(routerconfig.VelaEmbeddingModel)
)

func applicationDeployment(t *testing.T, cfg *routerconfig.RouterConfig, signalType string) routerconfig.ModelDeployment {
	t.Helper()
	name, _ := cfg.ApplicationTypeDeployment(signalType)
	deployment, ok := cfg.ModelDeployments[name]
	if !ok {
		t.Fatalf("%s rules resolve to no deployment: %q", signalType, name)
	}
	return deployment
}

// vela3FastMemoryLimit is the least Router memory limit that holds vela3-fast
// while it loads; 8Gi is OOM-killed.
var vela3FastMemoryLimit = resource.MustParse("12Gi")

// A sample that asks vela3-fast must give the Router room to load it.
func TestOperatorSamplesHoldTheirDecisionModel(t *testing.T) {
	names, err := filepath.Glob(filepath.Join("..", "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatal(err)
	}
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
			asked := false
			for _, deployment := range routerconfig.ModelRuntimeDeploymentsInUse(cfg) {
				asked = asked || deployment.Artifact == vela3Fast.RepoID
			}
			if !asked {
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
}

func TestTypedOperatorSampleServesThePinnedVela2(t *testing.T) {
	canonical := sampleModelConfig(t, "vllm.ai_v1alpha1_semanticrouter_model_runtime.yaml")
	binding, bound := canonical.Routing.ModelBindings["pii"]
	if !bound {
		t.Fatalf("the sample binds no pii deployment: %+v", canonical.Routing.ModelBindings)
	}
	release := routerconfig.GetModelByPath(routerconfig.Vela2Model08B)
	deployment := canonical.Global.ModelCatalog.Deployments[binding.Deployment]
	if !deployment.IsModelRuntime() || deployment.Artifact != release.RepoID || deployment.Revision != release.Revision {
		t.Fatalf("the pii deployment is not the pinned Vela 2.0 release on the model runtime: %+v", deployment)
	}
	data, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := routerconfig.ParseYAMLBytes(data); err != nil {
		t.Fatalf("the router rejects the sample: %v", err)
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
	if embedding := cfg.ModelDeployments[cfg.EmbeddingDeployment]; cfg.EmbeddingDeployment != routerconfig.EmbeddingDefaultDeployment || embedding.Device != "cpu" {
		t.Fatalf("CI embedding must use the canonical CPU model: %s %+v", cfg.EmbeddingDeployment, embedding)
	}
	name, deployment, ok, err := cfg.DecisionModelDeployment()
	if err != nil || !ok || name != routerconfig.DefaultDecisionDeployment || deployment.Device != "cpu" {
		t.Fatalf("CI must run the default CPU decision model: %s %+v (%v)", name, deployment, err)
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
