package config

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func TestDashboardProfileHelmConfigLoadsAsCanonicalRuntime(t *testing.T) {
	helm, err := exec.LookPath("helm")
	if err != nil {
		t.Skip("Helm is required for the rendered ConfigMap check; the maintained-asset test validates the source config unconditionally")
	}
	chart := dashboardConfigChart(t)
	values, err := filepath.Abs(filepath.Join("..", "..", "..", "..", "e2e", "profiles", "dashboard", "values.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	output, err := exec.Command(helm, "template", "semantic-router", chart,
		"--namespace", "vllm-semantic-router-system", "--values", values).CombinedOutput()
	if err != nil {
		t.Fatalf("render Dashboard Router ConfigMap: %v\n%s", err, output)
	}
	configMap := decodeYAMLMap(t, output, "rendered Dashboard ConfigMap")
	data := mustAssetMapValue(t, configMap, "data", "rendered Dashboard ConfigMap")
	configYAML, ok := data["config.yaml"].(string)
	if !ok || configYAML == "" {
		t.Fatal("rendered ConfigMap must contain config.yaml")
	}
	cfg, err := ParseYAMLBytes([]byte(configYAML))
	if err != nil {
		t.Fatalf("Dashboard ConfigMap cannot start the strict Router runtime: %v", err)
	}
	if cfg.DefaultModel != "base-model" || len(cfg.ModelConfig) != 1 {
		t.Fatalf("Dashboard must only use its simulator model, got default=%q models=%v", cfg.DefaultModel, cfg.ModelConfig)
	}
	if len(cfg.Decisions) != 1 || len(cfg.Decisions[0].ModelRefs) != 1 {
		t.Fatalf("expected one simulator route, got %v", cfg.Decisions)
	}
	ref := cfg.Decisions[0].ModelRefs[0]
	if ref.Model != "base-model" || ref.LoRAName != "general-expert" {
		t.Fatalf("Dashboard simulator route changed: %+v", ref)
	}
	if len(cfg.VLLMEndpoints) != 1 || cfg.VLLMEndpoints[0].Address != "vllm-llama3-8b-instruct.default.svc.cluster.local" || cfg.VLLMEndpoints[0].Port != 8000 {
		t.Fatalf("Dashboard route must bind its actual simulator Service: %+v", cfg.VLLMEndpoints)
	}
	if cfg.PromptGuard.Enabled || cfg.Tools.Enabled || cfg.SemanticCache.Enabled {
		t.Fatal("Dashboard API profile must not inherit optional prompt guard, tools or cache from chart defaults")
	}
}

// Exercise the actual chart's ConfigMap and helpers without fetching optional
// dependency charts. No dependency resources participate in this template.
func dashboardConfigChart(t *testing.T) string {
	t.Helper()
	const chartSource = "deploy/helm/semantic-router"
	chart := t.TempDir()
	metadata := decodeYAMLMap(t, mustReadRepoFile(t, chartSource+"/Chart.yaml"), "Chart.yaml")
	delete(metadata, "dependencies")
	encoded, marshalErr := yamlv3.Marshal(metadata)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	if err := os.WriteFile(filepath.Join(chart, "Chart.yaml"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(chart, "templates"), 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"values.yaml", "templates/_helpers.tpl", "templates/configmap.yaml"} {
		if err := os.WriteFile(filepath.Join(chart, name), mustReadRepoFile(t, chartSource+"/"+name), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return chart
}
