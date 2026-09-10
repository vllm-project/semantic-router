package config

import "testing"

var admissionMigratedFullConfigAssets = []string{
	"config/config.yaml",
	"deploy/kubernetes/istio/config.yaml",
	"deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.local",
	"deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.openai",
	"deploy/kubernetes/observability/dashboard/config.yaml",
	repoRel("e2e", "config", "config.modality-routing.yaml"),
	repoRel("e2e", "config", "config.testing.yaml"),
}

var admissionMigratedEmbeddedConfigAssets = []string{
	"deploy/kserve/configmap-router-config.yaml",
	"deploy/kserve/configmap-router-config-simulator.yaml",
}

var admissionMigratedValuesConfigAssets = []string{
	"deploy/helm/semantic-router/values.yaml",
	"deploy/kubernetes/agentgateway/semantic-router-values/values.yaml",
	"deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml",
	"deploy/kubernetes/aibrix/semantic-router-values/values.yaml",
	"deploy/kubernetes/dynamo/semantic-router-values/values.yaml",
	"deploy/kubernetes/istio/semantic-router-values/values.yaml",
	"deploy/kubernetes/llm-d/semantic-router-values/values.yaml",
}

func TestMaintainedConfigAssetsKeepConcurrencyBound(t *testing.T) {
	for _, rel := range admissionMigratedFullConfigAssets {
		t.Run(rel, func(t *testing.T) {
			assertAdmissionBoundForEveryDeployment(t, rel, readMaintainedConfigAsset(t, rel))
		})
	}
	for _, rel := range admissionMigratedEmbeddedConfigAssets {
		t.Run(rel, func(t *testing.T) {
			assertAdmissionBoundForEveryDeployment(t, rel, readEmbeddedConfigAsset(t, rel))
		})
	}
	for _, rel := range admissionMigratedValuesConfigAssets {
		t.Run(rel, func(t *testing.T) {
			assertAdmissionBoundForEveryDeployment(t, rel, readValuesConfigAsset(t, rel))
		})
	}
}

func assertAdmissionBoundForEveryDeployment(t *testing.T, rel string, data []byte) {
	t.Helper()
	cfg, err := ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("%s failed to parse: %v", rel, err)
	}
	for deployment := range admissionDeploymentKeys {
		if cfg.ModelAdmission[deployment].MaxConcurrency <= 0 {
			t.Fatalf("%s leaves %s without a concurrency bound after migration", rel, deployment)
		}
	}
}
