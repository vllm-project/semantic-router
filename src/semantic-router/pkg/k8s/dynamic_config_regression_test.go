package k8s

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type dynamicConfigProfileFixture struct {
	baseCanonical    config.CanonicalConfig
	baseRouterConfig *config.RouterConfig
	pool             v1alpha1.IntelligentPool
	route            v1alpha1.IntelligentRoute
}

func TestDynamicConfigProfileConvertsToRuntimeConfig(t *testing.T) {
	fixture := mustLoadDynamicConfigProfileFixture(t)
	runtimeCfg := mustBuildDynamicConfigRuntimeConfig(t, fixture)
	assertDynamicConfigRuntimeConfig(t, runtimeCfg, fixture.route)
}

func repoRootForDynamicConfigTest(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve dynamic-config regression test path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../"))
}

func mustReadDynamicConfigTestFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %s: %v", path, err)
	}
	return data
}

func mustLoadDynamicConfigProfileFixture(t *testing.T) dynamicConfigProfileFixture {
	t.Helper()

	repoRoot := repoRootForDynamicConfigTest(t)
	baseConfigPath := filepath.Join(repoRoot, "e2e", "profiles", "dynamic-config", "values.yaml")
	baseConfigBytes := readValuesConfigBlock(t, baseConfigPath)

	baseRouterConfig, err := config.ParseYAMLBytes(baseConfigBytes)
	if err != nil {
		t.Fatalf("failed to parse dynamic-config base values: %v", err)
	}

	var fixture dynamicConfigProfileFixture
	fixture.baseRouterConfig = baseRouterConfig
	mustUnmarshalDynamicConfigYAML(t, baseConfigBytes, &fixture.baseCanonical, "dynamic-config base canonical config")
	mustUnmarshalDynamicConfigFile(
		t,
		filepath.Join(repoRoot, "e2e", "profiles", "dynamic-config", "crds", "intelligentpool.yaml"),
		&fixture.pool,
		"intelligent pool",
	)
	mustUnmarshalDynamicConfigFile(
		t,
		filepath.Join(repoRoot, "e2e", "profiles", "dynamic-config", "crds", "intelligentroute.yaml"),
		&fixture.route,
		"intelligent route",
	)

	return fixture
}

func mustBuildDynamicConfigRuntimeConfig(t *testing.T, fixture dynamicConfigProfileFixture) *config.RouterConfig {
	t.Helper()

	if err := validateCRDs(&fixture.pool, &fixture.route, fixture.baseRouterConfig); err != nil {
		t.Fatalf("dynamic-config CRDs failed validation: %v", err)
	}

	canonicalCfg, err := NewCRDConverter().Convert(&fixture.pool, &fixture.route, &fixture.baseCanonical)
	if err != nil {
		t.Fatalf("failed to convert dynamic-config CRDs: %v", err)
	}

	renderedCanonical, err := yaml.Marshal(canonicalCfg)
	if err != nil {
		t.Fatalf("failed to marshal converted canonical config: %v", err)
	}

	runtimeCfg, err := config.ParseYAMLBytes(renderedCanonical)
	if err != nil {
		t.Fatalf("failed to parse converted dynamic-config runtime config: %v", err)
	}

	return runtimeCfg
}

func assertDynamicConfigRuntimeConfig(t *testing.T, runtimeCfg *config.RouterConfig, route v1alpha1.IntelligentRoute) {
	t.Helper()

	if runtimeCfg.DefaultModel != "general-expert" {
		t.Fatalf("expected dynamic-config default model to remain LoRA alias, got %q", runtimeCfg.DefaultModel)
	}
	if len(runtimeCfg.Decisions) != len(route.Spec.Decisions) {
		t.Fatalf("expected %d decisions after dynamic-config conversion, got %d", len(route.Spec.Decisions), len(runtimeCfg.Decisions))
	}

	endpoints := runtimeCfg.GetEndpointsForModel("general-expert")
	if len(endpoints) == 0 {
		t.Fatal("expected dynamic-config LoRA alias to inherit endpoints from base model")
	}

	address, endpointName, found, detailErr := runtimeCfg.SelectBestEndpointWithDetailsForModel("general-expert")
	if detailErr != nil {
		t.Fatalf("failed to resolve dynamic-config default endpoint: %v", detailErr)
	}
	if !found || address == "" || endpointName == "" {
		t.Fatalf("expected dynamic-config default endpoint resolution, got address=%q endpoint=%q found=%t", address, endpointName, found)
	}
}

func readValuesConfigBlock(t *testing.T, path string) []byte {
	t.Helper()
	var root map[string]interface{}
	if err := yaml.Unmarshal(mustReadDynamicConfigTestFile(t, path), &root); err != nil {
		t.Fatalf("failed to decode %s: %v", path, err)
	}
	rawConfig, ok := root["config"]
	if !ok {
		t.Fatalf("%s is missing top-level config block", path)
	}
	data, err := yaml.Marshal(rawConfig)
	if err != nil {
		t.Fatalf("failed to marshal config block from %s: %v", path, err)
	}
	return data
}

func mustUnmarshalDynamicConfigFile(t *testing.T, path string, target interface{}, subject string) {
	t.Helper()
	mustUnmarshalDynamicConfigYAML(t, mustReadDynamicConfigTestFile(t, path), target, subject)
}

func mustUnmarshalDynamicConfigYAML(t *testing.T, data []byte, target interface{}, subject string) {
	t.Helper()
	if err := yaml.Unmarshal(data, target); err != nil {
		t.Fatalf("failed to decode %s: %v", subject, err)
	}
}
