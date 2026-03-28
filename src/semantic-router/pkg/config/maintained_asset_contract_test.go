package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

var maintainedFullConfigAssets = []string{
	"deploy/recipes/balance.yaml",
	"deploy/kubernetes/istio/config.yaml",
	"deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.local",
	"deploy/kubernetes/llmd-base/llmd+public-llm/config.yaml.openai",
	"deploy/kubernetes/observability/dashboard/config.yaml",
	"deploy/openshift/config-openshift.yaml",
	repoRel("e2e", "config", "config.agent-smoke.amd.yaml"),
	repoRel("e2e", "config", "config.agent-smoke.cpu.yaml"),
	repoRel("e2e", "config", "config.authz-rbac-demo.yaml"),
	repoRel("e2e", "config", "config.authz-rbac.yaml"),
	repoRel("e2e", "config", "config.e2e.yaml"),
	repoRel("e2e", "config", "config.hallucination.yaml"),
	repoRel("e2e", "config", "config.image-gen.yaml"),
	repoRel("e2e", "config", "config.memory-user.yaml"),
	repoRel("e2e", "config", "config.modality-routing.yaml"),
	repoRel("e2e", "config", "config.multi-endpoint.yaml"),
	repoRel("e2e", "config", "config.multi-provider.yaml"),
	repoRel("e2e", "config", "config.response-api-redis.yaml"),
	repoRel("e2e", "config", "config.response-api.yaml"),
	repoRel("e2e", "config", "config.testing.yaml"),
	repoRel("e2e", "config", "onnx-binding", "config.onnx-binding-test.yaml"),
	repoRel("e2e", "config", "onnx-binding", "config.onnx-classifiers-test.yaml"),
	repoRel("e2e", "profiles", "routing-strategies", "config-with-embedding.yaml"),
	repoRel("bench", "hallucination", "config.yaml"),
	repoRel("bench", "hallucination", "config-7b.yaml"),
}

var maintainedEmbeddedConfigAssets = []string{
	"deploy/kserve/configmap-router-config.yaml",
	"deploy/kserve/configmap-router-config-simulator.yaml",
	"deploy/kserve/example-multi-model-config.yaml",
}

var maintainedValuesConfigAssets = []string{
	"deploy/helm/semantic-router/values.yaml",
	"deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml",
	"deploy/kubernetes/aibrix/semantic-router-values/values.yaml",
	"deploy/kubernetes/dynamo/semantic-router-values/values.yaml",
	"deploy/kubernetes/istio/semantic-router-values/values.yaml",
	"deploy/kubernetes/llm-d/semantic-router-values/values.yaml",
	repoRel("e2e", "profiles", "ai-gateway", "values.yaml"),
	repoRel("e2e", "profiles", "aibrix", "values.yaml"),
	repoRel("e2e", "profiles", "authz-rbac", "values.yaml"),
	repoRel("e2e", "profiles", "dynamic-config", "values.yaml"),
	repoRel("e2e", "profiles", "llm-d", "values.yaml"),
	repoRel("e2e", "profiles", "ml-model-selection", "values.yaml"),
	repoRel("e2e", "profiles", "multi-endpoint", "values.yaml"),
	repoRel("e2e", "profiles", "production-stack", "values.yaml"),
	repoRel("e2e", "profiles", "rag-hybrid-search", "values.yaml"),
	repoRel("e2e", "profiles", "response-api-redis-cluster", "values.yaml"),
	repoRel("e2e", "profiles", "response-api-redis", "values.yaml"),
	repoRel("e2e", "profiles", "response-api", "values.yaml"),
	repoRel("e2e", "profiles", "routing-strategies", "values-mcp.yaml"),
	repoRel("e2e", "profiles", "routing-strategies", "values.yaml"),
	repoRel("e2e", "profiles", "streaming", "values.yaml"),
}

type templatedConfigAsset struct {
	rel          string
	replacements map[string]string
}

var maintainedTemplatedConfigAssets = []templatedConfigAsset{
	{
		rel: repoRel("bench", "cpu-vs-gpu", "config-bench.yaml"),
		replacements: map[string]string{
			"USE_CPU_PLACEHOLDER":                       "false",
			"PROMPT_COMPRESSION_PLACEHOLDER":            "false",
			"PROMPT_COMPRESSION_MAX_TOKENS_PLACEHOLDER": "512",
		},
	},
	{
		rel: repoRel("bench", "cpu-vs-gpu", "config-bench-candle.yaml"),
		replacements: map[string]string{
			"PROMPT_COMPRESSION_PLACEHOLDER":            "false",
			"PROMPT_COMPRESSION_MAX_TOKENS_PLACEHOLDER": "512",
		},
	},
}

func TestMaintainedConfigAssetsUseCanonicalV03Contract(t *testing.T) {
	for _, rel := range maintainedFullConfigAssets {
		t.Run(rel, func(t *testing.T) {
			validateMaintainedConfigAsset(t, rel, readMaintainedConfigAsset(t, rel))
		})
	}

	for _, rel := range maintainedEmbeddedConfigAssets {
		t.Run(rel, func(t *testing.T) {
			validateMaintainedConfigAsset(t, rel, readEmbeddedConfigAsset(t, rel))
		})
	}

	for _, rel := range maintainedValuesConfigAssets {
		t.Run(rel, func(t *testing.T) {
			validateMaintainedConfigAsset(t, rel, readValuesConfigAsset(t, rel))
		})
	}

	for _, asset := range maintainedTemplatedConfigAssets {
		t.Run(asset.rel, func(t *testing.T) {
			validateMaintainedConfigAsset(t, asset.rel, readTemplatedConfigAsset(t, asset))
		})
	}
}

func readMaintainedConfigAsset(t *testing.T, rel string) []byte {
	t.Helper()
	return mustReadRepoFile(t, rel)
}

func readEmbeddedConfigAsset(t *testing.T, rel string) []byte {
	t.Helper()
	root := decodeYAMLMap(t, mustReadRepoFile(t, rel), rel)
	data := mustAssetMapValue(t, root, "data", rel)
	rawConfig, ok := data["config.yaml"].(string)
	if !ok || rawConfig == "" {
		t.Fatalf("%s is missing data.config.yaml", rel)
	}
	return []byte(rawConfig)
}

func readValuesConfigAsset(t *testing.T, rel string) []byte {
	t.Helper()
	root := decodeYAMLMap(t, mustReadRepoFile(t, rel), rel)
	rawConfig, ok := root["config"]
	if !ok {
		t.Fatalf("%s is missing top-level config block", rel)
	}
	data, err := yamlv3.Marshal(rawConfig)
	if err != nil {
		t.Fatalf("failed to marshal %s config block: %v", rel, err)
	}
	return data
}

func readTemplatedConfigAsset(t *testing.T, asset templatedConfigAsset) []byte {
	t.Helper()
	content := string(mustReadRepoFile(t, asset.rel))
	replacerPairs := make([]string, 0, len(asset.replacements)*2)
	for oldValue, newValue := range asset.replacements {
		replacerPairs = append(replacerPairs, oldValue, newValue)
	}
	return []byte(strings.NewReplacer(replacerPairs...).Replace(content))
}

func validateMaintainedConfigAsset(t *testing.T, rel string, data []byte) {
	t.Helper()
	raw := decodeYAMLMap(t, data, rel)
	assertNoLegacySteadyStateKeys(t, rel, raw)
	if _, err := ParseYAMLBytes(data); err != nil {
		t.Fatalf("%s no longer parses as a maintained canonical config asset: %v", rel, err)
	}
}

func assertNoLegacySteadyStateKeys(t *testing.T, rel string, raw map[string]interface{}) {
	t.Helper()
	for _, key := range []string{
		"signals",
		"decisions",
		"keyword_rules",
		"embedding_rules",
		"categories",
		"fact_check_rules",
		"user_feedback_rules",
		"preference_rules",
		"language_rules",
		"context_rules",
		"complexity_rules",
		"modality_rules",
		"role_bindings",
		"jailbreak",
		"pii",
		"default_model",
		"reasoning_families",
		"default_reasoning_effort",
		"model_config",
		"vllm_endpoints",
		"provider_profiles",
		"strategy",
		"bert_model",
	} {
		if _, ok := raw[key]; ok {
			t.Fatalf("%s still uses legacy top-level key %q; migrate it to canonical providers/routing/global", rel, key)
		}
	}
}

func mustReadRepoFile(t *testing.T, rel string) []byte {
	t.Helper()
	path := filepath.Join("..", "..", "..", "..", rel)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %s: %v", rel, err)
	}
	return data
}

func decodeYAMLMap(t *testing.T, data []byte, rel string) map[string]interface{} {
	t.Helper()
	var root map[string]interface{}
	if err := yamlv3.Unmarshal(data, &root); err != nil {
		t.Fatalf("failed to decode %s: %v", rel, err)
	}
	return root
}

func mustAssetMapValue(t *testing.T, root map[string]interface{}, key, rel string) map[string]interface{} {
	t.Helper()
	value, ok := root[key]
	if !ok {
		t.Fatalf("%s is missing %s", rel, key)
	}
	typed, ok := value.(map[string]interface{})
	if !ok {
		t.Fatalf("%s %s is not a map", rel, key)
	}
	return typed
}
