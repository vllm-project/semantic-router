package authoring

import (
	"os"
	"path/filepath"
	"reflect"
	goruntime "runtime"
	"sort"
	"testing"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

func TestLoadRuntimeConfigMatchesSharedFirstSliceRuntimeFixture(t *testing.T) {
	t.Parallel()

	runtimeCfg, err := LoadRuntimeConfig(sharedFixturePath(t, "td001-first-slice-authoring.yaml"))
	if err != nil {
		t.Fatalf("LoadRuntimeConfig() error = %v", err)
	}

	assertYAMLEqual(
		t,
		loadYAMLFile(t, sharedFixturePath(t, "td001-first-slice-runtime.yaml")),
		extractCompiledFirstSliceRuntime(runtimeCfg),
	)
}

func TestCompileRuntimeEmitsSharedFirstSliceUserYAML(t *testing.T) {
	t.Parallel()

	cfg, err := ParseFile(sharedFixturePath(t, "td001-first-slice-authoring.yaml"))
	if err != nil {
		t.Fatalf("ParseFile() error = %v", err)
	}

	runtimeCfg, err := CompileRuntime(cfg)
	if err != nil {
		t.Fatalf("CompileRuntime() error = %v", err)
	}

	emitted, err := dsl.EmitUserYAML(runtimeCfg)
	if err != nil {
		t.Fatalf("EmitUserYAML() error = %v", err)
	}

	var raw map[string]interface{}
	if err := yaml.Unmarshal(emitted, &raw); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}

	assertYAMLEqual(
		t,
		selectTopLevelKeys(loadYAMLFile(t, sharedFixturePath(t, "td001-first-slice-authoring.yaml")), "listeners", "signals", "providers", "decisions"),
		selectTopLevelKeys(raw, "listeners", "signals", "providers", "decisions"),
	)
}

func TestParseRejectsUnsupportedVersion(t *testing.T) {
	t.Parallel()

	_, err := Parse([]byte("version: v9.9\nproviders: {}\n"))
	if err == nil {
		t.Fatal("Parse() error = nil, want unsupported version error")
	}
}

func TestParseRejectsUnknownField(t *testing.T) {
	t.Parallel()

	_, err := Parse([]byte("version: v0.1\nproviders: {}\nextra_field: true\n"))
	if err == nil {
		t.Fatal("Parse() error = nil, want unknown field error")
	}
}

func TestCompileRuntimeRejectsUnknownDefaultModel(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		Version: CurrentVersion,
		Providers: Providers{
			Models: []Model{
				{
					Name: "known",
					Endpoints: []Endpoint{{
						Name:     "primary",
						Endpoint: "localhost:8000",
					}},
				},
			},
			DefaultModel: "missing",
		},
	}

	_, err := CompileRuntime(cfg)
	if err == nil {
		t.Fatal("CompileRuntime() error = nil, want unknown default model error")
	}
}

func sharedFixturePath(t *testing.T, name string) string {
	t.Helper()

	_, file, _, ok := goruntime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller() failed")
	}

	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", "..", "..", "..", "config", "testing", name))
}

func loadYAMLFile(t *testing.T, path string) map[string]interface{} {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile(%q) error = %v", path, err)
	}

	var raw map[string]interface{}
	if err := yaml.Unmarshal(data, &raw); err != nil {
		t.Fatalf("yaml.Unmarshal(%q) error = %v", path, err)
	}
	return raw
}

func selectTopLevelKeys(raw map[string]interface{}, keys ...string) map[string]interface{} {
	selected := make(map[string]interface{}, len(keys))
	for _, key := range keys {
		if value, exists := raw[key]; exists {
			selected[key] = value
		}
	}
	return selected
}

func assertYAMLEqual(t *testing.T, want, got interface{}) {
	t.Helper()

	normalizedWant := normalizeYAMLValue(t, want)
	normalizedGot := normalizeYAMLValue(t, got)
	if reflect.DeepEqual(normalizedWant, normalizedGot) {
		return
	}

	wantYAML, err := yaml.Marshal(normalizedWant)
	if err != nil {
		t.Fatalf("yaml.Marshal(want) error = %v", err)
	}
	gotYAML, err := yaml.Marshal(normalizedGot)
	if err != nil {
		t.Fatalf("yaml.Marshal(got) error = %v", err)
	}

	t.Fatalf("YAML mismatch\nwant:\n%s\ngot:\n%s", wantYAML, gotYAML)
}

func normalizeYAMLValue(t *testing.T, value interface{}) interface{} {
	t.Helper()

	data, err := yaml.Marshal(value)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}

	var normalized interface{}
	if err := yaml.Unmarshal(data, &normalized); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}
	return normalized
}

func extractCompiledFirstSliceRuntime(runtimeCfg *routerconfig.RouterConfig) map[string]interface{} {
	modelNames := sortedModelNames(runtimeCfg.ModelConfig)

	return map[string]interface{}{
		"listeners":                extractListeners(runtimeCfg.Listeners),
		"keyword_rules":            extractKeywordRules(runtimeCfg.KeywordRules),
		"vllm_endpoints":           extractEndpoints(runtimeCfg.VLLMEndpoints),
		"model_config":             extractModelConfig(runtimeCfg.ModelConfig, modelNames),
		"default_model":            runtimeCfg.DefaultModel,
		"reasoning_families":       extractReasoningFamilies(runtimeCfg.ReasoningFamilies),
		"default_reasoning_effort": runtimeCfg.DefaultReasoningEffort,
		"decisions":                extractDecisions(runtimeCfg.Decisions),
	}
}

func sortedModelNames(modelConfig map[string]routerconfig.ModelParams) []string {
	names := make([]string, 0, len(modelConfig))
	for name := range modelConfig {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func extractListeners(listeners []routerconfig.Listener) []map[string]interface{} {
	extracted := make([]map[string]interface{}, 0, len(listeners))
	for _, listener := range listeners {
		entry := map[string]interface{}{
			"name":    listener.Name,
			"address": listener.Address,
			"port":    listener.Port,
		}
		if listener.Timeout != "" {
			entry["timeout"] = listener.Timeout
		}
		extracted = append(extracted, entry)
	}
	return extracted
}

func extractKeywordRules(rules []routerconfig.KeywordRule) []map[string]interface{} {
	extracted := make([]map[string]interface{}, 0, len(rules))
	for _, rule := range rules {
		extracted = append(extracted, map[string]interface{}{
			"name":           rule.Name,
			"operator":       rule.Operator,
			"keywords":       append([]string(nil), rule.Keywords...),
			"case_sensitive": rule.CaseSensitive,
		})
	}
	return extracted
}

func extractEndpoints(endpoints []routerconfig.VLLMEndpoint) []map[string]interface{} {
	extracted := make([]map[string]interface{}, 0, len(endpoints))
	for _, endpoint := range endpoints {
		extracted = append(extracted, map[string]interface{}{
			"name":     endpoint.Name,
			"address":  endpoint.Address,
			"port":     endpoint.Port,
			"weight":   endpoint.Weight,
			"protocol": endpoint.Protocol,
			"model":    endpoint.Model,
		})
	}
	return extracted
}

func extractModelConfig(modelConfig map[string]routerconfig.ModelParams, modelNames []string) map[string]interface{} {
	extracted := make(map[string]interface{}, len(modelNames))
	for _, modelName := range modelNames {
		params := modelConfig[modelName]
		entry := map[string]interface{}{}
		if len(params.PreferredEndpoints) > 0 {
			entry["preferred_endpoints"] = append([]string(nil), params.PreferredEndpoints...)
		}
		if params.ReasoningFamily != "" {
			entry["reasoning_family"] = params.ReasoningFamily
		}
		if params.AccessKey != "" {
			entry["access_key"] = params.AccessKey
		}
		if params.ParamSize != "" {
			entry["param_size"] = params.ParamSize
		}
		if params.APIFormat != "" {
			entry["api_format"] = params.APIFormat
		}
		if params.Description != "" {
			entry["description"] = params.Description
		}
		if len(params.Capabilities) > 0 {
			entry["capabilities"] = append([]string(nil), params.Capabilities...)
		}
		if params.QualityScore != 0 {
			entry["quality_score"] = params.QualityScore
		}
		if pricing := extractPricing(params.Pricing); pricing != nil {
			entry["pricing"] = pricing
		}
		extracted[modelName] = entry
	}
	return extracted
}

func extractPricing(pricing routerconfig.ModelPricing) map[string]interface{} {
	if pricing.Currency == "" && pricing.PromptPer1M == 0 && pricing.CompletionPer1M == 0 {
		return nil
	}
	return map[string]interface{}{
		"currency":          pricing.Currency,
		"prompt_per_1m":     pricing.PromptPer1M,
		"completion_per_1m": pricing.CompletionPer1M,
	}
}

func extractReasoningFamilies(families map[string]routerconfig.ReasoningFamilyConfig) map[string]interface{} {
	extracted := make(map[string]interface{}, len(families))
	for name, family := range families {
		extracted[name] = map[string]interface{}{
			"type":      family.Type,
			"parameter": family.Parameter,
		}
	}
	return extracted
}

func extractDecisions(decisions []routerconfig.Decision) []map[string]interface{} {
	extracted := make([]map[string]interface{}, 0, len(decisions))
	for _, decision := range decisions {
		entry := map[string]interface{}{
			"name":      decision.Name,
			"rules":     extractRuleNode(decision.Rules),
			"modelRefs": extractModelRefs(decision.ModelRefs),
		}
		if decision.Description != "" {
			entry["description"] = decision.Description
		}
		if decision.Priority != 0 {
			entry["priority"] = decision.Priority
		}
		extracted = append(extracted, entry)
	}
	return extracted
}

func extractRuleNode(node routerconfig.RuleNode) map[string]interface{} {
	entry := map[string]interface{}{}
	if node.Type != "" {
		entry["type"] = node.Type
	}
	if node.Name != "" {
		entry["name"] = node.Name
	}
	if node.Operator != "" {
		entry["operator"] = node.Operator
	}
	if len(node.Conditions) > 0 {
		conditions := make([]map[string]interface{}, 0, len(node.Conditions))
		for _, condition := range node.Conditions {
			conditions = append(conditions, extractRuleNode(condition))
		}
		entry["conditions"] = conditions
	}
	return entry
}

func extractModelRefs(modelRefs []routerconfig.ModelRef) []map[string]interface{} {
	extracted := make([]map[string]interface{}, 0, len(modelRefs))
	for _, modelRef := range modelRefs {
		entry := map[string]interface{}{
			"model": modelRef.Model,
		}
		if modelRef.UseReasoning != nil {
			entry["use_reasoning"] = *modelRef.UseReasoning
		}
		if modelRef.ReasoningEffort != "" {
			entry["reasoning_effort"] = modelRef.ReasoningEffort
		}
		if modelRef.LoRAName != "" {
			entry["lora_name"] = modelRef.LoRAName
		}
		extracted = append(extracted, entry)
	}
	return extracted
}
