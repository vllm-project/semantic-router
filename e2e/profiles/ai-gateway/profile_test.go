package aigateway

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestFeatureRecipesReuseBaselinePluginContracts(t *testing.T) {
	config := profileConfig(t)
	baseline := profileMap(t, config, "routing")
	routing := profileMap(t, profileNamed(t, config["recipes"], "e2e-plugins"), "routing")
	signals := profileMap(t, routing, "signals")
	if len(signals) != 1 || signals["keywords"] == nil {
		t.Fatalf("plugin feature recipe must use only its explicit keyword signals: %#v", signals)
	}
	names := []string{
		"plugin_request_mutations", "tools_passthrough", "tools_filtered", "tools_none",
		"tool_selection_add_weather", "tool_selection_add_calc",
		"tool_selection_filter", "tool_selection_filter_threshold", "tool_selection_add_topk_one",
		"tool_selection_with_system_prompt",
	}
	if len(routing["decisions"].([]any)) != len(names) || len(signals["keywords"].([]any)) != len(names) {
		t.Fatal("plugin recipe must preserve every feature route without additional policies")
	}
	for _, name := range names {
		original := profileNamed(t, baseline["decisions"], name+"_decision")
		feature := profileNamed(t, routing["decisions"], name+"_decision")
		if !reflect.DeepEqual(original, feature) {
			t.Fatalf("feature recipe changed baseline plugin behavior for %s", name)
		}
		originalSignal := profileNamed(t, profileMap(t, baseline, "signals")["keywords"], name)
		featureSignal := profileNamed(t, signals["keywords"], name)
		if !reflect.DeepEqual(originalSignal, featureSignal) {
			t.Fatalf("feature recipe changed keyword matching for %s", name)
		}
	}
}

func TestProfilePinsAcceptanceClassifierModels(t *testing.T) {
	catalog := profileMap(t, profileConfig(t), "global", "model_catalog")
	system := profileMap(t, catalog, "system")
	requireModel(t, system, "prompt_guard", "models/mmbert32k-jailbreak-detector-merged")
	requireModel(t, system, "domain_classifier", "models/mmbert32k-intent-classifier-merged")
	requireModel(t, system, "pii_classifier", "models/mmbert32k-pii-detector-merged")

	modules := profileMap(t, catalog, "modules")
	requireModuleModel(
		t,
		profileMap(t, modules, "prompt_guard"),
		"models/mmbert32k-jailbreak-detector-merged",
		"models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
	)
	classifier := profileMap(t, modules, "classifier")
	requireModuleModel(
		t,
		profileMap(t, classifier, "domain"),
		"models/mmbert32k-intent-classifier-merged",
		"models/mmbert32k-intent-classifier-merged/category_mapping.json",
	)
	requireModuleModel(
		t,
		profileMap(t, classifier, "pii"),
		"models/mmbert32k-pii-detector-merged",
		"models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
	)
}

func requireModel(t *testing.T, system map[string]any, name, want string) {
	t.Helper()
	if got := system[name]; got != want {
		t.Fatalf("system model %s = %v, want %s", name, got, want)
	}
}

func requireModuleModel(t *testing.T, module map[string]any, modelID, mappingPath string) {
	t.Helper()
	if module["enabled"] != true || module["model_id"] != modelID {
		t.Fatalf("classifier module is not pinned to %s: %#v", modelID, module)
	}
	if module["max_sequence_length"] != 0 {
		t.Fatalf("classifier module %s must retain the legacy 512-token budget: %#v", modelID, module)
	}
	for _, field := range []string{"jailbreak_mapping_path", "category_mapping_path", "pii_mapping_path"} {
		if value, ok := module[field]; ok && value == mappingPath {
			return
		}
	}
	t.Fatalf("classifier module %s does not use mapping %s: %#v", modelID, mappingPath, module)
}

func TestProtocolAndCacheRecipesReachTheirOwnedBoundaries(t *testing.T) {
	config := profileConfig(t)
	baselineCache := profileNamed(t, profileMap(t, config, "routing")["decisions"], "other_decision")["plugins"]
	for _, test := range []struct {
		recipe   string
		decision string
		plugins  any
	}{
		{"e2e-protocol", "e2e_protocol_upstream", nil},
		{"e2e-cache", "e2e_cache_decision", baselineCache},
	} {
		t.Run(test.recipe, func(t *testing.T) {
			routing := profileMap(t, profileNamed(t, config["recipes"], test.recipe), "routing")
			if routing["signals"] != nil || len(routing["decisions"].([]any)) != 1 {
				t.Fatal("boundary fixture must not depend on learned signals or competing decisions")
			}
			decision := profileNamed(t, routing["decisions"], test.decision)
			if !reflect.DeepEqual(decision["rules"], map[string]any{"operator": "AND"}) {
				t.Fatal("boundary fixture must select its real upstream/cache policy unconditionally")
			}
			if !reflect.DeepEqual(decision["plugins"], test.plugins) {
				t.Fatal("protocol path must not cache; cache path must retain its original plugin settings")
			}
			models := decision["modelRefs"].([]any)
			if len(models) != 1 || models[0].(map[string]any)["model"] != "base-model" || models[0].(map[string]any)["lora_name"] != "general-expert" {
				t.Fatal("boundary fixture must dispatch to the existing provider fixture")
			}
		})
	}
}

func TestFeatureEntrypointsPreserveDefaultSecurityPrecedence(t *testing.T) {
	config := profileConfig(t)
	for _, recipe := range []string{"e2e-protocol", "e2e-plugins", "e2e-cache", "e2e-domain", "e2e-fallback"} {
		found := false
		for _, raw := range config["entrypoints"].([]any) {
			entrypoint := raw.(map[string]any)
			if entrypoint["recipe"] == recipe && reflect.DeepEqual(entrypoint["model_names"], []any{recipe}) {
				found = true
			}
		}
		if !found {
			t.Fatalf("feature model must resolve explicitly to its owned recipe: %s", recipe)
		}
	}
	routing := profileMap(t, config, "routing")
	for _, test := range []struct {
		name, signalType, signalName string
		priority                     int
		threshold                    float64
	}{
		{"block_jailbreak", "jailbreak", "jailbreak_standard", 1000, 0.5},
		{"block_pii", "pii", "pii_deny_all", 999, 0.7},
	} {
		decision := profileNamed(t, routing["decisions"], test.name)
		signal := profileNamed(t, profileMap(t, routing, "signals")[test.signalType], test.signalName)
		if decision["priority"] != test.priority || signal["threshold"] != test.threshold {
			t.Fatalf("default security policy changed: %s", test.name)
		}
		plugins := decision["plugins"].([]any)
		if len(plugins) != 1 || plugins[0].(map[string]any)["type"] != "fast_response" {
			t.Fatalf("default security must still prevent provider dispatch: %s", test.name)
		}
	}
}

func TestDomainRecipePreservesEveryBaselineLabelAndSelection(t *testing.T) {
	config := profileConfig(t)
	baseline := profileMap(t, config, "routing")
	routing := profileMap(t, profileNamed(t, config["recipes"], "e2e-domain"), "routing")
	signals := profileMap(t, routing, "signals")
	if len(signals) != 1 || !reflect.DeepEqual(signals["domains"], profileMap(t, baseline, "signals")["domains"]) {
		t.Fatal("domain recipe must retain the original domain labels without competing signal families")
	}
	decisions := routing["decisions"].([]any)
	if len(decisions) != 14 {
		t.Fatal("domain recipe must cover every published domain")
	}
	for _, raw := range decisions {
		decision := raw.(map[string]any)
		original := profileNamed(t, baseline["decisions"], decision["name"].(string))
		want := make(map[string]any, len(original))
		for key, value := range original {
			want[key] = value
		}
		want["plugins"] = []any{}
		if !reflect.DeepEqual(decision, want) {
			t.Fatalf("domain recipe changed classification or model selection: %s", decision["name"])
		}
	}
}

func TestFallbackRecipeRetainsARealUnmatchedPath(t *testing.T) {
	config := profileConfig(t)
	routing := profileMap(t, profileNamed(t, config["recipes"], "e2e-fallback"), "routing")
	if len(routing["decisions"].([]any)) != 1 {
		t.Fatal("fallback fixture must not have a catch-all decision")
	}
	signals := profileMap(t, routing, "signals")
	keyword := profileNamed(t, signals["keywords"], "fallback_contract_keyword")
	if len(signals) != 1 || !reflect.DeepEqual(keyword["keywords"], []any{"__E2E_FALLBACK_MATCH__"}) {
		t.Fatal("fallback match must depend only on the explicit positive fixture marker")
	}
	decision := profileNamed(t, routing["decisions"], "fallback_contract_match")
	models := decision["modelRefs"].([]any)
	if decision["plugins"] != nil || len(models) != 1 || models[0].(map[string]any)["model"] != "base-model" {
		t.Fatal("matched fallback fixture must use its explicit provider without plugins")
	}
	defaults := profileMap(t, profileMap(t, config, "providers"), "defaults")
	if defaults["model"] != "general-expert" {
		t.Fatal("unmatched fallback must retain the declared default provider")
	}
}

func TestGuardProfileMatchesCanonicalPublishedOperatingPoint(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var canonical map[string]any
	if err := yaml.Unmarshal(raw, &canonical); err != nil {
		t.Fatal(err)
	}
	modules := profileMap(t, profileMap(t, profileMap(t, canonical, "global"), "model_catalog"), "modules")
	guard := profileMap(t, modules, "prompt_guard")
	profile := profileMap(t, profileMap(t, profileConfig(t), "routing"), "signals")
	signal := profileNamed(t, profile["jailbreak"], "jailbreak_standard")
	if signal["threshold"] != guard["threshold"] {
		t.Fatalf("Guard operating point differs from canonical published configuration: profile=%v canonical=%v", signal["threshold"], guard["threshold"])
	}
}

func profileConfig(t *testing.T) map[string]any {
	t.Helper()
	raw, err := os.ReadFile(filepath.Base(valuesFile))
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	return profileMap(t, values, "config")
}

func profileMap(t *testing.T, value map[string]any, keys ...string) map[string]any {
	t.Helper()
	for _, key := range keys {
		result, ok := value[key].(map[string]any)
		if !ok {
			t.Fatalf("missing profile mapping %q", key)
		}
		value = result
	}
	return value
}

func profileNamed(t *testing.T, value any, name string) map[string]any {
	t.Helper()
	for _, raw := range value.([]any) {
		item := raw.(map[string]any)
		if item["name"] == name {
			return item
		}
	}
	t.Fatalf("missing profile item %q", name)
	return nil
}
