package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const nestedPluginSettingsYAML = `routing:
  decisions:
    - name: nested
      plugins:
        - type: memory
          configuration:
            enabled: true
            reflection:
              enabled: true
              algorithm: heuristic
              max_inject_tokens: 384
              recency_decay_days: 14
              dedup_threshold: 0.9
              block_patterns: [api key, password]
        - type: tool_selection
          configuration:
            enabled: true
            advanced_filtering:
              enabled: true
              retrieval_strategy: hybrid_history
              candidate_pool_size: 30
              min_lexical_overlap: 2
              min_combined_score: 0.4
              weights: {embed: 0.5, lexical: 0.2, tag: 0.1, name: 0.1, category: 0.1}
              use_category_filter: true
              category_confidence_threshold: 0.7
              allow_tools: [docs.search]
              block_tools: [admin.delete]
              hybrid_history:
                history_horizon: 4
                min_history_steps: 2
                history_confidence_threshold: 0.3
                weight_semantic: 0.5
                weight_history_transition: 0.3
                weight_decision_prior: 0.2
                repetition_penalty_strength: 0.1
`

func TestDecisionPluginsReadNestedSettings(t *testing.T) {
	cfg, err := ParseRoutingYAMLBytes([]byte(nestedPluginSettingsYAML))
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Routing struct {
			Decisions []struct {
				Plugins []struct {
					Configuration map[string]interface{} `yaml:"configuration"`
				} `yaml:"plugins"`
			} `yaml:"decisions"`
		} `yaml:"routing"`
	}
	if err := yaml.Unmarshal([]byte(nestedPluginSettingsYAML), &document); err != nil {
		t.Fatal(err)
	}
	plugins := document.Routing.Decisions[0].Plugins
	decision := cfg.Decisions[0]
	// Written back with its YAML keys, each decoded block must equal the input.
	for _, tc := range []struct {
		name    string
		written interface{}
		decoded interface{}
	}{
		{"memory.reflection", plugins[0].Configuration["reflection"], decision.GetMemoryConfig().Reflection},
		{"tool_selection.advanced_filtering", plugins[1].Configuration["advanced_filtering"], decision.GetToolSelectionConfig().AdvancedFiltering},
	} {
		encoded, err := yaml.Marshal(tc.decoded)
		if err != nil {
			t.Fatal(err)
		}
		var decoded interface{}
		if err := yaml.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(decoded, tc.written) {
			t.Errorf("%s is decoded as:\n%s", tc.name, encoded)
		}
	}

	invalid := strings.Replace(nestedPluginSettingsYAML, "candidate_pool_size: 30", "candidate_pool_size: -1", 1)
	if _, err := ParseRoutingYAMLBytes([]byte(invalid)); err == nil || !strings.Contains(err.Error(), "candidate_pool_size") {
		t.Errorf("negative per-decision candidate_pool_size: got error %v", err)
	}
}

// Decision plugin configurations are decoded from JSON, so every field,
// nested ones included, must accept its YAML key as a JSON key.
func TestDecisionPluginFieldsAcceptYAMLKeysAsJSON(t *testing.T) {
	seen := map[reflect.Type]bool{}
	var check func(path string, typ reflect.Type)
	check = func(path string, typ reflect.Type) {
		for typ.Kind() == reflect.Pointer || typ.Kind() == reflect.Slice || typ.Kind() == reflect.Map {
			typ = typ.Elem()
		}
		if typ.Kind() != reflect.Struct || seen[typ] {
			return
		}
		seen[typ] = true
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			yamlKey := strings.Split(field.Tag.Get("yaml"), ",")[0]
			if yamlKey == "" || yamlKey == "-" {
				continue
			}
			jsonKey := strings.Split(field.Tag.Get("json"), ",")[0]
			if jsonKey == "" {
				jsonKey = field.Name
			}
			// encoding/json matches object keys case-insensitively.
			if !strings.EqualFold(jsonKey, yamlKey) {
				t.Errorf("%s.%s: field %s.%s has JSON key %q", path, yamlKey, typ.Name(), field.Name, jsonKey)
			}
			check(path+"."+yamlKey, field.Type)
		}
	}
	for pluginType, payload := range DecisionPluginSchemaSamples() {
		check(pluginType, reflect.TypeOf(payload))
	}
}
