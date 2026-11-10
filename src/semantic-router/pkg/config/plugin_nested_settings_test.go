package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const nestedPluginSettingsYAML = `recipes:
  - name: default
    decisions:
      - name: nested
        plugins:
          - type: context
            configuration:
              enabled: true
              sources:
                - type: memory
                  match:
                    recall: {method: embedding, top_k: 4, min_similarity: 0.6}
                  reflection:
                    enabled: true
                    algorithm: heuristic
                    max_inject_tokens: 384
                    recency_decay_days: 14
                    dedup_threshold: 0.9
                    block_patterns: [api key, password]
              compress:
                targets: {tool_outputs: {mode: extractive, min_tokens: 1500, target_tokens: 600}}
                match:
                  recall: {method: bm25}
                  judge: {preset: relevance, min_ratio: 0.5}
              budget: {max_tokens: 6000, reserve_output_tokens: auto}
          - type: tools
            configuration:
              enabled: true
              select:
                mode: filter
                match:
                  judge:
                    question: {type: choice, instructions: "Which listed tool does the request need?"}
                    top_k: 4
                    min_score: 0.2
                    min_ratio: 0.3
                need_threshold: 0.6
                preserve_count: 1
                input: {history: true}
                sticky: {enabled: false, max_tools: 8, max_new_tools_per_turn: 1, pin_called_tools: false}
`

func TestDecisionPluginsReadNestedSettings(t *testing.T) {
	cfg, err := ParseRoutingYAMLBytes([]byte(nestedPluginSettingsYAML))
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Recipes []struct {
			Decisions []struct {
				Plugins []struct {
					Configuration map[string]interface{} `yaml:"configuration"`
				} `yaml:"plugins"`
			} `yaml:"decisions"`
		} `yaml:"recipes"`
	}
	if err := yaml.Unmarshal([]byte(nestedPluginSettingsYAML), &document); err != nil {
		t.Fatal(err)
	}
	plugins := document.Recipes[0].Decisions[0].Plugins
	decision := cfg.Decisions[0]
	// Written back with its YAML keys, each decoded block must equal the input.
	for _, tc := range []struct {
		name    string
		written interface{}
		decoded interface{}
	}{
		{"context.sources", plugins[0].Configuration["sources"], decision.GetContextConfig().Sources},
		{"context.compress", plugins[0].Configuration["compress"], decision.GetContextConfig().Compress},
		{"context.budget", plugins[0].Configuration["budget"], decision.GetContextConfig().Budget},
		{"tools.select", plugins[1].Configuration["select"], decision.GetToolsConfig().Select},
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

	invalid := strings.Replace(nestedPluginSettingsYAML, "preserve_count: 1", "preserve_count: -1", 1)
	if _, err := ParseRoutingYAMLBytes([]byte(invalid)); err == nil || !strings.Contains(err.Error(), "preserve_count") {
		t.Errorf("negative per-decision preserve_count: got error %v", err)
	}
}

// Decision plugin configurations, and the RAG backend configurations their
// context sources hold, are decoded from JSON and, from the DSL, encoded to it,
// so every field, nested ones included, must use its YAML key as its JSON key.
func TestDecisionPluginFieldsUseYAMLKeysAsJSONKeys(t *testing.T) {
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
			yamlKey, yamlOptions, _ := strings.Cut(field.Tag.Get("yaml"), ",")
			if yamlKey == "" || yamlKey == "-" {
				continue
			}
			jsonKey, jsonOptions, _ := strings.Cut(field.Tag.Get("json"), ",")
			if jsonKey == "" {
				jsonKey = field.Name
			}
			// An unset field the YAML omits must not be written as null.
			if jsonKey != yamlKey || strings.Contains(yamlOptions, "omitempty") != strings.Contains(jsonOptions, "omitempty") {
				t.Errorf("%s.%s: field %s.%s has JSON tag %q", path, yamlKey, typ.Name(), field.Name, field.Tag.Get("json"))
			}
			check(path+"."+yamlKey, field.Type)
		}
	}
	for pluginType, payload := range DecisionPluginSchemaSamples() {
		check(pluginType, reflect.TypeOf(payload))
	}
	for backend, payload := range map[string]interface{}{
		"milvus": MilvusRAGConfig{}, "qdrant": QdrantRAGConfig{}, "external_api": ExternalAPIRAGConfig{}, "mcp": MCPRAGConfig{},
		"openai": OpenAIRAGConfig{}, "vectorstore": VectorStoreRAGConfig{}, "hybrid": HybridRAGConfig{},
	} {
		check("context.sources.backend_config("+backend+")", reflect.TypeOf(payload))
	}
}
