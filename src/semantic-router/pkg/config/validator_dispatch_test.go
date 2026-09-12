package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Exercise the public loaders, including normalization and defaults, rather
// than only invoking a family validator or the post-CRD entry point directly.
func TestConfigLoadValidatesGlobalsForEverySource(t *testing.T) {
	cases := []struct {
		name   string
		global string
		want   string
	}{
		{"cache_similarity", `stores: {response_cache: {enabled: true, similarity_threshold: 1.5}}`, "similarity_threshold"},
		{"cache_guard", `stores: {response_cache: {enabled: true, polarity_guard: {mode: nli_typo}}}`, "polarity_guard mode"},
		{"memory_similarity", `stores: {memory: {default_similarity_threshold: 1.5}}`, "default_similarity_threshold"},
		{"learning_backend", `router: {learning: {state_store: {backend: redis_typo}}}`, "state_store.backend"},
		{"learning_ttl", `router: {learning: {state_store: {ttl_seconds: -1}}}`, "state_store.ttl_seconds"},
		{"learning_strategy", `router: {learning: {adaptation: {strategy: unknown}}}`, "adaptation.strategy"},
		{"retired_selector", `router: {model_selection: {method: elo}}`, "model_selection.method=elo"},
		{"embedding_dimensions", remoteEmbeddingGlobal("dimensions: 64"), "must match embedding_config.target_dimension"},
		{"embedding_retries", remoteEmbeddingGlobal("max_retries: -1"), "endpoint.max_retries"},
		{"embedding_timeout", remoteEmbeddingGlobal("timeout_seconds: -1"), "endpoint.timeout_seconds"},
		{"admission_overflow", `model_catalog: {admission: {prompt_guard: {max_concurrency: 1, on_overflow: drop}}}`, "on_overflow"},
		{"admission_queue", `model_catalog: {admission: {prompt_guard: {max_concurrency: 1, max_queue: -1}}}`, "max_queue"},
		{"tool_filter", `integrations: {tools: {advanced_filtering: {enabled: true, min_combined_score: 1.5}}}`, "min_combined_score"},
		{"prompt_profile", `model_catalog: {modules: {prompt_compression: {profile: unknown}}}`, "prompt_compression.profile"},
		{"modality_method", `model_catalog: {modules: {modality_detector: {enabled: true, method: unknown}}}`, "modality_detection.method"},
		{"hallucination_backend", `model_catalog: {modules: {hallucination_mitigation: {detector: {backend: unknown}}}}`, "hallucination detector backend"},
		{"remom_models", `integrations: {looper: {remom: {model_names: [""]}}}`, "remom: model_names"},
		{"fusion_models", `integrations: {looper: {fusion: {model_names: [""]}}}`, "fusion: model_names"},
		{"flow_backend", `integrations: {looper: {flow: {state: {store_backend: unknown}}}}`, "state.store_backend"},
		{"category_backend", `model_catalog: {modules: {classifier: {domain: {backend: {protocol: unknown, model: classifier}}}}}`, "classifier.domain.backend.protocol"},
		{"complexity_backend", `model_catalog: {modules: {complexity: {backend: {protocol: unknown, model: classifier}}}}`, "complexity.backend.protocol"},
		{"pii_error_policy", `model_catalog: {modules: {classifier: {pii: {on_error: unknown}}}}`, "classifier.pii.on_error"},
		{"prompt_guard_variant", `model_catalog: {modules: {prompt_guard: {variant: unknown}}}`, "prompt_guard.variant"},
		{"prompt_guard_wiring", `model_catalog: {external: [], modules: {prompt_guard: {enabled: true, variant: "", protocol: http_chat}}}`, "requires an entry in external_models"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var fileError string
			for _, source := range []ConfigSource{ConfigSourceFile, ConfigSourceKubernetes} {
				t.Run(string(source), func(t *testing.T) {
					raw := globalValidationDocument(t, source, tc.global)
					cfg, err := ParseYAMLBytes(raw)
					if err == nil || !strings.Contains(err.Error(), tc.want) {
						t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, tc.want)
					}
					if cfg != nil {
						t.Fatal("invalid config must not be returned to runtime constructors")
					}
					if source == ConfigSourceFile {
						fileError = err.Error()
					} else if err.Error() != fileError {
						t.Errorf("Kubernetes error = %q, file error = %q", err, fileError)
					}
					path := filepath.Join(t.TempDir(), "config.yaml")
					if err := os.WriteFile(path, raw, 0o600); err != nil {
						t.Fatal(err)
					}
					if cfg, err := Parse(path); cfg != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
						t.Fatalf("Parse() config = %v, error = %v, want rejection containing %q", cfg, err, tc.want)
					}
				})
			}
		})
	}
}

func TestConfigLoadAcceptsValidGlobalsForEverySource(t *testing.T) {
	for _, source := range []ConfigSource{ConfigSourceFile, ConfigSourceKubernetes} {
		t.Run(string(source), func(t *testing.T) {
			raw := globalValidationDocument(t, source, `
router:
  learning:
    state_store: {backend: local, ttl_seconds: 0}
stores:
  response_cache: {enabled: true, similarity_threshold: 1, polarity_guard: {mode: lexical}}
  memory: {default_similarity_threshold: 0}
model_catalog:
  admission:
    prompt_guard: {max_concurrency: 1, max_queue: 1, on_overflow: wait}
`)
			cfg, err := ParseYAMLBytes(raw)
			if err != nil {
				t.Fatalf("valid static configuration rejected: %v", err)
			}
			if err := ValidateKubernetesConfigContracts(cfg); err != nil {
				t.Fatalf("valid complete configuration rejected: %v", err)
			}
		})
	}
}

func globalValidationDocument(t *testing.T, source ConfigSource, fragment string) []byte {
	t.Helper()
	var global map[string]any
	if err := yaml.Unmarshal([]byte(fragment), &global); err != nil {
		t.Fatal(err)
	}
	router, ok := global["router"].(map[string]any)
	if !ok {
		router = map[string]any{}
		global["router"] = router
	}
	router["config_source"] = source
	raw, err := json.Marshal(map[string]any{
		"version": "v0.3",
		"global":  global,
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func remoteEmbeddingGlobal(endpointField string) string {
	return `model_catalog:
  embeddings:
    semantic:
      embedding_config: {backend: openai_compatible, model_type: remote, target_dimension: 128}
      endpoint:
        base_url: http://127.0.0.1:8000/v1
        model: test-embedding
        ` + endpointField + "\n"
}

func TestKubernetesValidationDefersRoutingUntilCRDsAreMerged(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*RouterConfig)
		want   string
	}{
		{
			name: "decision_model",
			mutate: func(cfg *RouterConfig) {
				cfg.Decisions = []Decision{{Name: "invalid", ModelRefs: []ModelRef{{Model: ""}}}}
			},
			want: "model name cannot be empty",
		},
		{
			name: "complexity_recipe_boundaries",
			mutate: func(cfg *RouterConfig) {
				cfg.Recipes = []RoutingRecipe{{Name: "internal", Profile: RoutingProfile{Signals: Signals{
					ComplexityRules: []ComplexityRule{{Name: "difficulty", HardBelow: floatPtr(0.4), EasyAbove: floatPtr(0.8)}},
				}}}}
			},
			want: "declares hard_below/easy_above",
		},
		{
			name: "cross_recipe_classifier",
			mutate: func(cfg *RouterConfig) {
				for _, name := range []RecipeName{"private", "public"} {
					cfg.Recipes = append(cfg.Recipes, RoutingRecipe{Name: name, Profile: RoutingProfile{Signals: Signals{
						ClassifierRules: []ClassifierSignalRule{{Name: "risk", Type: "local", ModelPath: "models/" + string(name), Labels: []string{"SAFE", "RISKY"}}},
					}}})
				}
			},
			want: "incompatible local classifiers",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{ConfigSource: ConfigSourceKubernetes}
			tc.mutate(cfg)
			if err := validateConfigStructure(cfg); err != nil {
				t.Fatalf("initial Kubernetes validation must defer routing contracts: %v", err)
			}
			if err := ValidateKubernetesConfigContracts(cfg); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("post-CRD validation error = %v, want %q", err, tc.want)
			}
			cfg.ConfigSource = ConfigSourceFile
			if err := validateConfigStructure(cfg); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("file validation error = %v, want %q", err, tc.want)
			}
		})
	}
}
