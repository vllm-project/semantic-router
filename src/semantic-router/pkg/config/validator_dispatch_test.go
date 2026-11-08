package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestConfigContractValidatorsRejectNil(t *testing.T) {
	for _, tc := range []struct {
		name     string
		validate configContractValidator
		want     string
	}{
		{"kubernetes", ValidateKubernetesConfigContracts, "router configuration is nil"},
		{"structure", validateConfigStructure, "router configuration is nil"},
		{"static", func(cfg *RouterConfig) error {
			return validateConfigContractsAtStage(cfg, staticConfigValidation)
		}, "router configuration is nil"},
		{"deployments", validateModelDeploymentContracts, "model bindings require router configuration"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.validate(nil); err == nil || err.Error() != tc.want {
				t.Fatalf("nil config error = %v, want %q", err, tc.want)
			}
		})
	}
}

// Exercise the public loaders, including normalization and defaults, rather
// than only invoking a family validator or the post-CRD entry point directly.
func TestConfigLoadValidatesGlobalsForEverySource(t *testing.T) {
	cases := []struct {
		name   string
		global string
		want   string
	}{
		{"preview_timeout", `services: {api: {routing_preview: {request_timeout_seconds: 0}}}`, "routing_preview.request_timeout_seconds"},
		{"preview_concurrency", `services: {api: {routing_preview: {max_concurrency: 0}}}`, "routing_preview.max_concurrency"},
		{"cache_similarity", `plugins: [{type: response_cache, configuration: {enabled: true, semantic: {match: {recall: {top_k: 1, min_similarity: 1.5}}}}}]`, "min_similarity must be a cosine similarity"},
		{"memory_similarity", `stores: {memory: {default_similarity_threshold: 1.5}}`, "default_similarity_threshold"},
		{"learning_backend", `router: {learning: {state_store: {backend: redis_typo}}}`, "state_store.backend"},
		{"learning_ttl", `router: {learning: {state_store: {ttl_seconds: -1}}}`, "state_store.ttl_seconds"},
		{"learning_strategy", `router: {learning: {adaptation: {strategy: unknown}}}`, "adaptation.strategy"},
		{"retired_selector", `router: {model_selection: {method: elo}}`, "global.router.model_selection.method is removed"},
		{"embedding_dimensions", remoteEmbeddingGlobal, "answers with 128 dimensions"},
		{"embedding_deployment", `model_catalog: {embeddings: {default: missing}}`, `global.model_catalog.embeddings.default: unknown deployment "missing"`},
		{"admission_overflow", `model_catalog: {admission: {primary: {max_concurrency: 1, on_overflow: drop}}}`, "on_overflow"},
		{"admission_queue", `model_catalog: {admission: {primary: {max_concurrency: 1, max_queue: -1}}}`, "max_queue"},
		{"deployment_provider", `model_catalog: {deployments: {encoder: {artifact: models/encoder, provider: unknown}}}`, "unsupported provider"},
		{"prompt_profile", `model_catalog: {modules: {prompt_compression: {profile: unknown}}}`, "prompt_compression.profile"},
		{"remom_models", `integrations: {looper: {remom: {model_names: [""]}}}`, "remom.model_names was removed"},
		{"fusion_models", `integrations: {looper: {fusion: {model_names: [""]}}}`, "fusion.model_names was removed"},
		{"flow_backend", `integrations: {looper: {flow: {state: {store_backend: unknown}}}}`, "state.store_backend"},
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
  response_cache: {}
  memory: {default_similarity_threshold: 0}
model_catalog:
  admission:
    primary: {max_concurrency: 1, max_queue: 1, on_overflow: wait}
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
		"version": "v0.5",
		"global":  global,
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

const remoteEmbeddingGlobal = `model_catalog:
  deployments:
    remote-embed: {provider: openai_embeddings, endpoint: "http://127.0.0.1:8000/v1", model: test-embedding, capabilities: {dimensions: 128}}
stores:
  memory:
    embedding: {deployment: remote-embed, dimension: 64}
`

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
			name: "recipe_application_rule_label",
			mutate: func(cfg *RouterConfig) {
				cfg.Recipes = []RoutingRecipe{{Name: "internal", Profile: RoutingProfile{Signals: Signals{
					ComplexityRules: []ComplexityRule{{Name: "difficulty", Labels: []string{"impossible"}}},
				}}}}
			},
			want: `"impossible" is not a complexity label`,
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
