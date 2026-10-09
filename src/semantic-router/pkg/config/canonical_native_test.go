package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const nativeRoutingTestYAML = `
version: v0.3
listeners:
  - name: native
    port: 8801
    systemone:
      models: [vllm-sr/auto]
providers:
  models:
    - name: kai
      api_format: systemone
      deployment: local-kai
    - name: strong
      api_format: systemone
      provider_model_id: vllm-sr/Decision-2.0-Nox-4B
      backend_refs:
        - provider: systemone-compatible
          base_url: http://127.0.0.1:8900/v1
          api_key_env: NATIVE_TEST_TOKEN
    - name: reviewer
      api_format: openai
      backend_refs:
        - provider: openai-compatible
          base_url: http://127.0.0.1:8901/v1
global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
entrypoints:
  - api: systemone
    model_names: [vllm-sr/auto]
    recipe: native-decisions
recipes:
  - name: native-decisions
    routing:
      budget: {deadline: 3s, max_calls: 4}
      decisions:
        - name: classify
          rules: {}
          modelRefs: [{model: kai}, {model: strong}, {model: reviewer}]
          algorithm:
            type: cascade
            quality:
              type: uncalibrated
              acceptance:
                rules:
                  - {state: document, question: task, question_type: choice, field: confidence, predicate: {gte: 0.8}}
            stages:
              - {name: fast, kind: native, model: kai}
              - {name: strong, kind: native, model: strong, timeout: 1s}
              - name: review
                kind: judge
                model: reviewer
                enabled: false
                generation: {max_output_tokens: 256}
`

func TestCanonicalNativeRoutingRoundTrip(t *testing.T) {
	t.Setenv("NATIVE_TEST_TOKEN", "test-only-value")
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	chat, ok := cfg.ResolveEntrypoint(ChatAPI, DefaultEntrypointModel)
	if !ok || chat.Source != EntrypointBuiltin || chat.Recipe != DefaultRecipeName {
		t.Fatalf("Chat default changed: %+v", chat)
	}
	native, ok := cfg.ResolveEntrypoint(SystemOneAPI, DefaultEntrypointModel)
	if !ok || native.Recipe != "native-decisions" || native.Source != EntrypointExplicit {
		t.Fatalf("native entrypoint = %+v", native)
	}
	if cfg.ModelConfig["kai"].Deployment != "local-kai" || cfg.ModelConfig["strong"].APIFormat != APIFormatSystemOne {
		t.Fatal("local/native provider identity was lost")
	}
	used := ModelRuntimeDeploymentsInUse(cfg)
	if len(used) != 1 || used["local-kai"].Artifact != "vllm-sr/Decision-2.0-Kai-0.6B" {
		t.Fatalf("native runtime demand = %+v", used)
	}
	canonical := CanonicalConfigFromRouterConfig(cfg)
	encoded, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "test-only-value") || !strings.Contains(string(encoded), "api_key_env: NATIVE_TEST_TOKEN") {
		t.Fatal("export must preserve the credential reference without revealing its value")
	}
	roundTrip, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	before := findRecipe(cfg.Recipes, "native-decisions").Profile
	after := findRecipe(roundTrip.Recipes, "native-decisions").Profile
	if !reflect.DeepEqual(before.Budget, after.Budget) || !reflect.DeepEqual(before.Decisions, after.Decisions) {
		t.Fatal("native policy changed across canonical export/parse")
	}
}

func TestCanonicalNativeRoutingRejectsAmbiguousContracts(t *testing.T) {
	tests := []struct{ name, old, replacement, want string }{
		{"unknown API", "api: systemone", "api: mystery", ".api must"},
		{"both transports", "deployment: local-kai", "deployment: local-kai\n      backend_refs: [{provider: systemone-compatible, endpoint: localhost:9000}]", "cannot be combined"},
		{"unknown deployment", "deployment: local-kai", "deployment: missing", "must reference a model_runtime"},
		{"missing budget", "budget: {deadline: 3s, max_calls: 4}", "budget: null", "routing.budget"},
		{"zero budget", "max_calls: 4", "max_calls: 0", "max_calls must be positive"},
		{"infinite deadline", "deadline: 3s", "deadline: 0s", "positive duration"},
		{"unknown stage model", "model: strong, timeout", "model: absent, timeout", "undeclared modelRef"},
		{"duplicate stage", "name: strong, kind", "name: fast, kind", "distinct and non-empty"},
		{"wrong role", "name: strong, kind: native", "name: strong, kind: judge", "generation.max_output_tokens"},
		{"wrong native protocol", "name: strong, kind: native, model: strong", "name: strong, kind: native, model: reviewer", "requires api_format: systemone"},
		{"unsupported fallback", "kind: judge", "kind: typed_fallback", "separate response contract"},
		{"missing LLM bound", "generation: {max_output_tokens: 256}", "generation: {max_output_tokens: 0}", "generation.max_output_tokens"},
		{"raw score", "field: confidence", "field: value", "score value is not confidence"},
		{"unsupported native type", "question_type: choice", "question_type: span", "question_type must be"},
		{"native Chat recipe", "api: systemone", "api: chat", "native algorithms require"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			raw := strings.Replace(nativeRoutingTestYAML, test.old, test.replacement, 1)
			if test.name == "native Chat recipe" {
				raw = strings.ReplaceAll(raw, "vllm-sr/auto", "native-alias")
			}
			if raw == nativeRoutingTestYAML {
				t.Fatal("test mutation did not match")
			}
			_, err := ParseYAMLBytes([]byte(raw))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v, want error containing %q", err, test.want)
			}
		})
	}
}

func TestNativeRuntimeDemandRequiresPublicationAndEnabledStage(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	cfg.Listeners[0].SystemOne = nil
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("unpublished native recipes started models: %v", used)
	}
	cfg.Listeners[0].SystemOne = &ListenerSystemOne{Models: []string{DefaultEntrypointModel}}
	recipe := findRecipe(cfg.Recipes, "native-decisions")
	enabled := false
	recipe.Profile.Decisions[0].Algorithm.Stages[0].Enabled = &enabled
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("disabled native action started models: %v", used)
	}
}

func TestSystemOneDoesNotInheritChatEntrypoints(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.EffectiveEntrypoints(SystemOneAPI); len(got) != 0 {
		t.Fatalf("native API inherited Chat entrypoints: %+v", got)
	}
}

func TestSystemOneProviderAliasesAreDirectNativeTargets(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "models: [vllm-sr/auto]", "models: [kai, strong]", 1)
	cfg, err := ParseYAMLBytes([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	for _, alias := range []string{"kai", "strong"} {
		if !cfg.IsSystemOneBackend(alias) {
			t.Fatalf("native provider alias %q is not directly callable", alias)
		}
	}
	for _, alias := range []string{"reviewer", "vllm-sr/Decision-2.0-Nox-4B", "local-kai", "vllm-sr/auto"} {
		if cfg.IsSystemOneBackend(alias) {
			t.Fatalf("non-provider identity %q became a native provider alias", alias)
		}
	}
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 1 || used["local-kai"].Artifact == "" {
		t.Fatalf("direct native provider did not request its local deployment: %+v", used)
	}
	cfg.Listeners[0].SystemOne.Models = []string{"reviewer"}
	if err := validateListenerSystemOne(cfg, cfg.Listeners[0]); err == nil {
		t.Fatal("native listener accepted a Chat provider")
	}
}

func TestNativeAliasCollisionsAreScopedAndUnambiguous(t *testing.T) {
	// A System One routing name can coexist with a Chat provider. The two
	// APIs have separate request model namespaces.
	raw := strings.ReplaceAll(nativeRoutingTestYAML, "vllm-sr/auto", "reviewer")
	if _, err := ParseYAMLBytes([]byte(raw)); err != nil {
		t.Fatalf("cross-API alias was incorrectly rejected: %v", err)
	}
	raw = strings.ReplaceAll(nativeRoutingTestYAML, "vllm-sr/auto", "strong")
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil || !strings.Contains(err.Error(), "native provider model") {
		t.Fatalf("same-API routing/provider collision was accepted: %v", err)
	}
	// Same-target aliasing is harmless, but an external backend cannot
	// shadow a local deployment's public name.
	for _, test := range []struct {
		publicName string
		allowed    bool
	}{{"kai", true}, {"strong", false}} {
		raw := strings.Replace(nativeRoutingTestYAML, "device: cpu", "device: cpu\n        public_name: "+test.publicName, 1)
		_, err := ParseYAMLBytes([]byte(raw))
		if test.allowed && err != nil {
			t.Fatalf("same-target alias failed: %v", err)
		}
		if !test.allowed && (err == nil || !strings.Contains(err.Error(), "native alias conflicts")) {
			t.Fatalf("different-target collision accepted: %v", err)
		}
	}
}

func TestNativeModelsCannotBecomeChatDefaultsOrCandidates(t *testing.T) {
	raw := strings.Replace(nativeRoutingTestYAML, "providers:\n", "providers:\n  defaults: {model: kai}\n", 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil || !strings.Contains(err.Error(), "must be a Chat provider") {
		t.Fatalf("native alias became Chat default: %v", err)
	}
	raw = strings.Replace(nativeRoutingTestYAML, "entrypoints:\n", `routing:
  decisions:
    - name: invalid-chat
      rules: {}
      modelRefs: [{model: kai}]
entrypoints:
`, 1)
	if _, err := ParseYAMLBytes([]byte(raw)); err == nil || !strings.Contains(err.Error(), "Chat execution cannot use native") {
		t.Fatalf("native alias became Chat candidate: %v", err)
	}
	cfg, err := ParseYAMLBytes([]byte(nativeRoutingTestYAML))
	if err != nil {
		t.Fatal(err)
	}
	decision := Decision{Name: "chat-panel", Algorithm: &AlgorithmConfig{
		Type:   DecisionAlgorithmFusion,
		Fusion: &FusionAlgorithmConfig{QuorumFallbackTarget: "kai"},
	}}
	if err := validateNativeChatBoundary(cfg, decision); err == nil {
		t.Fatal("native alias became Chat quorum fallback")
	}
}
