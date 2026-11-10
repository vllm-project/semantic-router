package config

import (
	"strings"
	"testing"
)

const structureHead = `version: v0.5
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
`

const structureDecision = `    decisions:
      - name: d
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: m}]
`

func parseStructure(t *testing.T, body string) (*RouterConfig, error) {
	t.Helper()
	return ParseYAMLBytes([]byte(structureHead + body))
}

func TestOneRecipeNeedsNoEntrypoints(t *testing.T) {
	cfg, err := parseStructure(t, "recipes:\n  - name: privacy\n"+structureDecision)
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.DefaultRecipe().Name; got != "privacy" {
		t.Fatalf("default recipe = %q, want the only recipe", got)
	}
	entrypoints := cfg.EffectiveEntrypoints(ChatAPI)
	if len(entrypoints) != 1 || entrypoints[0].Recipe != "privacy" ||
		len(entrypoints[0].ModelNames) != 1 || entrypoints[0].ModelNames[0] != DefaultEntrypointModel {
		t.Fatalf("entrypoints = %+v, want %s serving the only recipe", entrypoints, DefaultEntrypointModel)
	}
}

func TestSeveralRecipesNeedExactlyOneDefaultEntrypoint(t *testing.T) {
	recipes := "recipes:\n  - name: fast\n" + structureDecision + "  - name: careful\n" + structureDecision
	for _, tc := range []struct {
		name        string
		entrypoints string
		want        RecipeName
		err         string
	}{
		{name: "no entrypoints", err: "entrypoints: 2 recipes need entrypoints, exactly one of them marked default: true"},
		{
			name: "none marked default",
			entrypoints: `entrypoints:
  - model_names: [team/fast]
    recipe: fast
  - model_names: [team/careful]
    recipe: careful
`,
			err: "entrypoints: 2 recipes need exactly one entrypoint marked default: true",
		},
		{
			name: "two marked default",
			entrypoints: `entrypoints:
  - model_names: [team/fast]
    recipe: fast
    default: true
  - model_names: [team/careful]
    recipe: careful
    default: true
`,
			err: "entrypoints[1].default: entrypoints[0] is already the default; mark exactly one entrypoint default",
		},
		{
			name: "one marked default",
			entrypoints: `entrypoints:
  - model_names: [team/fast]
    recipe: fast
  - model_names: [team/careful]
    recipe: careful
    default: true
`,
			want: "careful",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := parseStructure(t, recipes+tc.entrypoints)
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("error = %v, want %q", err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := cfg.DefaultRecipe().Name; got != tc.want {
				t.Fatalf("default recipe = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestPluginsLayerFieldByField(t *testing.T) {
	cfg, err := parseStructure(t, `global:
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        capture_request_body: false
        capture_response_body: false
        max_body_bytes: 1024
recipes:
  - name: default
    plugins:
      - type: router_replay
        configuration:
          capture_response_body: true
    decisions:
      - name: d
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: m}]
        plugins:
          - type: router_replay
            configuration:
              max_body_bytes: 4096
      - name: plain
        priority: 0
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: m}]
`)
	if err != nil {
		t.Fatal(err)
	}
	recipe := cfg.DefaultRecipe()
	for _, tc := range []struct {
		decision                  string
		requestBody, responseBody bool
		maxBodyBytes              int
	}{
		{decision: "d", responseBody: true, maxBodyBytes: 4096},
		{decision: "plain", responseBody: true, maxBodyBytes: 1024},
	} {
		var decision *Decision
		for i := range recipe.Profile.Decisions {
			if recipe.Profile.Decisions[i].Name == tc.decision {
				decision = &recipe.Profile.Decisions[i]
			}
		}
		replay := cfg.EffectiveRouterReplayConfig(decision)
		if replay == nil {
			t.Fatalf("decision %s records no replay", tc.decision)
		}
		if replay.CaptureRequestBody != tc.requestBody || replay.CaptureResponseBody != tc.responseBody ||
			replay.MaxBodyBytes != tc.maxBodyBytes {
			t.Fatalf("decision %s replay = %+v, want request body %t (global), response body %t (recipe), max body %d",
				tc.decision, replay, tc.requestBody, tc.responseBody, tc.maxBodyBytes)
		}
	}
	if replay := cfg.EffectiveRouterReplayConfig(nil); replay == nil || !replay.CaptureResponseBody || replay.MaxBodyBytes != 1024 {
		t.Fatalf("a request no decision matches records %+v, want the recipe's layer over global.plugins", replay)
	}
}

func TestModelFactsAreOnProviderModels(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(strings.Replace(structureHead, "  models:\n", `  models:
    - name: described
      description: A model routing reads about and the Router does not serve
      capabilities: [chat, reasoning]
      context_window_size: 32768
`, 1) + "recipes:\n  - name: default\n" + structureDecision))
	if err != nil {
		t.Fatal(err)
	}
	facts := cfg.ModelConfig["described"]
	if facts.ContextWindowSize != 32768 || len(facts.Capabilities) != 2 || facts.Capabilities[1] != "reasoning" {
		t.Fatalf("facts of described = %+v", facts)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	for _, model := range exported.Providers.Models {
		if model.Name == "described" && model.ContextWindowSize != 32768 {
			t.Fatalf("exported providers.models[described] = %+v, want its facts", model)
		}
	}
}

func TestPreviousVersionIsRefusedWithOneLine(t *testing.T) {
	_, err := ParseYAMLBytes([]byte("version: v0.3\nrouting:\n  modelCards:\n    - name: m\n"))
	want := "version: invalid_value: config schema v0.3 is removed; run `vllm-sr config migrate` (version: v0.5)"
	if err == nil || err.Error() != want {
		t.Fatalf("error = %v, want %q", err, want)
	}
}
