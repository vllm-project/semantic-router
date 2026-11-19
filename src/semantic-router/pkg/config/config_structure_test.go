package config

import (
	"os"
	"path/filepath"
	"slices"
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

func TestEntrypointDelegateModels(t *testing.T) {
	recipes := "recipes:\n  - name: main\n" + structureDecision
	for _, field := range []string{"review_model", "memory_model"} {
		otherField := "memory_model"
		if field == otherField {
			otherField = "review_model"
		}
		for _, tc := range []struct{ name, model, extra, target, err string }{
			{name: "omitted"},
			{name: "other entrypoint alias", model: "review-alias", target: "  - model_names: [review, review-alias]\n    recipe: main\n"},
			{name: "normalized", model: " review ", target: "  - model_names: [review]\n    recipe: main\n"},
			{name: "both calls to one target", model: "review", extra: "    OTHER: review\n", target: "  - model_names: [review]\n    recipe: main\n"},
			{name: "unknown", model: "missing", err: "unknown entrypoint model"},
			{name: "backend is not an entrypoint", model: "m", err: "unknown entrypoint model"},
			{name: "self alias", model: "main-alias", err: "own model name"},
			{name: "chain", model: "review", target: "  - model_names: [review]\n    recipe: main\n    FIELD: third\n  - model_names: [third]\n    recipe: main\n", err: "must not have a review_model or memory_model"},
			{name: "chain through the other field", model: "review", target: "  - model_names: [review]\n    recipe: main\n    OTHER: third\n  - model_names: [third]\n    recipe: main\n", err: "must not have a review_model or memory_model"},
			{name: "cycle", model: "review", target: "  - model_names: [review]\n    recipe: main\n    FIELD: main\n", err: "must not have a review_model or memory_model"},
			{name: "wrong API", model: "review", target: "  - api: systemone\n    model_names: [review]\n    recipe: main\n", err: "unknown entrypoint model"},
		} {
			t.Run(field+"/"+tc.name, func(t *testing.T) {
				fields := strings.NewReplacer("FIELD", field, "OTHER", otherField)
				entries := "entrypoints:\n  - model_names: [main, main-alias]\n    recipe: main\n    default: true\n"
				if tc.model != "" {
					entries += "    " + field + ": \"" + tc.model + "\"\n"
				}
				cfg, err := parseStructure(t, recipes+entries+fields.Replace(tc.extra+tc.target))
				if tc.err != "" {
					if err == nil || !strings.Contains(err.Error(), "entrypoints[0]."+field) || !strings.Contains(err.Error(), tc.err) {
						t.Fatalf("error = %v, want path and %s", err, tc.err)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				want := map[string]string{field: strings.TrimSpace(tc.model)}
				if tc.extra != "" {
					want[otherField] = "review"
				}
				main := cfg.Entrypoints[0]
				exported := CanonicalConfigFromRouterConfig(cfg).Entrypoints[0]
				for name, got := range map[string]map[string]string{
					"normalized": {"review_model": main.ReviewModel, "memory_model": main.MemoryModel},
					"exported":   {"review_model": exported.ReviewModel, "memory_model": exported.MemoryModel},
				} {
					for _, key := range []string{field, otherField} {
						if got[key] != want[key] {
							t.Fatalf("%s %s = %q, want %q", name, key, got[key], want[key])
						}
					}
				}
			})
		}
	}
}

// Every name of the review_model target is a review entrypoint; the main
// entrypoint, a backend model and another API are not.
func TestIsReviewEntrypointMatchesTargetAliases(t *testing.T) {
	recipes := "recipes:\n  - name: main\n" + structureDecision
	cfg, err := parseStructure(t, recipes+"entrypoints:\n"+
		"  - model_names: [main]\n    recipe: main\n    default: true\n    review_model: review-alias\n"+
		"  - model_names: [review, review-alias]\n    recipe: main\n")
	if err != nil {
		t.Fatal(err)
	}
	for model, want := range map[string]bool{"review": true, "review-alias": true, "main": false, "m": false, "": false} {
		if got := cfg.IsReviewEntrypoint(ChatAPI, model); got != want {
			t.Errorf("IsReviewEntrypoint(chat, %q) = %v, want %v", model, got, want)
		}
	}
	if cfg.IsReviewEntrypoint(SystemOneAPI, "review") {
		t.Error("a review_model binds within its own API")
	}
	var empty *RouterConfig
	if empty.IsReviewEntrypoint(ChatAPI, "review") {
		t.Error("a nil config has no review entrypoint")
	}
}

// A target that one entrypoint sends its memory calls to and another its
// reviews holds both roles, whichever entrypoint comes first.
func TestADelegateTargetHoldsEveryRoleItIsGiven(t *testing.T) {
	recipes := "recipes:\n  - name: main\n" + structureDecision
	cfg, err := parseStructure(t, recipes+"entrypoints:\n"+
		"  - model_names: [main]\n    recipe: main\n    default: true\n    memory_model: helper\n"+
		"  - model_names: [coding]\n    recipe: main\n    review_model: helper\n"+
		"  - model_names: [helper]\n    recipe: main\n")
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.IsReviewEntrypoint(ChatAPI, "helper") {
		t.Error("IsReviewEntrypoint(chat, helper) = false, want true")
	}
	if got := cfg.DelegateRoles(ChatAPI, "helper"); !slices.Equal(got, []string{DelegateMemory, DelegateReview}) {
		t.Errorf("DelegateRoles(chat, helper) = %q, want memory and review", got)
	}
}

// The documented gateway for Claude Code and Codex accepts each harness's
// losses only on its own entrypoint.
func TestOneGatewayExampleScopesLossesToEachHarness(t *testing.T) {
	page := filepath.Join("..", "..", "..", "..", "website", "docs", "installation", "agent-harness.md")
	content, err := os.ReadFile(page)
	if err != nil {
		t.Fatal(err)
	}
	for _, fence := range docFences(string(content)) {
		if fence.lang != "yaml" || !strings.Contains(fence.body, "model_names: [vllm-sr/codex]") {
			continue
		}
		cfg, err := ParseYAMLBytes([]byte(fence.body))
		if err != nil {
			t.Fatal(err)
		}
		for model, want := range map[string][]string{
			"vllm-sr/claude-code": {"tools.defer_loading", "tool_result.is_error", "messages[].role", "content.file"},
			"vllm-sr/codex":       {"reasoning.signature"},
		} {
			entrypoint, ok := cfg.ResolveEntrypoint(ChatAPI, model)
			if got := cfg.AcceptedLosses(&entrypoint); !ok || !slices.Equal(got, want) {
				t.Fatalf("%s accepts %v, want %v", model, got, want)
			}
		}
		if len(cfg.AcceptedLosses(nil)) != 0 {
			t.Fatalf("global losses = %v, want none", cfg.AcceptedLosses(nil))
		}
		return
	}
	t.Fatal("one-gateway example missing")
}

// The documented configs must resolve a review or memory request to its own
// recipe and backend.
func TestCodexDelegateExamplesRouteToTheirRecipes(t *testing.T) {
	page := filepath.Join("..", "..", "..", "..", "website", "docs", "installation", "agent-harness.md")
	content, err := os.ReadFile(page)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		recipe, target, backend string
		delegate                func(EntrypointMapping) string
	}{
		{"review", "vllm-sr/review", "codex-auto-review", func(e EntrypointMapping) string { return e.ReviewModel }},
		{"memory", "vllm-sr/memory", "local-memory", func(e EntrypointMapping) string { return e.MemoryModel }},
	} {
		t.Run(tc.recipe, func(t *testing.T) {
			for _, fence := range docFences(string(content)) {
				if fence.lang != "yaml" || !strings.Contains(fence.body, tc.recipe+"_model: "+tc.target) {
					continue
				}
				cfg, err := ParseYAMLBytes([]byte(fence.body))
				if err != nil {
					t.Fatal(err)
				}
				main, ok := cfg.ResolveEntrypoint(ChatAPI, "vllm-sr/auto")
				if !ok || !main.Default || tc.delegate(main) != tc.target {
					t.Fatalf("main entrypoint = %+v", main)
				}
				recipe, ok := cfg.RecipeForRequestModel(tc.target)
				if !ok || string(recipe.Name) != tc.recipe || len(recipe.Profile.Decisions) != 1 {
					t.Fatalf("%s recipe = %+v", tc.recipe, recipe)
				}
				if got := recipe.Profile.Decisions[0].ModelRefs[0].Model; got != tc.backend {
					t.Fatalf("%s backend = %q", tc.recipe, got)
				}
				if tc.recipe == "review" && (!cfg.IsReviewEntrypoint(ChatAPI, tc.target) || cfg.IsReviewEntrypoint(ChatAPI, "vllm-sr/auto")) {
					t.Fatal("only the review_model target is a review entrypoint")
				}
				return
			}
			t.Fatalf("complete Codex %s example missing", tc.recipe)
		})
	}
}
