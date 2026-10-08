package config

import (
	"gopkg.in/yaml.v2"
	"reflect"
	"strings"
	"testing"
)

func TestEffectiveEntrypointsDefaultOverrideAndSource(t *testing.T) {
	for _, tc := range []struct {
		name     string
		entries  []EntrypointMapping
		defaults []string
		source   EntrypointSource
	}{
		{"omitted", nil, []string{DefaultEntrypointModel}, EntrypointBuiltin},
		{"empty", []EntrypointMapping{}, []string{DefaultEntrypointModel}, EntrypointBuiltin},
		{"named only", []EntrypointMapping{{ModelNames: []string{"support"}, Recipe: "support"}}, []string{DefaultEntrypointModel}, EntrypointBuiltin},
		{"override", []EntrypointMapping{{ModelNames: []string{"company/auto", "MoM"}, Recipe: DefaultRecipeName}}, []string{"company/auto", "MoM"}, EntrypointExplicit},
		{"multiple mappings", []EntrypointMapping{{ModelNames: []string{"first"}, Recipe: DefaultRecipeName}, {ModelNames: []string{"second"}, Recipe: DefaultRecipeName}}, []string{"first", "second"}, EntrypointExplicit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{Entrypoints: tc.entries}
			if got := cfg.DefaultEntrypointNames(); !reflect.DeepEqual(got, tc.defaults) {
				t.Fatalf("default names=%v, want %v", got, tc.defaults)
			}
			for _, name := range tc.defaults {
				entry, ok := cfg.ResolveEntrypoint(ChatAPI, name)
				if !ok || entry.Recipe != DefaultRecipeName || entry.Source != tc.source {
					t.Fatalf("entrypoint=%+v, found=%v", entry, ok)
				}
				if _, ok := cfg.ResolveEntrypoint(SystemOneAPI, name); ok {
					t.Fatal("Chat entrypoint leaked into native System One")
				}
			}
			if len(cfg.Entrypoints) != len(tc.entries) {
				t.Fatal("effective normalization mutated authored config")
			}
			cfg.Entrypoints = nil
			if names := cfg.DefaultEntrypointNames(); !reflect.DeepEqual(names, []string{DefaultEntrypointModel}) {
				t.Fatalf("removing override did not restore default: %v", names)
			}
		})
	}
}

func TestEntrypointNamesHaveNoLegacyOrAlgorithmMeaning(t *testing.T) {
	cfg := &RouterConfig{}
	for _, name := range []string{"auto", "MoM", "vllm-sr/remom", "vllm-sr/fusion", "openrouter/fusion", "vllm-sr/flow"} {
		if cfg.IsEntrypointModelName(name) {
			t.Fatalf("undeclared name %q became an entrypoint", name)
		}
		cfg.Entrypoints = []EntrypointMapping{{ModelNames: []string{name}, Recipe: DefaultRecipeName}}
		if !cfg.IsEntrypointModelName(name) {
			t.Fatalf("explicit name %q did not resolve", name)
		}
		cfg.Entrypoints = nil
	}
}

func TestCanonicalEntrypointRoundTripPreservesDefaultProvenance(t *testing.T) {
	for _, extra := range []string{"", "entrypoints: []\n", "entrypoints:\n  - model_names: [company/route]\n    recipe: default\n"} {
		cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + extra))
		if err != nil {
			t.Fatal(err)
		}
		exported, err := yamlForEntrypointRoundTrip(cfg)
		if err != nil {
			t.Fatal(err)
		}
		again, err := ParseYAMLBytes(exported)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(cfg.Entrypoints, again.Entrypoints) {
			t.Fatalf("source changed: before=%+v after=%+v", cfg.Entrypoints, again.Entrypoints)
		}
		if len(cfg.Entrypoints) == 0 && strings.Contains(string(exported), "model_names") {
			t.Fatal("export materialized a builtin alias as an explicit override")
		}
	}
}

func yamlForEntrypointRoundTrip(cfg *RouterConfig) ([]byte, error) {
	return yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
}

// Unmapped named recipes remain dormant; the default recipe is always reachable.
func moveTestRoutingToUnmappedRecipe(cfg *RouterConfig) {
	cfg.Recipes = []RoutingRecipe{
		{Name: DefaultRecipeName},
		{Name: "unmapped", Profile: RoutingProfile{Signals: cfg.Signals, Projections: cfg.Projections, Decisions: cfg.Decisions, ModelBindings: cfg.ModelBindings}},
	}
}
