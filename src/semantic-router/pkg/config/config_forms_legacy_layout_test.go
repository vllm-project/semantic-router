package config

import (
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// What a layout before config v0.3 held that has no place in v0.5 is dropped
// with a note at its place: the model registry, an endpoint no model prefers
// and the stray top-level jailbreak mapping.
func TestPreV03MigrationNotes(t *testing.T) {
	for name, want := range map[string]map[string]string{
		"removed-section-pre-v03-mom-registry": {
			"mom_registry": "migration_action: mom_registry is dropped: a model's repository is the artifact of its deployment in global.model_catalog.deployments",
		},
		"removed-section-pre-v03-layout": {
			"vllm_endpoints":         "migration_action: vllm_endpoints[spare] is dropped: no model_config model names it",
			"jailbreak_mapping_path": "migration_change: top-level jailbreak_mapping_path is dropped",
		},
	} {
		wantMigrationNotes(t, name, want)
	}
}

// The model names of a Looper algorithm are served by the recipe migrate
// writes for them, or dropped when no decision ran them or an entrypoint
// serves them already, with a note at their place.
func TestLooperModelNameNotes(t *testing.T) {
	for name, want := range map[string]map[string]string{
		"removed-field-looper-model-names": {
			"global.integrations.looper.fusion.model_names": "migration_change: the fusion model names vllm-sr/fusion, team/panel are served by recipe default-fusion, which migrate wrote: recipe default with only its fusion decisions (panel); a request none of them matches now goes to the default model",
		},
		"removed-field-looper-model-names-unserved": {
			"global.integrations.looper.remom.model_names": "migration_change: the remom model names vllm-sr/remom are dropped: no decision of the default recipe runs the remom algorithm",
			"global.integrations.looper.flow.model_names":  "migration_change: the flow model names team/flow are dropped: an entrypoint serves them already",
		},
	} {
		wantMigrationNotes(t, name, want)
	}
}

// After migrate, a request for a Looper algorithm's model name evaluates the
// default recipe's decisions of that algorithm alone, as it did, and the
// default entrypoint every decision of the default recipe.
func TestMigratedLooperModelNamesSelectTheAlgorithmDecisions(t *testing.T) {
	for after, want := range map[string]map[string][]string{
		"migrate/removed-field-looper-model-names": {
			"vllm-sr/fusion": {"panel"},
			"team/panel":     {"panel"},
			"vllm-sr/auto":   {"panel", "direct"},
		},
		"migrate_releases/v0.4.0-accuracy": {
			"vllm-sr/fusion": {"accuracy_deliberation"},
			"vllm-sr/flow":   {"accuracy_workflow"},
			"vllm-sr/auto":   {"omni", "accuracy_workflow", "accuracy_long_context_direct", "accuracy_deliberation", "accuracy_direct"},
		},
	} {
		cfg, err := ParseYAMLBytesWithoutEnvExpansion(readGolden(t, filepath.Join("testdata", after, "after.yaml")))
		if err != nil {
			t.Fatalf("%s: %v", after, err)
		}
		for model, decisions := range want {
			recipe, ok := cfg.RecipeForRoutingModel(model)
			if !ok {
				t.Fatalf("%s: %s selects no recipe", after, model)
			}
			var got []string
			for _, decision := range recipe.Profile.Decisions {
				got = append(got, decision.Name)
			}
			if !slices.Equal(got, decisions) {
				t.Errorf("%s: %s evaluates the decisions %v of recipe %s, want %v", after, model, got, recipe.Name, decisions)
			}
		}
	}
}

// After migrate, each model name a configuration exposed reaches the recipe
// it reached before: the auto model names and the default recipe's
// entrypoints the default recipe, a Looper algorithm's names the recipe of its
// decisions, another entrypoint its own recipe. A backend model, and a name
// the earlier release did not serve, reach none.
func TestMigratedModelNamesReachTheirRecipes(t *testing.T) {
	for after, want := range map[string]map[string]string{
		"migrate_releases/v0.4.0-config": {
			"vllm-sr/auto": "default", "auto": "default", "exhaustive-reference": "default",
			"vllm-sr/exhaustive-default": "default", "vllm-sr/exhaustive-privacy": "privacy-first",
			"vllm-sr/remom": "default-remom", "vllm-sr/fusion": "default-fusion", "vllm-sr/flow": "default-flow",
			"qwen3-8b": "",
		},
		"migrate_releases/v0.4.0-vela-amd": {"vela-auto": "default", "vllm-sr/auto": ""},
		"migrate/removed-field-router-names-beside-entrypoint": {
			"vllm-sr/auto": "default", "auto": "default", "reference": "default", "team/default": "default", "team/private": "private",
		},
		"migrate/removed-field-router-names-single": {"vllm-sr/auto": "default", "auto": "default", "reference": "default"},
	} {
		cfg, err := ParseYAMLBytesWithoutEnvExpansion(readGolden(t, filepath.Join("testdata", after, "after.yaml")))
		if err != nil {
			t.Fatalf("%s: %v", after, err)
		}
		for model, recipe := range want {
			got, ok := cfg.RecipeForRoutingModel(model)
			switch {
			case recipe == "" && ok:
				t.Errorf("%s: %s reaches recipe %s, want none", after, model, got.Name)
			case recipe != "" && !ok:
				t.Errorf("%s: %s reaches no recipe, want %s", after, model, recipe)
			case recipe != "" && string(got.Name) != recipe:
				t.Errorf("%s: %s reaches recipe %s, want %s", after, model, got.Name, recipe)
			}
		}
	}
}

// After migrate, a configuration an earlier release shipped serves the model
// names that release served, those it served without the file naming them
// included: vllm-sr/auto, auto and MoM for the default recipe when no auto
// model name is written (releases before v0.4 served auto and MoM, and v0.5
// serves the default recipe as vllm-sr/auto), and a Looper algorithm's
// built-in name for its decisions when it names no model names. The names an
// entrypoint serves already stay its own.
func TestMigrateKeepsTheModelNamesAReleaseServed(t *testing.T) {
	auto := map[string]string{"vllm-sr/auto": "default", "auto": "default", "MoM": "default"}
	for before, want := range map[string]map[string]string{
		"testdata/migrate_releases/v0.4.0-config/before.yaml": {
			"vllm-sr/auto": "default", "auto": "default", "exhaustive-reference": "default",
			"vllm-sr/exhaustive-default": "default", "vllm-sr/exhaustive-privacy": "privacy-first",
			"vllm-sr/remom": "default-remom", "vllm-sr/fusion": "default-fusion", "vllm-sr/flow": "default-flow",
		},
		"testdata/migrate_releases/v0.4.0-cli-template/before.yaml":              auto,
		maintainedRepoPrefix + "config/recipes/built-in/v0.4/mom-v1/config.yaml": auto,
		"testdata/migrate_releases/v0.1.0-config/before.yaml":                    auto,
		"testdata/migrate_releases/v0.1.0-cli-template/before.yaml":              auto,
		"testdata/migrate_releases/v0.2.0-config/before.yaml":                    auto,
		"testdata/migrate_releases/v0.2.0-cli-template/before.yaml":              auto,
		"testdata/migrate/invalid-value-version-default-model-names/before.yaml": {
			"vllm-sr/auto": "default", "auto": "default", "MoM": "default", "vllm-sr/fusion": "default-fusion",
		},
		"testdata/migrate/invalid-value-version-served-model-names/before.yaml": {
			"vllm-sr/auto": "default", "auto": "default", "MoM": "private",
		},
	} {
		migrated, _, err := MigrateConfig(readGolden(t, before))
		if err != nil {
			t.Errorf("%s: %v", before, err)
			continue
		}
		cfg, err := ParseYAMLBytesWithoutEnvExpansion(migrated)
		if err != nil {
			t.Fatalf("%s: %v", before, err)
		}
		served := map[string]string{}
		for _, entrypoint := range cfg.EffectiveEntrypoints(ChatAPI) {
			for _, name := range entrypoint.ModelNames {
				served[name] = string(entrypoint.Recipe)
			}
		}
		if !maps.Equal(served, want) {
			t.Errorf("%s: migrated, it serves %v, want %v", before, served, want)
		}
	}
}

// An auto model name another entrypoint serves already stays that
// entrypoint's, even when it is the only one written.
func TestMigrateKeepsAServedAutoModelNameOnItsEntrypoint(t *testing.T) {
	before := append(readGolden(t, "testdata/migrate/invalid-value-version-served-model-names/before.yaml"),
		[]byte("global:\n  router:\n    auto_model_names: [MoM]\n")...)
	migrated, _, err := MigrateConfig(before)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(migrated)
	if err != nil {
		t.Fatal(err)
	}
	for _, entrypoint := range cfg.EffectiveEntrypoints(ChatAPI) {
		if slices.Contains(entrypoint.ModelNames, "MoM") && entrypoint.Recipe != "private" {
			t.Errorf("MoM reaches %s, want private", entrypoint.Recipe)
		}
	}
}

// The recipe migrate writes for Looper model names copies the default recipe
// as migrated, so a note that asks something of the default recipe, or of a
// decision the copy keeps, asks it of the copy too.
func TestLooperModelNameRecipeNotesCopiedPlaces(t *testing.T) {
	for noted, want := range map[string]bool{
		"recipes[default]":                             true,
		"recipes[default].signals":                     true,
		"recipes[default].decisions[panel].plugins[0]": true,
		"recipes[default].decisions[direct]":           false,
		"recipes[default-other]":                       false,
		"providers.models[small]":                      false,
	} {
		var document yaml.Node
		if err := yaml.Unmarshal(readGolden(t, filepath.Join("testdata", "migrate", "removed-field-looper-model-names", "before.yaml")), &document); err != nil {
			t.Fatal(err)
		}
		m := &Migration{root: documentRoot(&document), notes: []ConfigWarning{{Code: migrationActionCode, Field: noted, Message: "fit it"}}}
		for _, at := range looperModelNameForms[0].matches(m.root) {
			if err := migrateLooperModelNames(m, at); err != nil {
				t.Fatal(err)
			}
		}
		if got := slices.ContainsFunc(m.notes, func(note ConfigWarning) bool { return note.Field == "recipes[default-fusion]" }); got != want {
			t.Errorf("a note at %s: the copy has a note %v, want %v (%+v)", noted, got, want, m.notes)
		}
	}
}

// wantMigrationNotes migrates a golden case and wants a note at each path
// that starts with its phrase.
func wantMigrationNotes(t *testing.T, name string, want map[string]string) {
	t.Helper()
	_, notes, err := MigrateConfig(readGolden(t, filepath.Join("testdata", "migrate", name, "before.yaml")))
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	for path, phrase := range want {
		if !slices.ContainsFunc(notes, func(note ConfigWarning) bool {
			return note.Field == path && strings.HasPrefix(note.Code+": "+note.Message, phrase)
		}) {
			t.Errorf("%s: no note at %s starting %q in %+v", name, path, phrase, notes)
		}
	}
}
