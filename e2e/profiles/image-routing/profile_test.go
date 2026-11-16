package imagerouting

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"gopkg.in/yaml.v3"
)

// The image question goes to provider-mocker's System One fixture, which
// declares that it reads images, together with the request's images.
func TestProfileAsksTheImageQuestionOfTheSystemOneFixture(t *testing.T) {
	config := profileSection(t, profileValues(t), "config")
	deployment := profileSection(t, config, "global", "model_catalog", "deployments", "fake-systemone")
	if deployment["endpoint"] != "http://provider-mocker.default.svc.cluster.local:8000/v1/systemone" ||
		!slices.Contains(stringList(profileSection(t, deployment, "capabilities")["modalities"]), "image") {
		t.Fatalf("fake-systemone must be provider-mocker's fixture and read images: %#v", deployment)
	}
	rule := namedEntry(t, profileSection(t, onlyRecipe(t, config), "signals")["systemone"], "photo_kind")
	if rule["deployment"] != "fake-systemone" || profileSection(t, rule, "input")["image"] != true {
		t.Fatalf("photo_kind must ask fake-systemone with the request's images: %#v", rule)
	}
}

// Each decision with an output names one model whose facts and API carry that
// output, so the profile loads.
func TestProfileOutputDecisionsNameModelsThatCarryTheirOutput(t *testing.T) {
	config := profileSection(t, profileValues(t), "config")
	models := profileSection(t, config, "providers")["models"]
	decisions := onlyRecipe(t, config)["decisions"]
	for decision, want := range map[string]struct{ model, api string }{
		"image":          {"mock/image-generation", "images"},
		"text-and-image": {"mock/omni-image", "responses"},
	} {
		entry := namedEntry(t, decisions, decision)
		model := namedEntry(t, models, want.model)
		refs, _ := entry["modelRefs"].([]any)
		var ref map[string]any
		if len(refs) == 1 {
			ref, _ = refs[0].(map[string]any)
		}
		if _, ok := entry["output"].(map[string]any); !ok || ref["model"] != want.model ||
			model["api_format"] != want.api || !slices.Contains(stringList(model["capabilities"]), "image_generation") {
			t.Fatalf("decision %s must answer with %s on the %s API: %#v", decision, want.model, want.api, entry)
		}
	}
}

func profileValues(t *testing.T) map[string]any {
	t.Helper()

	raw, err := os.ReadFile(filepath.Base(valuesFile))
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	return values
}

func onlyRecipe(t *testing.T, config map[string]any) map[string]any {
	t.Helper()

	recipes, _ := config["recipes"].([]any)
	if len(recipes) != 1 {
		t.Fatalf("the profile must declare one recipe: %#v", config["recipes"])
	}
	recipe, ok := recipes[0].(map[string]any)
	if !ok {
		t.Fatalf("the profile's recipe is not a mapping: %#v", recipes[0])
	}
	return recipe
}

func profileSection(t *testing.T, values map[string]any, keys ...string) map[string]any {
	t.Helper()

	for _, key := range keys {
		section, ok := values[key].(map[string]any)
		if !ok {
			t.Fatalf("profile section %q is missing or is not a mapping", key)
		}
		values = section
	}
	return values
}

func namedEntry(t *testing.T, entries any, name string) map[string]any {
	t.Helper()

	list, _ := entries.([]any)
	for _, raw := range list {
		if entry, ok := raw.(map[string]any); ok && entry["name"] == name {
			return entry
		}
	}
	t.Fatalf("no entry named %q in %#v", name, entries)
	return nil
}

func stringList(raw any) []string {
	list, _ := raw.([]any)
	values := make([]string, 0, len(list))
	for _, item := range list {
		if value, ok := item.(string); ok {
			values = append(values, value)
		}
	}
	return values
}
