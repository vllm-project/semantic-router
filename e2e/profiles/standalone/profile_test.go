package standalone

import (
	"os"
	"slices"
	"testing"

	"gopkg.in/yaml.v3"
)

// The error testcase needs an actual route miss, while the positive and
// failover cases still need explicit backend choices in the same deployment.
func TestErrorFixtureDoesNotFallBackToThePositiveRoute(t *testing.T) {
	type decision struct {
		Name      string `yaml:"name"`
		ModelRefs []struct {
			Model string `yaml:"model"`
		} `yaml:"modelRefs"`
	}
	var values struct {
		Config struct {
			Providers struct {
				Defaults struct {
					Model string `yaml:"model"`
				} `yaml:"defaults"`
			} `yaml:"providers"`
			Routing struct {
				Decisions []decision `yaml:"decisions"`
			} `yaml:"routing"`
			Entrypoints []struct {
				ModelNames []string `yaml:"model_names"`
				Recipe     string   `yaml:"recipe"`
			} `yaml:"entrypoints"`
			Recipes []struct {
				Name    string `yaml:"name"`
				Routing struct {
					Decisions []decision `yaml:"decisions"`
				} `yaml:"routing"`
			} `yaml:"recipes"`
		} `yaml:"configOverride"`
	}
	raw, err := os.ReadFile("values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err = yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(NewProfile().GetTestCases(), "routing-error-codes") {
		t.Fatal("the standalone profile must exercise its error fixture")
	}
	if values.Config.Providers.Defaults.Model != "" {
		t.Fatal("a provider default would turn the required no_route response into a successful fallback")
	}
	var errorRecipe string
	for _, entrypoint := range values.Config.Entrypoints {
		if slices.Contains(entrypoint.ModelNames, "e2e-no-route") {
			errorRecipe = entrypoint.Recipe
		}
	}
	if errorRecipe == "" {
		t.Fatal("the error probe must resolve to a declared recipe")
	}
	var found bool
	for _, recipe := range values.Config.Recipes {
		if recipe.Name == errorRecipe {
			found = true
			if len(recipe.Routing.Decisions) != 0 {
				t.Fatal("the error recipe must not select a backend")
			}
		}
	}
	if !found {
		t.Fatalf("the error recipe %q is missing", errorRecipe)
	}
	modelsByDecision := map[string][]string{}
	for _, route := range values.Config.Routing.Decisions {
		for _, ref := range route.ModelRefs {
			modelsByDecision[route.Name] = append(modelsByDecision[route.Name], ref.Model)
		}
	}
	for name, want := range map[string][]string{
		"standalone_default":  {"primary-model"},
		"standalone_failover": {"unreachable-model", "fallback-model"},
	} {
		if !slices.Equal(modelsByDecision[name], want) {
			t.Fatalf("%s must retain its explicit backend selection: got %v, want %v", name, modelsByDecision[name], want)
		}
	}
}
