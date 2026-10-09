package recipe

import (
	"errors"
	"io"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type configDocument struct {
	Routing struct {
		Decisions []struct {
			Name string `yaml:"name"`
		} `yaml:"decisions"`
	} `yaml:"routing"`
	Entrypoints []struct {
		ModelNames []string `yaml:"model_names"`
		Recipe     string   `yaml:"recipe"`
	} `yaml:"entrypoints"`
	Recipes []struct {
		Name    string `yaml:"name"`
		Routing struct {
			Decisions []struct {
				Name string `yaml:"name"`
			} `yaml:"decisions"`
		} `yaml:"routing"`
	} `yaml:"recipes"`
}

type configProjection struct {
	counts          Counts
	defaultModels   []string
	modelsByRecipe  map[string][]string
	unifiedModelIDs []string
}

func projectConfig(data []byte) (configProjection, error) {
	if len(data) == 0 {
		return configProjection{}, errors.New("missing or empty")
	}
	var config configDocument
	// Config is an independently versioned runtime contract, so intentionally
	// do not use KnownFields here. Only the count projection above is consumed.
	if err := decodeYAML(data, &config); err != nil {
		return configProjection{}, err
	}
	projection := configProjection{modelsByRecipe: map[string][]string{}}
	resolved := &routerconfig.RouterConfig{}
	for _, entrypoint := range config.Entrypoints {
		resolved.Entrypoints = append(resolved.Entrypoints, routerconfig.EntrypointMapping{ModelNames: entrypoint.ModelNames, Recipe: routerconfig.RecipeName(entrypoint.Recipe)})
	}
	models := map[string]struct{}{}
	for _, entrypoint := range resolved.EffectiveEntrypoints(routerconfig.ChatAPI) {
		recipeName := string(entrypoint.Recipe)
		for _, model := range entrypoint.ModelNames {
			models[model] = struct{}{}
			projection.modelsByRecipe[recipeName] = append(projection.modelsByRecipe[recipeName], model)
		}
		projection.modelsByRecipe[recipeName] = stableUnique(projection.modelsByRecipe[recipeName])
	}
	projection.defaultModels = projection.modelsByRecipe[string(routerconfig.DefaultRecipeName)]
	counts := Counts{UnifiedModels: len(models), Recipes: len(config.Recipes), Decisions: len(config.Routing.Decisions)}
	for _, recipe := range config.Recipes {
		counts.Decisions += len(recipe.Routing.Decisions)
	}
	if counts.Recipes == 0 {
		counts.Recipes = 1
	}
	projection.counts = counts
	projection.unifiedModelIDs = make([]string, 0, len(models))
	for model := range models {
		projection.unifiedModelIDs = append(projection.unifiedModelIDs, model)
	}
	sort.Strings(projection.unifiedModelIDs)
	return projection, nil
}

func (projection configProjection) requestModelFor(expectedRecipe string) (string, error) {
	if candidates := projection.modelsByRecipe[strings.TrimSpace(expectedRecipe)]; len(candidates) > 0 {
		return candidates[0], nil
	}
	if strings.TrimSpace(expectedRecipe) == "" && len(projection.defaultModels) > 0 {
		return projection.defaultModels[0], nil
	}
	return "", errors.New("config has no unambiguous request-facing model")
}

func decodeYAML(data []byte, target any) error {
	decoder := yaml.NewDecoder(strings.NewReader(string(data)))
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("must contain exactly one YAML document")
		}
		return err
	}
	return nil
}
