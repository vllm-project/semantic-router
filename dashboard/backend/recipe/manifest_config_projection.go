package recipe

import (
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

type configDocument struct {
	Global struct {
		Router configRouterDocument `yaml:"router"`
	} `yaml:"global"`
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

type configRouterDocument struct {
	AutoModelName  string    `yaml:"auto_model_name"`
	AutoModelNames yaml.Node `yaml:"auto_model_names"`
}

type configProjection struct {
	counts          Counts
	autoModels      []string
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
	autoModels, err := projectAutoModels(config.Global.Router)
	if err != nil {
		return configProjection{}, err
	}
	projection.autoModels = autoModels

	models := map[string]struct{}{}
	for _, model := range projection.autoModels {
		models[model] = struct{}{}
	}
	for _, entrypoint := range config.Entrypoints {
		recipeName := strings.TrimSpace(entrypoint.Recipe)
		for _, model := range entrypoint.ModelNames {
			if model = strings.TrimSpace(model); model != "" {
				models[model] = struct{}{}
				projection.modelsByRecipe[recipeName] = append(projection.modelsByRecipe[recipeName], model)
			}
		}
		projection.modelsByRecipe[recipeName] = stableUnique(projection.modelsByRecipe[recipeName])
	}
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

func projectAutoModels(router configRouterDocument) ([]string, error) {
	// Presence is part of the compatibility contract: an explicit list, even an
	// empty one, replaces both the legacy field and historical defaults.
	if router.AutoModelNames.Kind != 0 {
		var explicit []string
		if err := router.AutoModelNames.Decode(&explicit); err != nil {
			return nil, fmt.Errorf("global.router.auto_model_names: %w", err)
		}
		return stableUnique(cleanStrings(explicit)), nil
	}

	configured := strings.TrimSpace(router.AutoModelName)
	if configured == "" {
		configured = "MoM"
	}
	return stableUnique([]string{"vllm-sr/auto", "auto", configured}), nil
}

func (projection configProjection) requestModelFor(expectedRecipe string) (string, error) {
	if candidates := projection.modelsByRecipe[strings.TrimSpace(expectedRecipe)]; len(candidates) > 0 {
		return candidates[0], nil
	}
	if len(projection.autoModels) > 0 {
		return projection.autoModels[0], nil
	}
	if len(projection.unifiedModelIDs) == 1 {
		return projection.unifiedModelIDs[0], nil
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
