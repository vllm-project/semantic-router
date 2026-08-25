package config

import (
	"fmt"
	"sort"
	"strings"
)

func validateCanonicalAuthoringFields(raw map[string]interface{}) error {
	unsupported := make([]string, 0)
	providers := nestedStringMap(raw["providers"])
	collectUnsupportedFields("providers", providers, []string{"defaults", "models"}, &unsupported)
	defaults := nestedStringMap(providers["defaults"])
	collectUnsupportedFields("providers.defaults", defaults, []string{
		"default_model", "reasoning_families", "default_reasoning_effort",
	}, &unsupported)
	for familyName, rawFamily := range nestedStringMap(defaults["reasoning_families"]) {
		collectUnsupportedFields(
			"providers.defaults.reasoning_families."+familyName,
			nestedStringMap(rawFamily), []string{"type", "parameter"}, &unsupported,
		)
	}
	models, _ := providers["models"].([]interface{})
	for modelIndex, rawModel := range models {
		path := fmt.Sprintf("providers.models[%d]", modelIndex)
		model := nestedStringMap(rawModel)
		collectUnsupportedFields(path, model, []string{
			"name", "reasoning_family", "provider_model_id", "backend_refs", "control",
			"pricing", "api_format", "external_model_ids",
		}, &unsupported)
		backendRefs, _ := model["backend_refs"].([]interface{})
		for connectionIndex, rawConnection := range backendRefs {
			collectUnsupportedFields(
				fmt.Sprintf("%s.backend_refs[%d]", path, connectionIndex),
				nestedStringMap(rawConnection),
				[]string{
					"name", "endpoint", "protocol", "weight", "type", "base_url", "provider",
					"auth_header", "auth_prefix", "extra_headers", "api_version", "chat_path",
					"credential", "api_key", "api_key_env",
				},
				&unsupported,
			)
		}
		control := nestedStringMap(model["control"])
		collectUnsupportedFields(path+".control", control, []string{"retry", "timeout"}, &unsupported)
		collectUnsupportedFields(
			path+".control.retry", nestedStringMap(control["retry"]),
			[]string{"count", "on"}, &unsupported,
		)
		timeout := nestedStringMap(control["timeout"])
		collectUnsupportedFields(
			path+".control.timeout", timeout, []string{"request", "stream"}, &unsupported,
		)
		for _, field := range []string{"request", "stream"} {
			if value, found := timeout[field]; found && value == "" {
				return fmt.Errorf("%s.control.timeout.%s must be omitted or contain a duration", path, field)
			}
		}
		pricing := nestedStringMap(model["pricing"])
		pricingFields := []string{
			"input_cost_per_million_tokens", "output_cost_per_million_tokens",
			"cache_read_cost_per_million_tokens", "cache_write_cost_per_million_tokens",
		}
		collectUnsupportedFields(path+".pricing", pricing, pricingFields, &unsupported)
		for _, field := range pricingFields {
			if value, found := pricing[field]; found {
				if value == nil {
					continue
				}
				if _, isString := value.(string); !isString {
					return fmt.Errorf("%s.pricing.%s must be a quoted decimal string", path, field)
				}
			}
		}
	}

	recipes, _ := raw["recipes"].([]interface{})
	for recipeIndex, rawRecipe := range recipes {
		path := fmt.Sprintf("recipes[%d]", recipeIndex)
		recipe := nestedStringMap(rawRecipe)
		collectUnsupportedFields(path, recipe, []string{"name", "description", "routing"}, &unsupported)
	}
	for _, source := range rawRoutingDocuments(raw) {
		decisions, _ := source.document["decisions"].([]interface{})
		for decisionIndex, rawDecision := range decisions {
			if _, found := nestedStringMap(rawDecision)["id"]; found {
				unsupported = append(unsupported, fmt.Sprintf("%s.decisions[%d].id", source.prefix, decisionIndex))
			}
		}
	}

	entrypoints, _ := raw["entrypoints"].([]interface{})
	for entrypointIndex, rawEntrypoint := range entrypoints {
		path := fmt.Sprintf("entrypoints[%d]", entrypointIndex)
		entrypoint := nestedStringMap(rawEntrypoint)
		collectUnsupportedFields(path, entrypoint, []string{
			"model_names", "recipe", "assignments",
		}, &unsupported)
		validateAuthoringAssignments(path+".assignments", nestedStringMap(entrypoint["assignments"]), &unsupported)
	}

	if len(unsupported) == 0 {
		return nil
	}
	sort.Strings(unsupported)
	return fmt.Errorf("unsupported v0.3 authoring fields: %s", strings.Join(unsupported, ", "))
}

func validateAuthoringAssignments(path string, assignments map[string]interface{}, unsupported *[]string) {
	for decisionName, rawSet := range assignments {
		setPath := path + "." + decisionName
		set := nestedStringMap(rawSet)
		collectUnsupportedFields(setPath, set, []string{"models", "fallback"}, unsupported)
		models, _ := set["models"].([]interface{})
		for modelIndex, rawModel := range models {
			modelPath := fmt.Sprintf("%s.models[%d]", setPath, modelIndex)
			model := nestedStringMap(rawModel)
			collectUnsupportedFields(
				modelPath, model,
				[]string{"model", "priority", "weight", "lora", "reasoning"}, unsupported,
			)
			collectUnsupportedFields(
				modelPath+".reasoning", nestedStringMap(model["reasoning"]),
				[]string{"enabled", "effort", "description"}, unsupported,
			)
		}
		collectUnsupportedFields(
			setPath+".fallback", nestedStringMap(set["fallback"]), []string{"strategy", "on"}, unsupported,
		)
	}
}

func collectUnsupportedFields(path string, value map[string]interface{}, allowed []string, unsupported *[]string) {
	if len(value) == 0 {
		return
	}
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, field := range allowed {
		allowedSet[field] = struct{}{}
	}
	for field := range value {
		if _, found := allowedSet[field]; !found {
			*unsupported = append(*unsupported, path+"."+field)
		}
	}
}
