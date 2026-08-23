package config

import (
	"fmt"
	"sort"
	"strings"
)

func validateCanonicalAuthoringFields(raw map[string]interface{}) error {
	unsupported := make([]string, 0)
	models, _ := raw["models"].([]interface{})
	for modelIndex, rawModel := range models {
		path := fmt.Sprintf("models[%d]", modelIndex)
		model := nestedStringMap(rawModel)
		collectUnsupportedFields(path, model, []string{"name", "card", "connections", "runtime", "pricing"}, &unsupported)
		collectUnsupportedFields(path+".card", nestedStringMap(model["card"]), []string{
			"aliases", "param_size", "context_window_size", "description", "capabilities",
			"reasoning", "loras", "quality_score", "modality", "tags",
		}, &unsupported)
		connections, _ := model["connections"].([]interface{})
		for connectionIndex, rawConnection := range connections {
			collectUnsupportedFields(
				fmt.Sprintf("%s.connections[%d]", path, connectionIndex),
				nestedStringMap(rawConnection),
				[]string{"provider", "interface", "endpoint", "model", "credential", "weight"},
				&unsupported,
			)
		}
		collectUnsupportedFields(path+".runtime", nestedStringMap(model["runtime"]), []string{
			"max_retries", "request_timeout", "stream_timeout",
		}, &unsupported)
		collectUnsupportedFields(path+".pricing", nestedStringMap(model["pricing"]), []string{
			"input_cost_per_million_tokens", "output_cost_per_million_tokens",
			"cache_read_cost_per_million_tokens", "cache_write_cost_per_million_tokens",
		}, &unsupported)
	}

	recipes, _ := raw["recipes"].([]interface{})
	for recipeIndex, rawRecipe := range recipes {
		path := fmt.Sprintf("recipes[%d]", recipeIndex)
		recipe := nestedStringMap(rawRecipe)
		collectUnsupportedFields(path, recipe, []string{"name", "description", "document"}, &unsupported)
		document := nestedStringMap(recipe["document"])
		decisions, _ := document["decisions"].([]interface{})
		for decisionIndex, rawDecision := range decisions {
			decision := nestedStringMap(rawDecision)
			if _, found := decision["id"]; found {
				unsupported = append(unsupported, fmt.Sprintf("%s.document.decisions[%d].id", path, decisionIndex))
			}
		}
	}

	entrypoints, _ := raw["entrypoints"].([]interface{})
	for entrypointIndex, rawEntrypoint := range entrypoints {
		path := fmt.Sprintf("entrypoints[%d]", entrypointIndex)
		entrypoint := nestedStringMap(rawEntrypoint)
		collectUnsupportedFields(path, entrypoint, []string{
			"name", "aliases", "recipe", "assignments", "rules",
		}, &unsupported)
		validateAuthoringAssignments(path+".assignments", nestedStringMap(entrypoint["assignments"]), &unsupported)
		rules, _ := entrypoint["rules"].([]interface{})
		for ruleIndex, rawRule := range rules {
			rulePath := fmt.Sprintf("%s.rules[%d]", path, ruleIndex)
			rule := nestedStringMap(rawRule)
			collectUnsupportedFields(rulePath, rule, []string{"name", "matches", "recipe", "assignments"}, &unsupported)
			validateAuthoringAssignments(rulePath+".assignments", nestedStringMap(rule["assignments"]), &unsupported)
		}
	}

	if len(unsupported) == 0 {
		return nil
	}
	sort.Strings(unsupported)
	return fmt.Errorf("unsupported v0.4 authoring fields: %s", strings.Join(unsupported, ", "))
}

func validateAuthoringAssignments(path string, assignments map[string]interface{}, unsupported *[]string) {
	for decisionName, rawSet := range assignments {
		setPath := path + "." + decisionName
		set := nestedStringMap(rawSet)
		collectUnsupportedFields(setPath, set, []string{"models", "fallback"}, unsupported)
		models, _ := set["models"].([]interface{})
		for modelIndex, rawModel := range models {
			collectUnsupportedFields(
				fmt.Sprintf("%s.models[%d]", setPath, modelIndex), nestedStringMap(rawModel),
				[]string{"model", "priority", "weight", "lora", "reasoning"}, unsupported,
			)
		}
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
