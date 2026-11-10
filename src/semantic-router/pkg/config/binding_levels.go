package config

import (
	"fmt"
	"slices"
	"strings"
)

// Which deployment answers a signal is settled, most specific first, by:
//
//   - the rule's deployment;
//   - for an application signal, its type's binding: model_bindings.<type>
//     in a recipe, else bindings.<type> globally;
//   - system.decision_model.
//
// A systemone rule asks its own question, so no binding chooses its
// deployment. Embeddings ask no question; they are named where they are
// used (embeddings.default).

// SharedAnswerSignalTypes are the signal types a binding can choose the
// deployment of: the application signals.
var SharedAnswerSignalTypes = []string{
	SignalTypeDomain, SignalTypeFactCheck, SignalTypeUserFeedback, SignalTypeJailbreak, SignalTypeSafety, SignalTypePII,
	SignalTypeHallucination, SignalTypeRefusal, SignalTypeTurnRelation, SignalTypeComplexity, SignalTypeModality,
}

// ruleLevelDeployment is where a rule of a signal type that asks its own
// question writes its deployment.
var ruleLevelDeployment = map[string]string{
	SignalTypeSystemOne: "recipes[].signals.systemone[].deployment",
}

// checkBindingKeys checks the keys one bindings map of a configuration
// writes: each names an application signal type.
func checkBindingKeys(path string, bindings map[string]ModelBinding) error {
	for _, key := range sortedModelKeys(bindings) {
		switch {
		case slices.Contains(SharedAnswerSignalTypes, key):
		case strings.HasPrefix(key, "systemone."):
			return fmt.Errorf("%s.%s: systemone rules each ask their own question; set deployment on the rule (recipes[].signals.systemone[].deployment)", path, key)
		case ruleLevelDeployment[key] != "":
			return fmt.Errorf("%s.%s: %s rules each ask their own question, so a binding cannot choose their deployment; set %s, or leave it out to use system.decision_model", path, key, key, ruleLevelDeployment[key])
		default:
			return fmt.Errorf("%s.%s: unknown signal type%s; a binding is keyed by one of %s", path, key, suggestKey(key), strings.Join(SharedAnswerSignalTypes, ", "))
		}
	}
	return nil
}

// suggestKey names the signal type closest to an unknown key.
func suggestKey(key string) string {
	best, bestDistance := "", 4
	for _, candidate := range SharedAnswerSignalTypes {
		if distance := levenshtein(key, candidate); distance < bestDistance {
			best, bestDistance = candidate, distance
		}
	}
	if best == "" {
		return ""
	}
	return fmt.Sprintf(" (did you mean %s?)", best)
}

// checkCanonicalBindingKeys checks the bindings maps a document writes.
func checkCanonicalBindingKeys(canonical *CanonicalConfig) error {
	var global map[string]ModelBinding
	if canonical.Global != nil {
		global = canonical.Global.ModelCatalog.Bindings
	}
	if err := checkBindingKeys("global.model_catalog.bindings", global); err != nil {
		return err
	}
	for _, recipe := range canonical.Recipes {
		if err := checkBindingKeys(recipePath(recipe.Name)+".model_bindings", recipe.ModelBindings); err != nil {
			return err
		}
	}
	return nil
}

// explainUnknownField says why a field a configuration may reasonably write
// is not accepted where it is written; empty when there is nothing to add.
func explainUnknownField(key, path string) string {
	if (key == "api_key" || key == "key" || key == "token") && (path == "auth" || strings.HasSuffix(path, ".auth")) {
		return "a key is never written in the configuration; set api_key_env or api_key_file"
	}
	return ""
}
