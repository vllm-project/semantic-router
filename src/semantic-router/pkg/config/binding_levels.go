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
	SignalTypeSystemOne: "routing.signals.systemone[].deployment",
}

// checkBindingKeys checks the keys one bindings map of a configuration
// writes and lists its deprecated keys and binding forms, one warning per
// binding.
func checkBindingKeys(path string, bindings map[string]ModelBinding) ([]ConfigWarning, error) {
	var warnings []ConfigWarning
	for _, key := range sortedModelKeys(bindings) {
		note := ""
		switch {
		case slices.Contains(SharedAnswerSignalTypes, key):
		case IsDeprecatedBindingKey(key):
			note = fmt.Sprintf("%s is deprecated; key the binding by its signal type: %s", key, BindingKey(key))
		case strings.HasPrefix(key, "safety.") || strings.HasPrefix(key, "complexity."):
			note = fmt.Sprintf("a binding for one rule is deprecated; set deployment on the rule (%s)", ruleDeploymentPath(key))
		case strings.HasPrefix(key, "systemone."):
			return nil, fmt.Errorf("%s.%s: systemone rules each ask their own question; set deployment on the rule (routing.signals.systemone[].deployment)", path, key)
		case ruleLevelDeployment[key] != "":
			return nil, fmt.Errorf("%s.%s: %s rules each ask their own question, so a binding cannot choose their deployment; set %s, or leave it out to use system.decision_model", path, key, key, ruleLevelDeployment[key])
		default:
			return nil, fmt.Errorf("%s.%s: unknown signal type%s; a binding is keyed by one of %s", path, key, suggestKey(key), strings.Join(SharedAnswerSignalTypes, ", "))
		}
		if fields := bindings[key].DeprecatedFields(); len(fields) > 0 {
			form := fmt.Sprintf("%s: the signal decides the contract and the provider the adapter", strings.Join(fields, ", "))
			if note == "" {
				note = fmt.Sprintf("%s; write the binding as %s: %s", form, key, bindings[key].Deployment)
			} else {
				note += "; and " + form
			}
		}
		if note != "" {
			warnings = append(warnings, ConfigWarning{Code: warningDeprecatedModelField, Field: path + "." + key, Message: note})
		}
	}
	return warnings, nil
}

// ruleDeploymentPath is where the rule a deprecated rule key names writes its
// deployment.
func ruleDeploymentPath(key string) string {
	if strings.HasPrefix(key, "safety.") {
		return "routing.signals.safety[" + strings.TrimPrefix(key, "safety.") + "].deployment"
	}
	return "routing.signals.complexity[" + strings.TrimPrefix(key, "complexity.") + "].deployment"
}

// suggestKey names the signal type, or the deprecated consumer name of
// earlier releases, closest to an unknown key.
func suggestKey(key string) string {
	best, bestDistance := "", 4
	candidates := slices.Clone(SharedAnswerSignalTypes)
	for consumer := range bindingKeyOfConsumer {
		candidates = append(candidates, consumer)
	}
	for _, candidate := range candidates {
		if distance := levenshtein(key, candidate); distance < bestDistance {
			best, bestDistance = candidate, distance
		}
	}
	if best == "" {
		return ""
	}
	return fmt.Sprintf(" (did you mean %s?)", BindingKey(best))
}

// checkCanonicalBindingKeys checks the bindings maps a document writes.
func checkCanonicalBindingKeys(canonical *CanonicalConfig) ([]ConfigWarning, error) {
	var global map[string]ModelBinding
	if canonical.Global != nil {
		global = canonical.Global.ModelCatalog.Bindings
	}
	warnings, err := checkBindingKeys("global.model_catalog.bindings", global)
	if err != nil {
		return nil, err
	}
	routing, err := checkBindingKeys("routing.model_bindings", canonical.Routing.ModelBindings)
	if err != nil {
		return nil, err
	}
	warnings = append(warnings, routing...)
	for _, recipe := range canonical.Recipes {
		scoped, err := checkBindingKeys(fmt.Sprintf("recipes[%s].routing.model_bindings", recipe.Name), recipe.Routing.ModelBindings)
		if err != nil {
			return nil, err
		}
		warnings = append(warnings, scoped...)
	}
	return warnings, nil
}

// explainUnknownField says why a field a configuration may reasonably write
// is not accepted where it is written; empty when there is nothing to add.
func explainUnknownField(key, path string) string {
	if (key == "api_key" || key == "key" || key == "token") && (path == "auth" || strings.HasSuffix(path, ".auth")) {
		return "a key is never written in the configuration; set api_key_env or api_key_file"
	}
	return ""
}
