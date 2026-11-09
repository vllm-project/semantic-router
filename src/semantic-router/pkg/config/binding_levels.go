package config

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// Which deployment answers a signal is settled at one of three levels:
//
//   - a rule that asks its own question writes deployment on the rule:
//     decision, classifiers, safety (and its hazard part), complexity and
//     the decision selector;
//   - a signal type whose rules are labels or thresholds over one shared
//     answer is bound by type: bindings.<type> globally, or
//     model_bindings.<type> in a recipe;
//   - everything else asks system.decision_model.
//
// Embeddings, reranking and the hazard head ask no question; they are named
// where they are used (embeddings.default, the rag plugin's
// rerank.deployment, a safety rule's hazard.deployment).

// SharedAnswerSignalTypes are the signal types a binding can choose the
// deployment of.
var SharedAnswerSignalTypes = []string{
	SignalTypeDomain, SignalTypeJailbreak, SignalTypePII, SignalTypeFactCheck, SignalTypeUserFeedback,
	SignalTypeModality, SignalTypeHallucination, SignalTypePreference, SignalTypeReask,
}

// ruleLevelDeployment is where a rule of a signal type that asks its own
// question writes its deployment.
var ruleLevelDeployment = map[string]string{
	SignalTypeClassifier: "routing.signals.classifiers[].deployment",
	SignalTypeSafety:     "routing.signals.safety[].deployment",
	bindingKeyHazard:     "routing.signals.safety[].hazard.deployment",
	SignalTypeDecision:   "routing.signals.decision[].deployment",
	SignalTypeComplexity: "routing.signals.complexity[].deployment",
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
		case key == "embedding" && path != "global.model_catalog.bindings":
			// A routing profile that embeds with its own model binds it
			// here; embeddings.default names everyone else's.
		case key == "embedding":
			note = "embedding asks no question; name its deployment in global.model_catalog.embeddings.default"
		case key == RAGRerankerConsumer:
			note = "rag.reranker is deprecated; name the deployment in the rag plugin's rerank.deployment"
		case key == SignalTypeComplexity:
			note = "complexity rules ask their own question; set deployment on each rule (routing.signals.complexity[].deployment)"
		case strings.HasPrefix(key, "classifier.") || strings.HasPrefix(key, "safety.") || strings.HasPrefix(key, "complexity."):
			note = fmt.Sprintf("a binding for one rule is deprecated; set deployment on the rule (%s)", ruleDeploymentPath(key))
		case strings.HasPrefix(key, "decision."):
			return nil, fmt.Errorf("%s.%s: decision rules each ask their own question; set deployment on the rule (routing.signals.decision[].deployment)", path, key)
		case ruleLevelDeployment[key] != "":
			fallback := "system.decision_model"
			if key == bindingKeyHazard {
				fallback = "the pre-declared hazard deployment"
			}
			return nil, fmt.Errorf("%s.%s: %s rules each ask their own question, so a binding cannot choose their deployment; set %s, or leave it out to use %s", path, key, key, ruleLevelDeployment[key], fallback)
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
	switch {
	case strings.HasPrefix(key, "safety.") && strings.HasSuffix(key, ".hazard"):
		return "routing.signals.safety[" + strings.TrimSuffix(strings.TrimPrefix(key, "safety."), ".hazard") + "].hazard.deployment"
	case strings.HasPrefix(key, "safety."):
		return "routing.signals.safety[" + strings.TrimPrefix(key, "safety.") + "].deployment"
	case strings.HasPrefix(key, "classifier."):
		return "routing.signals.classifiers[" + strings.TrimPrefix(key, "classifier.") + "].deployment"
	}
	return "routing.signals.complexity[" + strings.TrimPrefix(key, "complexity.") + "].deployment"
}

// suggestKey names the signal type, or the deprecated consumer name of
// earlier releases, closest to an unknown key.
func suggestKey(key string) string {
	best, bestDistance := "", 4
	candidates := append(slices.Clone(SharedAnswerSignalTypes), "embedding")
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

// sharedAnswerRulePath matches a rule of a signal type whose rules share one
// answer: routing.signals.<signals key>[<index>].
var sharedAnswerRulePath = regexp.MustCompile(`(^|\.)signals\.(domains|jailbreak|pii|fact_check|user_feedbacks|modality|hallucination|preferences|reasks)\[\d+\]$`)

var sharedAnswerSignalOfKey = map[string]string{
	"domains": SignalTypeDomain, "jailbreak": SignalTypeJailbreak, "pii": SignalTypePII, "fact_check": SignalTypeFactCheck,
	"user_feedbacks": SignalTypeUserFeedback, "modality": SignalTypeModality, "hallucination": SignalTypeHallucination,
	"preferences": SignalTypePreference, "reasks": SignalTypeReask,
}

// explainUnknownField says why a field a configuration may reasonably write
// is not accepted where it is written; empty when there is nothing to add.
func explainUnknownField(key, path string) string {
	if key == "deployment" {
		if match := sharedAnswerRulePath.FindStringSubmatch(path); match != nil {
			signal := sharedAnswerSignalOfKey[match[2]]
			return fmt.Sprintf("%s rules share one answer, so their deployment is chosen for the signal type: model_bindings.%s: <deployment> (or global.model_catalog.bindings.%s)", signal, signal, signal)
		}
	}
	if (key == "api_key" || key == "key" || key == "token") && (path == "auth" || strings.HasSuffix(path, ".auth")) {
		return "a key is never written in the configuration; set api_key_env or api_key_file"
	}
	return ""
}
