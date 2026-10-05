package config

import (
	"fmt"
	"sort"
	"strings"
)

// The built-in model runtime serves every local model: it detects a package's
// format and owns numerics and kernels. The NLI explainer, and with it the
// response cache's NLI polarity tier, is retired; the lexical polarity guard
// always runs. The fields below selected those in the router;
// `vllm-sr config migrate` rewrites them.
var (
	removedModelProviders    = map[string]bool{"candle": true, "ort": true, "openvino": true}
	removedEmbeddingBackends = map[string]bool{"candle": true, "openvino": true}
	// EmbeddingGemma and MiniLM have no runtime family; Vela Embedding
	// (mmbert) replaces them.
	retiredEmbeddingTypes = map[string]bool{"gemma": true, "bert": true}
	removedEmbeddingPaths = []string{"gemma_model_path", "bert_model_path"}
	// A remote hallucination detector is a hallucination_detector binding;
	// the `backend: endpoint` shorthand was desugared into one.
	removedDetectorBackends = map[string]bool{"candle": true, "endpoint": true}
	removedDeploymentFields = []string{"precision", "custom_ops_profile", "compilation_cache_dir"}
	// Module paths under global.model_catalog.modules and their removed fields.
	removedModuleFields = []struct {
		path   []string
		fields []string
	}{
		{[]string{"prompt_guard"}, []string{"variant", "model_type", "use_modernbert", "use_mmbert_32k"}},
		{[]string{"classifier", "domain"}, []string{"variant", "use_modernbert", "use_mmbert_32k"}},
		{[]string{"classifier", "pii"}, []string{"use_modernbert", "use_mmbert_32k"}},
		{[]string{"hallucination_mitigation", "fact_check"}, []string{"use_modernbert", "use_mmbert_32k"}},
		{[]string{"feedback_detector"}, []string{"use_modernbert", "use_mmbert_32k"}},
		{[]string{"hallucination_mitigation", "detector"}, []string{"enable_nli_filtering", "nli_entailment_threshold"}},
		{[]string{"hallucination_mitigation"}, []string{"explainer", "nli_model"}},
	}
)

func rejectRemovedModelExecutionFields(raw map[string]interface{}) error {
	var removed []string
	catalog := nestedStringMap(nestedStringMap(raw["global"])["model_catalog"])
	deployments := nestedStringMap(catalog["deployments"])
	for _, name := range sortedMapKeys(deployments) {
		deployment := nestedStringMap(deployments[name])
		prefix := "global.model_catalog.deployments." + name
		if provider, _ := deployment["provider"].(string); removedModelProviders[strings.ToLower(strings.TrimSpace(provider))] {
			removed = append(removed, fmt.Sprintf("%s.provider: %s", prefix, provider))
		}
		removed = appendPresent(removed, prefix, deployment, removedDeploymentFields...)
	}
	modules := nestedStringMap(catalog["modules"])
	for _, module := range removedModuleFields {
		block := modules
		for _, key := range module.path {
			block = nestedStringMap(block[key])
		}
		removed = appendPresent(removed, "global.model_catalog.modules."+strings.Join(module.path, "."), block, module.fields...)
	}
	detector := nestedStringMap(nestedStringMap(modules["hallucination_mitigation"])["detector"])
	if backend, _ := detector["backend"].(string); removedDetectorBackends[strings.ToLower(strings.TrimSpace(backend))] {
		removed = append(removed, "global.model_catalog.modules.hallucination_mitigation.detector.backend: "+backend)
	}
	removed = appendPresent(removed, "global.model_catalog.modules.hallucination_mitigation.detector", detector, "endpoint")
	semantic := nestedStringMap(nestedStringMap(catalog["embeddings"])["semantic"])
	embeddingConfig := nestedStringMap(semantic["embedding_config"])
	if backend, _ := embeddingConfig["backend"].(string); removedEmbeddingBackends[strings.ToLower(strings.TrimSpace(backend))] {
		removed = append(removed, "global.model_catalog.embeddings.semantic.embedding_config.backend: "+backend)
	}
	removed = appendPresent(removed, "global.model_catalog.embeddings.semantic", semantic, removedEmbeddingPaths...)
	removed = appendRetiredEmbeddingType(removed, "global.model_catalog.embeddings.semantic.embedding_config.model_type", embeddingConfig["model_type"])
	global := nestedStringMap(raw["global"])
	stores := nestedStringMap(global["stores"])
	for _, name := range sortedMapKeys(stores) {
		removed = appendRetiredEmbeddingType(removed, "global.stores."+name+".embedding_model", nestedStringMap(stores[name])["embedding_model"])
	}
	selection := nestedStringMap(nestedStringMap(nestedStringMap(global["router"])["model_selection"])["ml"])
	removed = appendRetiredEmbeddingType(removed, "global.router.model_selection.ml.model_type", selection["model_type"])
	removed = appendPresent(removed, "global.model_catalog.system", nestedStringMap(catalog["system"]), "hallucination_explainer")
	removed = appendPresent(removed, "global.stores.response_cache", nestedStringMap(stores["response_cache"]), "polarity_guard")
	removed = append(removed, removedNLIRoutingFields("routing", nestedStringMap(raw["routing"]))...)
	if recipes, ok := raw["recipes"].([]interface{}); ok {
		for index, recipe := range recipes {
			removed = append(removed, removedNLIRoutingFields(fmt.Sprintf("recipes[%d].routing", index), nestedStringMap(nestedStringMap(recipe)["routing"]))...)
		}
	}
	if len(removed) == 0 {
		return nil
	}
	return fmt.Errorf(
		"removed model execution fields are no longer supported: %s; local models are served by the built-in model runtime (provider: model_runtime), which detects the package format and owns numerics; a remote hallucination detector is a hallucination_detector binding to an http deployment; the NLI explainer is retired; and Vela Embedding (mmbert) replaces the gemma and bert embedding models; run `vllm-sr config migrate --config old-config.yaml`",
		strings.Join(removed, ", "),
	)
}

// appendRetiredEmbeddingType records a field that selects a retired embedding model.
func appendRetiredEmbeddingType(removed []string, field string, value interface{}) []string {
	if model, _ := value.(string); retiredEmbeddingTypes[strings.ToLower(strings.TrimSpace(model))] {
		return append(removed, field+": "+model)
	}
	return removed
}

// removedNLIRoutingFields finds use_nli on hallucination rules and on
// hallucination plugins of one routing block, and the NLI name of the fusion
// grounding penalty (now contradiction_penalty).
func removedNLIRoutingFields(prefix string, routing map[string]interface{}) []string {
	var removed []string
	if rules, ok := nestedStringMap(routing["signals"])["hallucination"].([]interface{}); ok {
		for index, rule := range rules {
			removed = appendPresent(removed, fmt.Sprintf("%s.signals.hallucination[%d]", prefix, index), nestedStringMap(rule), "use_nli")
		}
	}
	decisions, _ := routing["decisions"].([]interface{})
	for index, decision := range decisions {
		grounding := nestedStringMap(nestedStringMap(nestedStringMap(nestedStringMap(decision)["algorithm"])["fusion"])["grounding"])
		removed = appendPresent(removed, fmt.Sprintf("%s.decisions[%d].algorithm.fusion.grounding", prefix, index), grounding, "nli_contradiction_penalty")
		plugins, _ := nestedStringMap(decision)["plugins"].([]interface{})
		for pluginIndex, plugin := range plugins {
			fields := nestedStringMap(plugin)
			if kind, _ := fields["type"].(string); kind == "hallucination" {
				removed = appendPresent(removed, fmt.Sprintf("%s.decisions[%d].plugins[%d].configuration", prefix, index, pluginIndex), nestedStringMap(fields["configuration"]), "use_nli")
			}
		}
	}
	return removed
}

func appendPresent(removed []string, prefix string, block map[string]interface{}, fields ...string) []string {
	for _, field := range fields {
		if _, ok := block[field]; ok {
			removed = append(removed, prefix+"."+field)
		}
	}
	return removed
}

func sortedMapKeys(values map[string]interface{}) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
