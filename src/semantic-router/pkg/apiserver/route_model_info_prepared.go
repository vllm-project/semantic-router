//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type preparedBindingInventory interface {
	PreparedBindings() ([]binding.PreparedBinding, bool)
}

func preparedModelsInfo(service classificationService) ([]ModelInfo, bool) {
	inventory, ok := service.(preparedBindingInventory)
	if !ok {
		return nil, false
	}
	entries, available := inventory.PreparedBindings()
	if !available {
		return nil, false
	}
	models := make([]ModelInfo, 0, len(entries))
	for _, entry := range entries {
		id, capability := entry.Identity, entry.Capability
		name, modelType := preparedModelNameAndType(id)
		model := ModelInfo{
			Name: name, Recipe: id.Recipe, Type: modelType, Loaded: true,
			ModelPath: entry.Artifact,
			Metadata: map[string]string{
				"binding": id.Name, "deployment": id.Deployment,
				"contract": id.Contract, "model_type": id.Adapter,
				"provider": capability.Provider, "device": capability.Device,
				"precision":           capability.Precision,
				"max_sequence_length": fmt.Sprint(capability.Limits.EffectiveTokens()),
				"overflow":            capability.Limits.Overflow,
			},
		}
		if capability.Device == "external" {
			model.Metadata["lifecycle"] = "external"
		}
		if capability.Embedding != nil {
			embedding := capability.Embedding
			model.Metadata["default_dimension"] = fmt.Sprint(embedding.Dimension)
			model.Metadata["pooling"] = embedding.Pooling
			model.Metadata["normalization"] = embedding.Normalization
			model.Metadata["modalities"] = strings.Join(embedding.Modalities, ",")
		}
		models = append(models, enrichModelInfo(model, nil))
	}
	return models, true
}

// Keep the established API names for existing consumers; new typed tasks are
// visible automatically, without adding another list of model families.
func preparedModelNameAndType(id binding.Identity) (string, string) {
	switch id.Name {
	case "domain_classifier":
		return "category_classifier", "intent_classification"
	case "prompt_guard":
		return "jailbreak_classifier", "security_detection"
	case "pii_classifier":
		return id.Name, "pii_detection"
	case "fact_check_classifier":
		return id.Name, "fact_check_classification"
	case "hallucination_detector":
		return id.Name, "hallucination_detection"
	case "hallucination_explainer":
		return id.Name, "nli_explainer"
	case "feedback_detector":
		return id.Name, "feedback_detection"
	}
	if id.Contract == "embedding.v1" {
		return id.Adapter + "_embedding_model", "embedding"
	}
	return id.Name, id.Contract
}

func modelsUseGPU(models []ModelInfo) bool {
	for _, model := range models {
		if !model.Loaded {
			continue
		}
		device, _, _ := strings.Cut(model.Metadata["device"], ":")
		switch device {
		case "cuda", "rocm", "migraphx", "metal":
			return true
		}
	}
	return false
}
