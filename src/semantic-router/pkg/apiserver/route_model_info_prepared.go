//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type preparedBindingInventory interface {
	PreparedBindings() ([]binding.PreparedBinding, bool)
}

func preparedModelsInfo(service classificationService) ([]ModelInfo, bool) {
	return preparedModelsInfoMatching(service, nil)
}

func preparedEmbeddingModelsInfo(service classificationService) ([]ModelInfo, bool) {
	return preparedModelsInfoMatching(service, func(capability binding.Capability) bool {
		return capability.Embedding != nil
	})
}

func preparedModelsInfoMatching(service classificationService, include func(binding.Capability) bool) ([]ModelInfo, bool) {
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
		if include != nil && !include(capability) {
			continue
		}
		name, modelType := preparedModelNameAndType(id)
		model := ModelInfo{
			Name: name, Recipe: id.Recipe, Type: modelType, Loaded: true,
			ModelPath: entry.Artifact,
			Metadata: map[string]string{
				"binding": id.Name, "deployment": id.Deployment,
				"resource_id": entry.ResourceID,
				"contract":    id.Contract, "model_type": id.Adapter,
				"provider": capability.Provider, "device": capability.Device,
				"precision":           capability.Precision,
				"max_sequence_length": fmt.Sprint(capability.Limits.EffectiveTokens()),
				"overflow":            capability.Limits.Overflow,
			},
		}
		if capability.Device == "external" {
			model.Metadata["lifecycle"] = "external"
		}
		addPreparedInputLimits(model.Metadata, capability)
		if capability.Embedding != nil {
			embedding := capability.Embedding
			model.Metadata["default_dimension"] = fmt.Sprint(embedding.Dimension)
			model.Metadata["default_layer"] = fmt.Sprint(embedding.Layer)
			model.Metadata["pooling"] = embedding.Pooling
			model.Metadata["normalization"] = embedding.Normalization
			model.Metadata["modalities"] = strings.Join(embedding.Modalities, ",")
		}
		model = enrichModelInfo(model, nil)
		// Native resource revisions retain the operator's source revision before
		// the actual artifact fingerprint. Use that declaration, never a guessed
		// name from a derived CK/export/cache directory.
		if sourceRevision, fingerprint, ok := strings.Cut(entry.Revision, ":"); ok && fingerprint != "" && capability.Device != "external" {
			if sourceRevision != "" {
				model.Metadata["source_revision"] = sourceRevision
			}
			if model.Registry == nil {
				model.Registry = config.GetModelRegistryInfoByRevision(sourceRevision)
				if model.Registry != nil {
					model.Metadata["registry_match"] = "source_revision"
				}
			}
		}
		models = append(models, model)
	}
	return models, true
}

// Keep the historical max_sequence_length field compatible, while exposing
// physical forward capacity separately from whole-document admission budgets.
func addPreparedInputLimits(metadata map[string]string, capability binding.Capability) {
	if limit := capability.Limits.ForwardTokens(); limit > 0 {
		metadata["forward_max_tokens"] = fmt.Sprint(limit)
	}
	if limit := capability.Limits.EffectiveTokens(); limit > 0 {
		metadata["input_max_tokens"] = fmt.Sprint(limit)
		if capability.Limits.Overflow == "window" {
			metadata["document_max_tokens"] = fmt.Sprint(limit)
		}
	}
	if capability.Window != nil {
		metadata["window_size"] = fmt.Sprint(capability.Window.Size)
		metadata["window_overlap"] = fmt.Sprint(capability.Window.Overlap)
	}
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
