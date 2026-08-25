package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func validatePromptAlgorithmConfig(
	decisionName string,
	modelRefs []ModelRef,
	algorithm *AlgorithmConfig,
) error {
	if algorithm.Prompt == nil {
		return fmt.Errorf("decision '%s': algorithm.type=prompt requires algorithm.prompt configuration", decisionName)
	}
	// Recipe documents are deliberately model-free. Candidate cardinality is
	// checked again on the Entrypoint-derived view after assignments have been
	// compiled into ModelRefs.
	if len(modelRefs) == 1 {
		return fmt.Errorf("decision '%s': algorithm.type=prompt requires at least two modelRefs", decisionName)
	}
	seenModels := make(map[string]struct{}, len(modelRefs))
	seenEffectiveModels := make(map[string]struct{}, len(modelRefs))
	for _, modelRef := range modelRefs {
		if _, exists := seenModels[modelRef.Model]; exists {
			return fmt.Errorf(
				"decision '%s': algorithm.type=prompt requires unique modelRefs; duplicate model %q",
				decisionName,
				modelRef.Model,
			)
		}
		seenModels[modelRef.Model] = struct{}{}
		effectiveModel := modelRef.Model
		if modelRef.LoRAName != "" {
			effectiveModel = modelRef.LoRAName
		}
		if _, exists := seenEffectiveModels[effectiveModel]; exists {
			return fmt.Errorf(
				"decision '%s': algorithm.type=prompt requires unique effective model identities; duplicate %q",
				decisionName,
				effectiveModel,
			)
		}
		seenEffectiveModels[effectiveModel] = struct{}{}
	}
	if len(modelRefs) > 0 && strings.TrimSpace(algorithm.Prompt.Model) == "" {
		return fmt.Errorf("decision '%s', algorithm.prompt: model is required", decisionName)
	}
	if strings.TrimSpace(algorithm.Prompt.Instructions) == "" {
		return fmt.Errorf("decision '%s', algorithm.prompt: instructions are required", decisionName)
	}
	if algorithm.Prompt.TimeoutSeconds < 0 {
		return fmt.Errorf("decision '%s', algorithm.prompt: timeout_seconds cannot be negative", decisionName)
	}
	if algorithm.OnError != "" && algorithm.OnError != "fallback" {
		return fmt.Errorf("decision '%s': algorithm.type=prompt on_error must be fallback", decisionName)
	}
	return nil
}

func validateDecisionPromptModel(cfg *RouterConfig, decision Decision) error {
	if decision.Algorithm == nil || decision.Algorithm.Prompt == nil {
		return nil
	}
	if len(decision.ModelRefs) == 0 && strings.TrimSpace(decision.Algorithm.Prompt.Model) == "" {
		// Model selection is materialized by the Entrypoint. The derived view is
		// validated later in the same contract pass.
		return nil
	}
	if !cfg.Looper.IsEnabled() {
		return fmt.Errorf(
			"decision '%s': algorithm.type=prompt requires global.integrations.looper.endpoint",
			decision.Name,
		)
	}
	model := strings.TrimSpace(decision.Algorithm.Prompt.Model)
	modelConfig, ok := cfg.ModelConfig[model]
	if !ok {
		return fmt.Errorf(
			"decision '%s', algorithm.prompt.model %q is not declared in the compiled Model set",
			decision.Name,
			model,
		)
	}
	if !promptHelperModalitySupported(modelConfig.Modality) {
		return fmt.Errorf(
			"decision '%s', algorithm.prompt.model %q must be chat-capable text modality",
			decision.Name,
			model,
		)
	}
	if cfg.IsEntrypointModelName(model) {
		return fmt.Errorf(
			"decision '%s', algorithm.prompt.model %q must be a concrete provider model",
			decision.Name,
			model,
		)
	}
	if !promptHelperHasChatBackend(cfg, model) {
		return fmt.Errorf(
			"decision '%s', algorithm.prompt.model %q requires an OpenAI Chat wire-format backend",
			decision.Name,
			model,
		)
	}
	return nil
}

func promptHelperHasChatBackend(cfg *RouterConfig, model string) bool {
	if cfg.RoutingSnapshot != nil {
		if candidate, ok := promptHelperSnapshotModel(cfg, model); ok {
			return promptHelperModelSupportsChat(candidate)
		}
	}
	for _, endpoint := range cfg.VLLMEndpoints {
		if endpoint.Model == model || endpoint.Name == model {
			return true
		}
	}
	return false
}

func promptHelperSnapshotModel(cfg *RouterConfig, model string) (routingsnapshot.Model, bool) {
	for _, candidate := range cfg.RoutingSnapshot.Models {
		if candidate.Name == model || slices.Contains(candidate.Aliases, model) {
			return candidate, true
		}
	}
	return routingsnapshot.Model{}, false
}

func promptHelperModelSupportsChat(model routingsnapshot.Model) bool {
	for _, backend := range model.Backends {
		if backend.WireFormat == llmprotocol.OpenAIChatV1 {
			return true
		}
	}
	return false
}

func promptHelperModalitySupported(modality string) bool {
	switch strings.ToLower(strings.TrimSpace(modality)) {
	case "", "text", "ar", "both", "multimodal":
		return true
	default:
		return false
	}
}
