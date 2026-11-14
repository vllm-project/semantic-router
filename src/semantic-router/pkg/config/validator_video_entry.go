package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// validateVideoEntryContracts keeps the route half of a client-visible video
// id unambiguous. The id splits at its first colon, so a videos route cannot
// itself contain one.
func validateVideoEntryContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for _, name := range sortedModelConfigNames(cfg.ModelConfig) {
		params := cfg.ModelConfig[name]
		declared, _ := llmprotocol.ModelCapabilities(params.Capabilities)
		if CanonicalAPIFormat(params.APIFormat) != APIFormatVideos && !declared.Supports(llmprotocol.CapabilityVideoGeneration) {
			continue
		}
		if strings.Contains(name, ":") {
			return CheckError(fmt.Sprintf("providers.models[%s]", name), CodeInvalidValue,
				"a videos route model name cannot contain ':' because the client video id splits there")
		}
	}
	return nil
}

func videoGenerationWarnings(cfg *RouterConfig) []ConfigWarning {
	if cfg == nil {
		return nil
	}
	var warnings []ConfigWarning
	for _, name := range sortedModelConfigNames(cfg.ModelConfig) {
		declared, _ := llmprotocol.ModelCapabilities(cfg.ModelConfig[name].Capabilities)
		if !declared.Supports(llmprotocol.CapabilityVideoGeneration) {
			continue
		}
		if endpoints := cfg.GetEndpointsForModel(name); len(endpoints) > 1 {
			warnings = append(warnings, ConfigWarning{
				Code: "video_generation_multiple_endpoints", Field: "providers.models[" + name + "]",
				Message: "video status and delete calls are pinned to the route, not one endpoint, and backend job state is per process",
			})
		}
	}
	return warnings
}

func sortedModelConfigNames(models map[string]ModelParams) []string {
	names := make([]string, 0, len(models))
	for name := range models {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}
