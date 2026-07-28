package config

import (
	"fmt"
	"strings"
)

type pluginConfigurationValidator func(*StructuredPayload, string) error

var decisionPluginConfigurationValidators = map[string]pluginConfigurationValidator{
	DecisionPluginSemanticCache:     validatePluginConfigurationAs[SemanticCachePluginConfig],
	DecisionPluginSystemPrompt:      validatePluginConfigurationAs[SystemPromptPluginConfig],
	DecisionPluginHeaderMutation:    validatePluginConfigurationAs[HeaderMutationPluginConfig],
	DecisionPluginHallucination:     validatePluginConfigurationAs[HallucinationPluginConfig],
	DecisionPluginResponseJailbreak: validatePluginConfigurationAs[ResponseJailbreakPluginConfig],
	DecisionPluginRouterReplay:      validatePluginConfigurationAs[RouterReplayPluginConfig],
	DecisionPluginMemory:            validatePluginConfigurationAs[MemoryPluginConfig],
	DecisionPluginRAG:               validateRAGPluginConfiguration,
	DecisionPluginImageGen:          validateImageGenPluginConfiguration,
	DecisionPluginFastResponse:      validatePluginConfigurationAs[FastResponsePluginConfig],
	DecisionPluginRequestParams:     validatePluginConfigurationAs[RequestParamsPluginConfig],
	DecisionPluginToolSelection:     validatePluginConfigurationAs[ToolSelectionPluginConfig],
	DecisionPluginTools:             validatePluginConfigurationAs[ToolsPluginConfig],
}

var ragBackendConfigurationValidators = map[string]pluginConfigurationValidator{
	"milvus":       validatePluginConfigurationAs[MilvusRAGConfig],
	"qdrant":       validatePluginConfigurationAs[QdrantRAGConfig],
	"external_api": validatePluginConfigurationAs[ExternalAPIRAGConfig],
	"mcp":          validatePluginConfigurationAs[MCPRAGConfig],
	"openai":       validatePluginConfigurationAs[OpenAIRAGConfig],
	"vectorstore":  validatePluginConfigurationAs[VectorStoreRAGConfig],
}

// ValidateDecisionPluginConfiguration validates a plugin payload against the
// schema owned by its type discriminator.
func ValidateDecisionPluginConfiguration(pluginType string, payload *StructuredPayload) error {
	return validateDecisionPluginConfiguration(pluginType, payload, "configuration")
}

func validateDecisionPluginConfiguration(pluginType string, payload *StructuredPayload, path string) error {
	normalizedType := NormalizeDecisionPluginType(strings.TrimSpace(pluginType))
	validator, ok := decisionPluginConfigurationValidators[normalizedType]
	if !ok {
		return fmt.Errorf(
			"%s.type: unsupported decision plugin %q; supported plugins: %s",
			strings.TrimSuffix(path, ".configuration"),
			pluginType,
			strings.Join(SupportedDecisionPluginTypes(), ", "),
		)
	}
	if payload == nil {
		return nil
	}
	return validator(payload, path)
}

func validatePluginConfigurationAs[T any](payload *StructuredPayload, path string) error {
	_, err := decodeStructuredPayloadStrict[T](payload, path)
	return err
}

func validateRAGPluginConfiguration(payload *StructuredPayload, path string) error {
	cfg, err := decodeStructuredPayloadStrict[RAGPluginConfig](payload, path)
	if err != nil {
		return err
	}
	return validateRAGBackendPayloadContract(cfg.Backend, cfg.BackendConfig, path+".backend_config")
}

func validateRAGBackendPayloadContract(backend string, payload *StructuredPayload, path string) error {
	if strings.TrimSpace(backend) == "" {
		if payload != nil {
			return fmt.Errorf("%s: backend is required to validate backend_config", path)
		}
		return nil
	}
	if payload == nil {
		return nil
	}

	if backend == "hybrid" {
		return validateHybridRAGPayloadContract(payload, path)
	}
	validator, ok := ragBackendConfigurationValidators[backend]
	if !ok {
		return fmt.Errorf("%s: unsupported RAG backend %q", path, backend)
	}
	return validator(payload, path)
}

func validateHybridRAGPayloadContract(payload *StructuredPayload, path string) error {
	hybrid, err := decodeStructuredPayloadStrict[HybridRAGConfig](payload, path)
	if err != nil {
		return err
	}
	if err := validateRAGBackendPayloadContract(
		hybrid.Primary,
		hybrid.PrimaryConfig,
		path+".primary_config",
	); err != nil {
		return err
	}
	if hybrid.Fallback == "" && hybrid.FallbackConfig == nil {
		return nil
	}
	return validateRAGBackendPayloadContract(
		hybrid.Fallback,
		hybrid.FallbackConfig,
		path+".fallback_config",
	)
}

func validateImageGenPluginConfiguration(payload *StructuredPayload, path string) error {
	cfg, err := decodeStructuredPayloadStrict[ImageGenPluginConfig](payload, path)
	if err != nil {
		return err
	}
	if strings.TrimSpace(cfg.Backend) == "" {
		if cfg.BackendConfig != nil {
			return fmt.Errorf("%s.backend_config: backend is required to validate backend_config", path)
		}
		return nil
	}
	if cfg.BackendConfig == nil {
		return nil
	}

	switch cfg.Backend {
	case "vllm_omni":
		_, err = decodeStructuredPayloadStrict[VLLMOmniImageGenConfig](cfg.BackendConfig, path+".backend_config")
	case "openai":
		_, err = decodeStructuredPayloadStrict[OpenAIImageGenConfig](cfg.BackendConfig, path+".backend_config")
	default:
		return fmt.Errorf("%s.backend_config: unsupported image_gen backend %q", path, cfg.Backend)
	}
	return err
}
