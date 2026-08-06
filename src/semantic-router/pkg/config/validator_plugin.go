package config

import (
	"fmt"
	"strings"
)

var decisionPluginPayloadFactories = map[string]func() interface{}{
	DecisionPluginSemanticCache:       func() interface{} { return &SemanticCachePluginConfig{} },
	DecisionPluginSystemPrompt:        func() interface{} { return &SystemPromptPluginConfig{} },
	DecisionPluginHeaderMutation:      func() interface{} { return &HeaderMutationPluginConfig{} },
	DecisionPluginHallucination:       func() interface{} { return &HallucinationPluginConfig{} },
	DecisionPluginResponseJailbreak:   func() interface{} { return &ResponseJailbreakPluginConfig{} },
	DecisionPluginRouterReplay:        func() interface{} { return &RouterReplayPluginConfig{} },
	DecisionPluginMemory:              func() interface{} { return &MemoryPluginConfig{} },
	DecisionPluginRAG:                 func() interface{} { return &RAGPluginConfig{} },
	DecisionPluginImageGen:            func() interface{} { return &ImageGenPluginConfig{} },
	DecisionPluginFastResponse:        func() interface{} { return &FastResponsePluginConfig{} },
	DecisionPluginRequestParams:       func() interface{} { return &RequestParamsPluginConfig{} },
	DecisionPluginTools:               func() interface{} { return &ToolsPluginConfig{} },
	DecisionPluginToolSelection:       func() interface{} { return &ToolSelectionPluginConfig{} },
	DecisionPluginProviderPromptCache: func() interface{} { return &ProviderPromptCachePluginConfig{} },
	DecisionPluginContextCompression:  func() interface{} { return &ContextCompressionPluginConfig{} },
}

func validateDecisionPluginPayload(
	decisionName string,
	index int,
	plugin DecisionPlugin,
) error {
	if !IsSupportedDecisionPluginType(plugin.Type) {
		return fmt.Errorf(
			"decision %q plugins[%d]: unsupported plugin type %q",
			decisionName,
			index,
			plugin.Type,
		)
	}
	if plugin.Configuration == nil {
		return fmt.Errorf(
			"decision %q plugins[%d] (%s): configuration is required",
			decisionName,
			index,
			plugin.Type,
		)
	}
	normalizedType := NormalizeDecisionPluginType(plugin.Type)
	factory := decisionPluginPayloadFactories[normalizedType]
	if factory == nil {
		return fmt.Errorf(
			"decision %q plugins[%d]: unsupported plugin type %q",
			decisionName,
			index,
			plugin.Type,
		)
	}
	target := factory()
	var err error
	if normalizedType == DecisionPluginSemanticCache ||
		normalizedType == DecisionPluginResponseJailbreak ||
		normalizedType == DecisionPluginProviderPromptCache ||
		normalizedType == DecisionPluginContextCompression {
		err = plugin.Configuration.DecodeIntoStrict(target)
	} else {
		err = plugin.Configuration.DecodeInto(target)
	}
	if err != nil {
		return fmt.Errorf(
			"decision %q plugins[%d] (%s): %w",
			decisionName,
			index,
			plugin.Type,
			err,
		)
	}
	return validateDecodedPluginContract(
		decisionName,
		index,
		plugin.Type,
		target,
	)
}

func validateDecodedPluginContract(
	decisionName string,
	index int,
	pluginType string,
	target interface{},
) error {
	switch typed := target.(type) {
	case *FastResponsePluginConfig:
		return validateFastResponsePlugin(decisionName, index, pluginType, typed)
	case *ResponseJailbreakPluginConfig:
		return validateResponseJailbreakPlugin(decisionName, index, pluginType, typed)
	case *ProviderPromptCachePluginConfig:
		return validateProviderPromptCachePlugin(decisionName, index, pluginType, typed)
	case *ContextCompressionPluginConfig:
		return validateContextCompressionPlugin(decisionName, index, pluginType, typed)
	}
	return nil
}

func validateFastResponsePlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *FastResponsePluginConfig,
) error {
	if strings.TrimSpace(typed.Message) != "" {
		return nil
	}
	return fmt.Errorf(
		"decision %q plugins[%d] (%s): message is required",
		decisionName,
		index,
		pluginType,
	)
}

func validateResponseJailbreakPlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *ResponseJailbreakPluginConfig,
) error {
	if typed.Threshold < 0 || typed.Threshold > 1 {
		return fmt.Errorf(
			"decision %q plugins[%d] (%s): threshold must be between 0 and 1",
			decisionName,
			index,
			pluginType,
		)
	}
	switch typed.Action {
	case "", "header", "block", "none":
		return nil
	default:
		return fmt.Errorf(
			"decision %q plugins[%d] (%s): action must be header, block, or none",
			decisionName,
			index,
			pluginType,
		)
	}
}

func validateProviderPromptCachePlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *ProviderPromptCachePluginConfig,
) error {
	switch typed.TTL {
	case "", "5m", "1h":
		return nil
	default:
		return fmt.Errorf(
			"decision %q plugins[%d] (%s): ttl must be 5m or 1h",
			decisionName,
			index,
			pluginType,
		)
	}
}

func validateContextCompressionPlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *ContextCompressionPluginConfig,
) error {
	if typed.MinTokens < 0 || typed.TargetTokens < 0 {
		return fmt.Errorf(
			"decision %q plugins[%d] (%s): token thresholds cannot be negative",
			decisionName,
			index,
			pluginType,
		)
	}
	minTokens := typed.MinTokens
	if minTokens == 0 {
		minTokens = 2000
	}
	targetTokens := typed.TargetTokens
	if targetTokens == 0 {
		targetTokens = minTokens / 2
	}
	if targetTokens >= minTokens {
		return fmt.Errorf(
			"decision %q plugins[%d] (%s): target_tokens must be less than min_tokens",
			decisionName,
			index,
			pluginType,
		)
	}
	return nil
}
