package config

import (
	"fmt"
	"strings"
)

var decisionPluginPayloadFactories = map[string]func() interface{}{
	DecisionPluginSemanticCache:     func() interface{} { return &SemanticCachePluginConfig{} },
	DecisionPluginSystemPrompt:      func() interface{} { return &SystemPromptPluginConfig{} },
	DecisionPluginHeaderMutation:    func() interface{} { return &HeaderMutationPluginConfig{} },
	DecisionPluginHallucination:     func() interface{} { return &HallucinationPluginConfig{} },
	DecisionPluginResponseJailbreak: func() interface{} { return &ResponseJailbreakPluginConfig{} },
	DecisionPluginRouterReplay:      func() interface{} { return &RouterReplayPluginConfig{} },
	DecisionPluginMemory:            func() interface{} { return &MemoryPluginConfig{} },
	DecisionPluginRAG:               func() interface{} { return &RAGPluginConfig{} },
	DecisionPluginImageGen:          func() interface{} { return &ImageGenPluginConfig{} },
	DecisionPluginFastResponse:      func() interface{} { return &FastResponsePluginConfig{} },
	DecisionPluginRequestParams:     func() interface{} { return &RequestParamsPluginConfig{} },
	DecisionPluginTools:             func() interface{} { return &ToolsPluginConfig{} },
	DecisionPluginToolSelection:     func() interface{} { return &ToolSelectionPluginConfig{} },
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
		normalizedType == DecisionPluginResponseJailbreak {
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
		if strings.TrimSpace(typed.Message) == "" {
			return fmt.Errorf(
				"decision %q plugins[%d] (%s): message is required",
				decisionName,
				index,
				pluginType,
			)
		}
	case *ResponseJailbreakPluginConfig:
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
		default:
			return fmt.Errorf(
				"decision %q plugins[%d] (%s): action must be header, block, or none",
				decisionName,
				index,
				pluginType,
			)
		}
	}
	return nil
}
