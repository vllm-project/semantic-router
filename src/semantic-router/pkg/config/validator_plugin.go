package config

import (
	"fmt"
	"strings"
)

var decisionPluginPayloadFactories = map[string]func() interface{}{
	DecisionPluginResponseCache:       func() interface{} { return &ResponseCachePluginConfig{} },
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
	if normalizedType == DecisionPluginResponseCache ||
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
	case *ResponseCachePluginConfig:
		return validateResponseCachePlugin(decisionName, index, pluginType, typed)
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

func validateResponseCachePlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *ResponseCachePluginConfig,
) error {
	scope := fmt.Sprintf("decision %q plugins[%d] (%s)", decisionName, index, pluginType)
	checks := []func() error{
		func() error { return validateResponseCacheCompatibilityFields(typed, scope) },
		func() error { return validateCacheMode(typed.Mode, scope) },
		func() error {
			return validateCacheThreshold(typed.EffectiveSimilarityThreshold(), scope)
		},
		func() error { return validateResponseCacheScopeAndTTL(typed, scope) },
		func() error { return validateResponseCacheControls(typed.EffectiveRequestControls(), scope) },
		func() error { return validateResponseCachePersonalization(typed.Personalized, scope) },
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

func validateResponseCacheCompatibilityFields(
	typed *ResponseCachePluginConfig,
	scope string,
) error {
	if typed.Semantic != nil && typed.SimilarityThreshold != nil {
		return fmt.Errorf("%s: semantic.similarity_threshold conflicts with deprecated similarity_threshold", scope)
	}
	if typed.RequestControls != nil &&
		(typed.AllowRequestControls || strings.TrimSpace(typed.ControlHeader) != "") {
		return fmt.Errorf("%s: request_controls conflicts with deprecated request-control fields", scope)
	}
	return nil
}

func validateResponseCacheScopeAndTTL(typed *ResponseCachePluginConfig, scope string) error {
	switch strings.TrimSpace(typed.Scope) {
	case "", "user", "team", "tenant", "global":
	default:
		return fmt.Errorf("%s: scope must be user, team, tenant, or global", scope)
	}
	if typed.TTLSeconds != nil && *typed.TTLSeconds < 0 {
		return fmt.Errorf("%s: ttl_seconds cannot be negative", scope)
	}
	return nil
}

func validateResponseCacheControls(
	controls ResponseCacheRequestControlsConfig,
	scope string,
) error {
	if controls.MaxTTLSeconds != nil && *controls.MaxTTLSeconds < 0 {
		return fmt.Errorf("%s: request_controls.max_ttl_seconds cannot be negative", scope)
	}
	for _, directive := range controls.Allowed {
		switch strings.TrimSpace(directive) {
		case "no-cache", "no-store", "bypass", "max-age", "ttl":
		default:
			return fmt.Errorf("%s: unsupported request control %q", scope, directive)
		}
	}
	return nil
}

func validateResponseCachePersonalization(
	personalized *ResponseCachePersonalizedConfig,
	scope string,
) error {
	if personalized == nil {
		return nil
	}
	switch strings.TrimSpace(personalized.Mode) {
	case "", "disabled", "exact":
		return nil
	default:
		return fmt.Errorf("%s: personalized.mode must be disabled or exact", scope)
	}
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
