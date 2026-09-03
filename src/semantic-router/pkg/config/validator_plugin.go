package config

import (
	"fmt"
	"strings"
)

var decisionPluginPayloadFactories = map[string]func() interface{}{
	DecisionPluginResponseCache:      func() interface{} { return &ResponseCachePluginConfig{} },
	DecisionPluginSystemPrompt:       func() interface{} { return &SystemPromptPluginConfig{} },
	DecisionPluginHeaderMutation:     func() interface{} { return &HeaderMutationPluginConfig{} },
	DecisionPluginHallucination:      func() interface{} { return &HallucinationPluginConfig{} },
	DecisionPluginResponseJailbreak:  func() interface{} { return &ResponseJailbreakPluginConfig{} },
	DecisionPluginRouterReplay:       func() interface{} { return &RouterReplayPluginConfig{} },
	DecisionPluginMemory:             func() interface{} { return &MemoryPluginConfig{} },
	DecisionPluginRAG:                func() interface{} { return &RAGPluginConfig{} },
	DecisionPluginFastResponse:       func() interface{} { return &FastResponsePluginConfig{} },
	DecisionPluginRequestParams:      func() interface{} { return &RequestParamsPluginConfig{} },
	DecisionPluginTools:              func() interface{} { return &ToolsPluginConfig{} },
	DecisionPluginToolSelection:      func() interface{} { return &ToolSelectionPluginConfig{} },
	DecisionPluginContextCompression: func() interface{} { return &ContextCompressionPluginConfig{} },
}

func validateDecisionPluginPayload(
	decisionName string,
	index int,
	plugin DecisionPlugin,
) error {
	// image_gen was removed when #3076 unified inference protocol translation:
	// the router no longer executes image-generation backends. Route image
	// generation through a vllm-omni modality route speaking the Responses-API
	// hosted image_generation tool instead. See issue #3129.
	if plugin.Type == "image_gen" {
		return fmt.Errorf(
			"decision %q plugins[%d]: plugin %q is unsupported: the image_gen route plugin was removed; use the Responses-API hosted image_generation tool with a vllm-omni modality route",
			decisionName,
			index,
			plugin.Type,
		)
	}
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

func validateContextCompressionPlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *ContextCompressionPluginConfig,
) error {
	scope := fmt.Sprintf("decision %q plugins[%d] (%s)", decisionName, index, pluginType)
	checks := []func() error{
		func() error { return validateContextCompressionMode(typed, scope) },
		func() error { return validateContextCompressionBudget(typed, scope) },
		func() error { return validateContextCompressionTargets(typed, scope) },
		func() error { return validateContextCompressionScoring(typed, scope) },
		func() error { return validateContextCompressionRecovery(typed, scope) },
		func() error { return validateContextCompressionControls(typed, scope) },
	}
	for _, check := range checks {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

func ValidateContextCompressionPluginConfig(
	typed *ContextCompressionPluginConfig,
) error {
	if typed == nil {
		return fmt.Errorf("context_compression configuration is required")
	}
	return validateContextCompressionPlugin(
		"preview",
		0,
		DecisionPluginContextCompression,
		typed,
	)
}

func validateContextCompressionBudget(
	typed *ContextCompressionPluginConfig,
	scope string,
) error {
	if typed.Budget == nil {
		return nil
	}
	trigger := typed.Budget.TriggerTokens
	target := typed.Budget.TargetTokens
	if trigger != nil && target != nil &&
		!trigger.Auto && !target.Auto &&
		trigger.Value > 0 && target.Value >= trigger.Value {
		return fmt.Errorf("%s: budget.target_tokens must be less than budget.trigger_tokens", scope)
	}
	return nil
}

func validateContextCompressionMode(
	typed *ContextCompressionPluginConfig,
	scope string,
) error {
	switch typed.EffectiveMode() {
	case ContextCompressionModeAuto, ContextCompressionModeAlways:
	default:
		return fmt.Errorf("%s: mode must be auto or always", scope)
	}
	switch typed.EffectiveFailureMode() {
	case ContextCompressionFailureOpen, ContextCompressionFailureClosed:
		return nil
	default:
		return fmt.Errorf("%s: failure_mode must be fail_open or fail_closed", scope)
	}
}

func validateContextCompressionTargets(
	typed *ContextCompressionPluginConfig,
	scope string,
) error {
	targets := []ContextCompressionTargetConfig{typed.EffectiveToolOutputTarget()}
	if typed.Targets != nil {
		targets = append(targets, typed.Targets.History, typed.Targets.RAG, typed.Targets.Memory)
	}
	for _, target := range targets {
		switch typed.EffectiveTargetMode(target) {
		case ContextCompressionTargetPreserve,
			ContextCompressionTargetExtractive,
			ContextCompressionTargetRecoverable:
		default:
			return fmt.Errorf("%s: target mode must be preserve, extractive, or recoverable", scope)
		}
		if target.MinTokens < 0 || target.TargetTokens < 0 {
			return fmt.Errorf("%s: target token thresholds cannot be negative", scope)
		}
		if target.MinTokens > 0 &&
			target.TargetTokens > 0 &&
			target.TargetTokens >= target.MinTokens {
			return fmt.Errorf("%s: target_tokens must be less than min_tokens", scope)
		}
	}
	return nil
}

func validateContextCompressionScoring(
	typed *ContextCompressionPluginConfig,
	scope string,
) error {
	scoring := typed.EffectiveScoring()
	switch scoring.Method {
	case ContextCompressionScoringBM25:
		return nil
	case ContextCompressionScoringEmbedding, ContextCompressionScoringHybrid:
		if strings.TrimSpace(scoring.EmbeddingModelRef) == "" {
			return fmt.Errorf("%s: scoring.embedding_model_ref is required for %s", scope, scoring.Method)
		}
		return nil
	default:
		return fmt.Errorf("%s: scoring.method must be bm25, embedding, or hybrid", scope)
	}
}

func validateContextCompressionRecovery(
	typed *ContextCompressionPluginConfig,
	scope string,
) error {
	if typed.Recovery == nil || !typed.Recovery.Enabled {
		return nil
	}
	switch strings.TrimSpace(typed.Recovery.Store) {
	case "redis", "valkey", "response_cache":
	default:
		return fmt.Errorf("%s: recovery.store must be redis, valkey, or response_cache", scope)
	}
	if typed.Recovery.TTLSeconds < 0 ||
		typed.Recovery.MaxBytesPerRequest < 0 ||
		typed.Recovery.MaxTotalBytes < 0 ||
		typed.Recovery.MaxRetrievals < 0 {
		return fmt.Errorf("%s: recovery limits cannot be negative", scope)
	}
	return nil
}

func validateContextCompressionControls(
	typed *ContextCompressionPluginConfig,
	scope string,
) error {
	if typed.RequestControls == nil {
		return nil
	}
	if typed.RequestControls.MaxTargetTokens < 0 {
		return fmt.Errorf("%s: request_controls.max_target_tokens cannot be negative", scope)
	}
	for _, directive := range typed.RequestControls.Allowed {
		switch strings.TrimSpace(directive) {
		case "bypass", "target":
		default:
			return fmt.Errorf("%s: unsupported request control %q", scope, directive)
		}
	}
	return nil
}
