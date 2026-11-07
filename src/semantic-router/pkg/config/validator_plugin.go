package config

import (
	"fmt"
	"strings"
)

// DecisionPluginSchemaSamples returns one fresh typed payload for every
// supported plugin. The config-schema generator consumes this exact registry,
// so validation and every schema consumer cannot acquire different plugin
// inventories or payload shapes.
func DecisionPluginSchemaSamples() map[string]interface{} {
	entries := DecisionPlugins.Entries()
	samples := make(map[string]interface{}, len(entries))
	for _, entry := range entries {
		samples[entry.Type] = entry.Spec.NewPayload()
	}
	return samples
}

func validateDecisionPluginPayload(
	decisionName string,
	index int,
	plugin DecisionPlugin,
) error {
	_, err := DecodeDecisionPluginAt(PluginAt{Decision: decisionName, Index: index, Type: plugin.Type}, plugin)
	return err
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
