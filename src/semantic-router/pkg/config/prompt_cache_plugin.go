package config

import "fmt"

const (
	PromptCacheTargetInstructions = "instructions"
	PromptCacheTargetTools        = "tools"

	PromptCacheUnsupportedSkip   = "skip"
	PromptCacheUnsupportedReject = "reject"

	promptCacheDefaultTTL = "5m"
)

// PromptCachePluginConfig controls route-local prompt-cache marker injection.
type PromptCachePluginConfig struct {
	Enabled       bool     `json:"enabled" yaml:"enabled"`
	TTL           string   `json:"ttl,omitempty" yaml:"ttl,omitempty"`
	Targets       []string `json:"targets,omitempty" yaml:"targets,omitempty"`
	OnUnsupported string   `json:"on_unsupported,omitempty" yaml:"on_unsupported,omitempty"`
}

func (c *PromptCachePluginConfig) EffectiveTTL() string {
	return c.withDefaults().TTL
}

func (c *PromptCachePluginConfig) EffectiveTargets() []string {
	return c.withDefaults().Targets
}

func (c *PromptCachePluginConfig) EffectiveOnUnsupported() string {
	return c.withDefaults().OnUnsupported
}

// GetPromptCacheConfig returns the route-local prompt-cache configuration.
func (d *Decision) GetPromptCacheConfig() *PromptCachePluginConfig {
	result := &PromptCachePluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginPromptCache, result)
}

func (c *PromptCachePluginConfig) withDefaults() PromptCachePluginConfig {
	if c == nil {
		return PromptCachePluginConfig{}
	}
	result := *c
	if result.TTL == "" {
		result.TTL = promptCacheDefaultTTL
	}
	if result.Targets == nil {
		result.Targets = []string{
			PromptCacheTargetInstructions,
			PromptCacheTargetTools,
		}
	} else {
		result.Targets = append([]string(nil), result.Targets...)
	}
	if result.OnUnsupported == "" {
		result.OnUnsupported = PromptCacheUnsupportedSkip
	}
	return result
}

func validatePromptCachePlugin(
	decisionName string,
	index int,
	pluginType string,
	config *PromptCachePluginConfig,
) error {
	scope := fmt.Sprintf("decision %q plugins[%d] (%s)", decisionName, index, pluginType)
	effective := config.withDefaults()
	switch effective.TTL {
	case "5m", "1h":
	default:
		return fmt.Errorf("%s: ttl must be 5m or 1h", scope)
	}
	switch effective.OnUnsupported {
	case PromptCacheUnsupportedSkip, PromptCacheUnsupportedReject:
	default:
		return fmt.Errorf("%s: on_unsupported must be skip or reject", scope)
	}
	if len(effective.Targets) == 0 {
		return fmt.Errorf("%s: targets must not be empty", scope)
	}
	seen := make(map[string]struct{}, len(effective.Targets))
	for _, target := range effective.Targets {
		switch target {
		case PromptCacheTargetInstructions, PromptCacheTargetTools:
		default:
			return fmt.Errorf("%s: targets must contain only instructions or tools", scope)
		}
		if _, ok := seen[target]; ok {
			return fmt.Errorf("%s: targets must not contain duplicates", scope)
		}
		seen[target] = struct{}{}
	}
	return nil
}

// ValidatePromptCachePluginConfig validates the public prompt-cache contract.
func ValidatePromptCachePluginConfig(pluginConfig *PromptCachePluginConfig) error {
	if pluginConfig == nil {
		return fmt.Errorf("prompt_cache configuration is required")
	}
	return validatePromptCachePlugin(
		"preview",
		0,
		DecisionPluginPromptCache,
		pluginConfig,
	)
}
