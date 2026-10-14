package config

import "strings"

// PreferenceUsesPrototypes preserves authored example banks and their cosine
// threshold/margin scale. An explicit decision task binding selects probability
// scoring instead, including when the shared module requests contrastive mode.
func (c *RouterConfig) PreferenceUsesPrototypes() bool {
	if c == nil || c.EffectiveModelBindings(c.Signals, c.ModelBindings)["preference"].Contract == DecisionTaskContract {
		return false
	}
	if c.PreferenceModel.UseContrastive != nil {
		return *c.PreferenceModel.UseContrastive
	}
	if c.FindExternalModelByRole(ModelRolePreference) != nil {
		return false
	}
	if strings.TrimSpace(c.PreferenceModel.EmbeddingModel) != "" {
		return true
	}
	for _, rule := range c.PreferenceRules {
		if len(rule.Examples) > 0 {
			return true
		}
	}
	return false
}

// PreferenceUsesDecisionTask keeps preparation, catalog admission and runtime
// selection aligned. Description-only rules use native judgment by default;
// explicit external classifiers retain their own execution contract.
func (c *RouterConfig) PreferenceUsesDecisionTask() bool {
	if c == nil {
		return false
	}
	if c.EffectiveModelBindings(c.Signals, c.ModelBindings)["preference"].Contract == DecisionTaskContract {
		return true
	}
	return !c.PreferenceUsesPrototypes() && c.FindExternalModelByRole(ModelRolePreference) == nil
}
