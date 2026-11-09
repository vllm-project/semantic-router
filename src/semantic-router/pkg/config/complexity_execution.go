package config

// ComplexityRuleUsesPrototypes preserves authored hard/easy example banks and
// their signed margin scale. A decision task binding explicitly selects native
// scoring instead; image banks continue to require the multimodal prototype path.
func (c *RouterConfig) ComplexityRuleUsesPrototypes(rule ComplexityRule) bool {
	if len(rule.Hard.ImageCandidates) > 0 || len(rule.Easy.ImageCandidates) > 0 {
		return true
	}
	if c != nil {
		// A rule whose deployment a configuration names asks it, its
		// candidates as examples.
		if task, ok, err := c.ResolveTask("complexity." + rule.Name); ok && err == nil && task.Explicit() {
			return false
		}
	}
	return len(rule.Hard.Candidates) > 0 || len(rule.Easy.Candidates) > 0 || c == nil || c.DecisionModel == ""
}
