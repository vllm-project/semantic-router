package config

// ReaskUsesDecisionTask selects probability-based intent judgment only when an
// author explicitly binds that contract. Unbound reask rules retain their
// published cosine-similarity scale, regardless of the default decision model.
func (c *RouterConfig) ReaskUsesDecisionTask() bool {
	if c == nil {
		return false
	}
	return c.EffectiveModelBindings(c.Signals, c.ModelBindings)["reask"].Contract == DecisionTaskContract
}
