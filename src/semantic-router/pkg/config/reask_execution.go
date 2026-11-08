package config

// ReaskUsesDecisionTask selects probability-based intent judgment when a
// binding names a deployment for reask: the signal implies the decision.v1
// contract unless the binding writes another. Unbound reask rules retain their
// published cosine-similarity scale, regardless of the default decision model.
func (c *RouterConfig) ReaskUsesDecisionTask() bool {
	if c == nil {
		return false
	}
	binding, bound := c.EffectiveModelBindings(c.Signals, c.ModelBindings)["reask"]
	return bound && (binding.Contract == "" || binding.Contract == DecisionTaskContract)
}
