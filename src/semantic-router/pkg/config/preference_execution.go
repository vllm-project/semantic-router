package config

import "strings"

// PreferenceUsesPrototypes reports whether preference rules score against
// embedding prototypes, keeping authored example banks and their cosine
// threshold and margin scale. A binding to a model_runtime or systemone
// deployment asks a judgment instead, even when the module asks for
// contrastive scoring; another binding, or an external model with the
// preference role, asks that model unless the module asks for contrastive
// scoring.
func (c *RouterConfig) PreferenceUsesPrototypes() bool {
	if c == nil {
		return false
	}
	native, bound := c.preferenceBinding()
	if native {
		return false
	}
	if c.PreferenceModel.UseContrastive != nil {
		return *c.PreferenceModel.UseContrastive
	}
	if bound || c.FindExternalModelByRole(ModelRolePreference) != nil {
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
// selection aligned. Description-only rules ask the decision model, or the
// model_runtime or systemone deployment a binding names; an external model
// keeps its own execution.
func (c *RouterConfig) PreferenceUsesDecisionTask() bool {
	if c == nil {
		return false
	}
	native, bound := c.preferenceBinding()
	if native {
		return true
	}
	return !bound && !c.PreferenceUsesPrototypes() && c.FindExternalModelByRole(ModelRolePreference) == nil
}

// preferenceBinding reports whether a rule or binding names the preference
// signal's deployment, and whether that deployment answers it as a native
// judgment. A binding to an undeclared deployment, which validation
// rejects, is read as native.
func (c *RouterConfig) preferenceBinding() (native, bound bool) {
	resolved, ok, err := c.ResolveTask("preference")
	switch {
	case err != nil && !IsDeferredModelError(err):
		return true, true
	case !ok || err != nil || !resolved.Explicit():
		return false, false
	}
	return resolved.Deployment.IsModelRuntime() || resolved.Deployment.IsSystemOne(), true
}

// PreferenceExternalDeployment is the openai_chat deployment a binding names
// for the preference signal, which asks it to pick a route.
func (c *RouterConfig) PreferenceExternalDeployment() (ModelDeployment, bool) {
	resolved, bound := c.boundTask("preference")
	if !bound || resolved.Deployment.Provider != OpenAIChatProvider {
		return ModelDeployment{}, false
	}
	return resolved.Deployment, true
}
