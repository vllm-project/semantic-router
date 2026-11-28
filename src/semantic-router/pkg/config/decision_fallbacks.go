package config

import "fmt"

// DecisionDeploymentChain lists the primary and its fallback deployments in
// call order. Try the primary's own list before any nested list, and visit a
// deployment at most once even when several lists name it.
func (c *RouterConfig) DecisionDeploymentChain(primary string) []string {
	chain := []string{primary}
	if len(c.ModelDeployments[primary].Fallbacks) == 0 {
		return chain
	}
	seen := map[string]bool{primary: true}
	for i := 0; i < len(chain); i++ {
		for _, next := range c.ModelDeployments[chain[i]].Fallbacks {
			if !seen[next] {
				seen[next] = true
				chain = append(chain, next)
			}
		}
	}
	return chain
}

func validateDecisionDeploymentFallbacks(cfg *RouterConfig) error {
	checked := make(map[string]bool)
	visiting := make(map[string]bool)
	for _, name := range sortedModelKeys(cfg.ModelDeployments) {
		if len(cfg.ModelDeployments[name].Fallbacks) == 0 {
			continue
		}
		if err := validateFallbackList(cfg, name, checked, visiting); err != nil {
			return err
		}
	}
	return nil
}

func validateFallbackList(cfg *RouterConfig, name string, checked, visiting map[string]bool) error {
	if checked[name] {
		return nil
	}
	path := "global.model_catalog.deployments." + name + ".fallbacks"
	if err := validateDecisionModelReference(cfg, name); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	visiting[name] = true
	seen := make(map[string]bool)
	for i, next := range cfg.ModelDeployments[name].Fallbacks {
		if seen[next] {
			return fmt.Errorf("%s[%d]: deployment %q is listed twice", path, i, next)
		}
		seen[next] = true
		if visiting[next] {
			return fmt.Errorf("%s[%d]: cyclic decision fallback to deployment %q", path, i, next)
		}
		if err := validateDecisionModelReference(cfg, next); err != nil {
			return fmt.Errorf("%s[%d]: %w", path, i, err)
		}
		if len(cfg.ModelDeployments[next].Fallbacks) != 0 {
			if err := validateFallbackList(cfg, next, checked, visiting); err != nil {
				return err
			}
		}
	}
	delete(visiting, name)
	checked[name] = true
	return nil
}
