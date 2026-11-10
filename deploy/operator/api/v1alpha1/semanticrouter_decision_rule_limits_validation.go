package v1alpha1

import (
	"encoding/json"
	"fmt"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ValidateDecisionRuleLimits checks the same admission and reconciliation
// budget before canonical routing is recursively decoded by the controller.
func (r *SemanticRouter) ValidateDecisionRuleLimits() error {
	spec := r.Spec.Config
	for _, recipes := range []interface{}{spec.Recipes, []interface{}{map[string]interface{}{"name": "default", "decisions": spec.Decisions}}} {
		data, err := json.Marshal(map[string]interface{}{
			"recipes": recipes,
			"global":  map[string]interface{}{"router": map[string]interface{}{"decision_rule_limits": spec.DecisionRuleLimits}},
		})
		if err != nil {
			return fmt.Errorf("config.decision_rule_limits: %w", err)
		}
		if _, err := routerconfig.DecisionRuleLimitsFromYAML(data); err != nil {
			return fmt.Errorf("config: %w", err)
		}
	}
	return nil
}
