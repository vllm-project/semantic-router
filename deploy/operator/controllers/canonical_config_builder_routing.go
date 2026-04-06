package controllers

import (
	"fmt"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
)

func (r *SemanticRouterReconciler) applyOperatorRouting(
	canonical *routercontract.CanonicalConfig,
	spec vllmv1alpha1.ConfigSpec,
) error {
	if len(spec.ComplexityRules) > 0 {
		complexity, err := r.convertComplexityRules(spec.ComplexityRules)
		if err != nil {
			return fmt.Errorf("config.complexity_rules: %w", err)
		}
		canonical.Routing.Signals.Complexity = complexity
	}
	if len(spec.Decisions) > 0 {
		decisions, err := convertToTypedConfig[[]routercontract.Decision](r, spec.Decisions)
		if err != nil {
			return fmt.Errorf("config.decisions: %w", err)
		}
		canonical.Routing.Decisions = decisions
	}
	return nil
}

func (r *SemanticRouterReconciler) convertComplexityRules(
	spec []vllmv1alpha1.ComplexityRulesConfig,
) ([]routercontract.ComplexityRule, error) {
	rules := make([]routercontract.ComplexityRule, 0, len(spec))
	for _, entry := range spec {
		rule, err := convertToTypedConfig[routercontract.ComplexityRule](r, entry)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", entry.Name, err)
		}
		if entry.Composer != nil {
			rule.Composer = &routercontract.RuleCombination{
				Operator:   entry.Composer.Operator,
				Conditions: convertCompositionConditions(entry.Composer.Conditions),
			}
		}
		rules = append(rules, rule)
	}
	return rules, nil
}

func convertCompositionConditions(conditions []vllmv1alpha1.CompositionCondition) []routercontract.RuleNode {
	result := make([]routercontract.RuleNode, 0, len(conditions))
	for _, condition := range conditions {
		result = append(result, routercontract.RuleNode{
			Type: condition.Type,
			Name: condition.Name,
		})
	}
	return result
}
