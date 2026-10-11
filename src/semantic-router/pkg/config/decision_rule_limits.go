package config

import "fmt"

const (
	DefaultDecisionRuleMaxDepth = 16
	DefaultDecisionRuleMaxNodes = 256
)

// DecisionRuleLimits bounds each decision independently. The root has depth
// one; both operators and leaves count as nodes. Nil members use defaults.
type DecisionRuleLimits struct {
	MaxDepth *int `yaml:"max_depth,omitempty" json:"max_depth,omitempty" jsonschema:"minimum=1,default=16"`
	MaxNodes *int `yaml:"max_nodes,omitempty" json:"max_nodes,omitempty" jsonschema:"minimum=1,default=256"`
}

// DefaultDecisionRuleLimits returns the shared defaults for every producer.
func DefaultDecisionRuleLimits() DecisionRuleLimits {
	return DecisionRuleLimits{
		MaxDepth: canonicalIntPtr(DefaultDecisionRuleMaxDepth),
		MaxNodes: canonicalIntPtr(DefaultDecisionRuleMaxNodes),
	}
}

// Effective resolves omitted members and rejects explicit nonpositive limits.
func (limits DecisionRuleLimits) Effective() (maxDepth, maxNodes int, err error) {
	maxDepth, maxNodes = DefaultDecisionRuleMaxDepth, DefaultDecisionRuleMaxNodes
	if limits.MaxDepth != nil {
		maxDepth = *limits.MaxDepth
	}
	if limits.MaxNodes != nil {
		maxNodes = *limits.MaxNodes
	}
	for _, field := range []struct {
		name  string
		value int
	}{{"max_depth", maxDepth}, {"max_nodes", maxNodes}} {
		if field.value < 1 {
			return 0, 0, fmt.Errorf("global.router.decision_rule_limits.%s must be a positive integer", field.name)
		}
	}
	return maxDepth, maxNodes, nil
}

// ValidateDecisionRuleLimits protects in-memory producers before they recurse
// over decisions. It also runs for static Kubernetes configuration.
func ValidateDecisionRuleLimits(cfg *RouterConfig) error {
	if cfg == nil {
		return fmt.Errorf("router configuration is nil")
	}
	depth, nodes, err := cfg.DecisionRuleLimits.Effective()
	if err != nil {
		return err
	}
	validate := func(decisions []Decision, recipe RecipeName) error {
		for i := range decisions {
			decision := &decisions[i]
			if err := validateRuleTreeBudget(decision.Rules, depth, nodes, func(node RuleNode) []RuleNode {
				return node.Conditions
			}); err != nil {
				return wrapRoutingProfileValidationError(recipe, fmt.Errorf("decision %q: %w", decision.Name, err))
			}
		}
		return nil
	}
	if err := validate(cfg.Decisions, DefaultRecipeName); err != nil {
		return err
	}
	for _, recipe := range cfg.Recipes {
		if err := validate(recipe.Profile.Decisions, recipe.Name); err != nil {
			return err
		}
	}
	return nil
}

// DecisionRuleLimitsFromYAML reads an enclosing document's budget without
// requiring its provider configuration to be complete. Existing trees are
// checked before callers recursively decode or edit that document.
func DecisionRuleLimitsFromYAML(data []byte) (DecisionRuleLimits, error) {
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return DecisionRuleLimits{}, err
	}
	return validateRawDecisionRuleLimits(raw)
}

// validateRawDecisionRuleLimits runs immediately after YAML decoding, before
// recursive document normalization, environment expansion or typed decoding.
func validateRawDecisionRuleLimits(raw map[string]interface{}) (DecisionRuleLimits, error) {
	limits := DecisionRuleLimits{}
	settings := nestedStringMap(nestedStringMap(nestedStringMap(raw["global"])["router"])["decision_rule_limits"])
	for _, field := range []struct {
		name string
		dest **int
	}{{"max_depth", &limits.MaxDepth}, {"max_nodes", &limits.MaxNodes}} {
		if value, exists := settings[field.name]; exists {
			integer, ok := value.(int)
			if !ok || integer < 1 {
				return limits, fmt.Errorf("global.router.decision_rule_limits.%s must be a positive integer", field.name)
			}
			*field.dest = &integer
		}
	}
	depth, nodes, err := limits.Effective()
	if err != nil {
		return limits, err
	}
	validate := func(routing map[string]interface{}, recipe RecipeName) error {
		decisions, _ := routing["decisions"].([]interface{})
		for _, value := range decisions {
			decision := nestedStringMap(value)
			if err := validateRuleTreeBudget(decision["rules"], depth, nodes, func(node interface{}) []interface{} {
				children, _ := nestedStringMap(node)["conditions"].([]interface{})
				return children
			}); err != nil {
				return wrapRoutingProfileValidationError(recipe, fmt.Errorf("decision %q: %w", decision["name"], err))
			}
		}
		return nil
	}
	if err := validate(nestedStringMap(raw["routing"]), DefaultRecipeName); err != nil {
		return limits, err
	}
	recipes, _ := raw["recipes"].([]interface{})
	for _, value := range recipes {
		recipe := nestedStringMap(value)
		name, _ := recipe["name"].(string)
		if err := validate(nestedStringMap(recipe["routing"]), RecipeName(name)); err != nil {
			return limits, err
		}
	}
	return limits, nil
}

// validateRuleTreeBudget uses a depth-first cursor stack, so even a very wide
// rejected tree never needs an auxiliary allocation for all its children.
func validateRuleTreeBudget[T any](root T, maxDepth, maxNodes int, children func(T) []T) error {
	type frame struct {
		children []T
		next     int
		path     string
	}
	stack := []frame{{children: children(root), path: "rules"}}
	count := 1
	for len(stack) > 0 {
		parent := &stack[len(stack)-1]
		if parent.next == len(parent.children) {
			stack = stack[:len(stack)-1]
			continue
		}
		child := parent.children[parent.next]
		path := fmt.Sprintf("%s.conditions[%d]", parent.path, parent.next)
		parent.next++
		depth := len(stack) + 1
		count++
		if depth > maxDepth {
			return fmt.Errorf("%s: depth %d exceeds max_depth=%d", path, depth, maxDepth)
		}
		if count > maxNodes {
			return fmt.Errorf("%s: node count %d exceeds max_nodes=%d", path, count, maxNodes)
		}
		stack = append(stack, frame{children: children(child), path: path})
	}
	return nil
}
