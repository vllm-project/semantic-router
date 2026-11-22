package config

import "strings"

// RefusedOnMatch reports whether a request is refused as soon as the rule
// signalType:name matches, whatever else it reads: a decision with a
// fast_response plugin matches whenever that rule does, and, routing by
// priority, no other decision can rank before it. A guard scan stops at the
// first window that makes such a rule match.
func (c *RouterConfig) RefusedOnMatch(signalType, name string) bool {
	if c == nil || c.Strategy == RoutingStrategyConfidence {
		return false
	}
	for index := range c.Decisions {
		refusal := &c.Decisions[index]
		if refusal.GetFastResponseConfig() == nil || refusal.Rules.IsCatchAll() {
			continue
		}
		if _, canFalse := ruleOutcomesOnMatch(&refusal.Rules, signalType, name); canFalse {
			continue
		}
		outranks := true
		for other := range c.Decisions {
			if other != index && !ranksBefore(refusal, &c.Decisions[other]) {
				outranks = false
				break
			}
		}
		if outranks {
			return true
		}
	}
	return false
}

// ranksBefore reports whether decision a is selected over b whenever both
// match, routing by priority: a lower tier, or in one tier a catch-all b or
// a higher priority.
func ranksBefore(a, b *Decision) bool {
	if a.Tier != b.Tier {
		return a.Tier < b.Tier
	}
	return b.Rules.IsCatchAll() || a.Priority > b.Priority
}

// ruleOutcomesOnMatch reports whether node can evaluate true, and whether it
// can evaluate false, when the rule signalType:name matches. Every other
// leaf, a condition on one of the rule's labels or with a predicate of its
// own included, is left free.
func ruleOutcomesOnMatch(node *RuleNode, signalType, name string) (canTrue, canFalse bool) {
	if node.IsLeaf() {
		if strings.EqualFold(strings.TrimSpace(node.Type), signalType) && node.Name == name && node.Label == "" && node.Predicate == nil {
			return true, false
		}
		return true, true
	}
	switch strings.ToUpper(node.Operator) {
	case RuleOperatorNot:
		if len(node.Conditions) != 1 {
			return true, true
		}
		childTrue, childFalse := ruleOutcomesOnMatch(&node.Conditions[0], signalType, name)
		return childFalse, childTrue
	case RuleOperatorOr:
		canTrue, canFalse = false, true
		for index := range node.Conditions {
			childTrue, childFalse := ruleOutcomesOnMatch(&node.Conditions[index], signalType, name)
			canTrue, canFalse = canTrue || childTrue, canFalse && childFalse
		}
		return canTrue, canFalse
	case RuleOperatorAnd, "":
		canTrue, canFalse = true, false
		for index := range node.Conditions {
			childTrue, childFalse := ruleOutcomesOnMatch(&node.Conditions[index], signalType, name)
			canTrue, canFalse = canTrue && childTrue, canFalse || childFalse
		}
		return canTrue, canFalse
	default:
		return true, true
	}
}
