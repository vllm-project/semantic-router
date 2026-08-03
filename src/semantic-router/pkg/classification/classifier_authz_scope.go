package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type partialTruth uint8

const (
	partialFalse partialTruth = iota
	partialTrue
	partialUnknown
)

// filterDecisionsByAuthz removes only decisions whose rule tree is already
// known to be false after authz evaluation. Decisions that still depend on
// non-authz signals, or whose rule shape is unsupported, remain candidates.
func filterDecisionsByAuthz(decisions []config.Decision, matchedAuthzRules []string) []config.Decision {
	matched := make(map[string]struct{}, len(matchedAuthzRules))
	for _, role := range matchedAuthzRules {
		matched[role] = struct{}{}
	}

	candidates := make([]config.Decision, 0, len(decisions))
	for _, decision := range decisions {
		if evaluateRuleWithAuthz(decision.Rules, matched) != partialFalse {
			candidates = append(candidates, decision)
		}
	}
	return candidates
}

func evaluateRuleWithAuthz(node config.RuleNode, matched map[string]struct{}) partialTruth {
	if node.IsLeaf() {
		if strings.EqualFold(strings.TrimSpace(node.Type), config.SignalTypeAuthz) {
			if _, ok := matched[node.Name]; ok {
				return partialTrue
			}
			return partialFalse
		}
		return partialUnknown
	}

	switch strings.ToUpper(node.Operator) {
	case "AND":
		return evaluateAuthzAND(node.Conditions, matched)
	case "NOT":
		if len(node.Conditions) != 1 {
			return partialUnknown
		}
		return negatePartialTruth(evaluateRuleWithAuthz(node.Conditions[0], matched))
	case "OR", "":
		return evaluateAuthzOR(node.Conditions, matched)
	default:
		return partialUnknown
	}
}

func evaluateAuthzAND(children []config.RuleNode, matched map[string]struct{}) partialTruth {
	result := partialTrue
	for _, child := range children {
		switch evaluateRuleWithAuthz(child, matched) {
		case partialFalse:
			return partialFalse
		case partialUnknown:
			result = partialUnknown
		}
	}
	return result
}

func evaluateAuthzOR(children []config.RuleNode, matched map[string]struct{}) partialTruth {
	result := partialFalse
	for _, child := range children {
		switch evaluateRuleWithAuthz(child, matched) {
		case partialTrue:
			return partialTrue
		case partialUnknown:
			result = partialUnknown
		}
	}
	return result
}

func negatePartialTruth(value partialTruth) partialTruth {
	switch value {
	case partialTrue:
		return partialFalse
	case partialFalse:
		return partialTrue
	default:
		return partialUnknown
	}
}
