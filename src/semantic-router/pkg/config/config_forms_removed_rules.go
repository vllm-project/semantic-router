package config

import (
	"fmt"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// A condition on a removed signal never matches. Migrate keeps what the rest
// of a decision's rules still decide: an OR drops the condition, an AND that
// needs it never matches, and a NOT of it always matches. A decision whose
// rules never match is dropped, and one whose rules always match routes every
// request it is tried on.

// ruleOutcome is what a rule tree decides once its conditions on a removed
// signal never match.
type ruleOutcome int

const (
	ruleDepends ruleOutcome = iota // on the conditions that remain
	ruleNever
	ruleAlways
)

// settleRule rewrites the rule tree at node for the conditions removed says
// never match, and returns what the tree decides and whether it held such a
// condition. A subtree without one is left as written.
func settleRule(node *yaml.Node, removed func(*yaml.Node) bool) (ruleOutcome, bool) {
	node = resolveAlias(node)
	if node == nil || node.Kind != yaml.MappingNode {
		return ruleDepends, false
	}
	operator := strings.ToUpper(strings.TrimSpace(scalarAt(node, "operator")))
	if operator == "" {
		if removed(node) {
			return ruleNever, true
		}
		return ruleDepends, false
	}
	conditions := resolveAlias(mappingValue(node, "conditions"))
	if conditions == nil || conditions.Kind != yaml.SequenceNode {
		return ruleDepends, false
	}
	outcomes := make([]ruleOutcome, len(conditions.Content))
	held := false
	for i, child := range conditions.Content {
		var changed bool
		outcomes[i], changed = settleRule(child, removed)
		held = held || changed
	}
	if !held {
		return ruleDepends, false
	}
	without := func(dropped ruleOutcome) {
		kept := conditions.Content[:0]
		for i, child := range conditions.Content {
			if outcomes[i] != dropped {
				kept = append(kept, child)
			}
		}
		conditions.Content = kept
	}
	switch operator {
	case RuleOperatorAnd:
		if slices.Contains(outcomes, ruleNever) {
			return ruleNever, true
		}
		without(ruleAlways)
		if len(conditions.Content) == 0 {
			return ruleAlways, true
		}
	case RuleOperatorNot:
		switch {
		case slices.Contains(outcomes, ruleNever):
			return ruleAlways, true
		case slices.Contains(outcomes, ruleAlways):
			return ruleNever, true
		}
	default:
		if slices.Contains(outcomes, ruleAlways) {
			return ruleAlways, true
		}
		without(ruleNever)
		if len(conditions.Content) == 0 {
			return ruleNever, true
		}
	}
	return ruleDepends, true
}

// neverMatches says why a condition never matches, and how to keep what the
// decisions that read it did.
type neverMatches struct {
	why, restore string
}

// settleDecision settles a decision's rules for the conditions removed says
// never match, and drops the decision when they never match. The notes name
// those conditions as what reports once the rules are settled.
func settleDecision(m *Migration, decisions, decision *yaml.Node, path string, removed func(*yaml.Node) bool, what func() string, reason neverMatches) {
	outcome, held := settleRule(mappingValue(decision, "rules"), removed)
	switch {
	case !held:
	case outcome == ruleNever:
		removeItem(decisions, decision)
		m.Note(path, fmt.Sprintf("the decision is dropped: its rules need %s, and %s; %s", what(), reason.why, reason.restore))
	case outcome == ruleAlways:
		matchAll := newMapping("operator", Scalar(RuleOperatorAnd), "conditions", sequence())
		matchAll.Style = yaml.FlowStyle
		setNode(decision, "rules", matchAll)
		m.Note(path+".rules", fmt.Sprintf("its rules match every request now: they held whenever %s did not match, and %s; %s", what(), reason.why, reason.restore))
	default:
		m.Changed(path+".rules", fmt.Sprintf("dropped from its rules: %s, since %s", what(), reason.why))
	}
}

// settleRemovedCondition settles the decision that holds the condition at,
// unless an earlier match settled it already. kind names the conditions the
// notes list, such as "kb rule".
func settleRemovedCondition(m *Migration, at FormMatch, kind string, removed func(*yaml.Node) bool, reason neverMatches) {
	decisions, decision := decisionHolding(at)
	if decision == nil || !slices.Contains(decisions.Content, decision) || !ruleTreeHolds(mappingValue(decision, "rules"), at.Node) {
		return
	}
	var names []string
	collect := func(node *yaml.Node) bool {
		if !removed(node) {
			return false
		}
		if name := scalarAt(node, "name"); !slices.Contains(names, name) {
			names = append(names, name)
		}
		return true
	}
	path := at.Path
	if decisionPath, _, found := strings.Cut(at.Path, "].rules"); found {
		path = decisionPath + "]"
	}
	settleDecision(m, decisions, decision, path, collect, func() string { return conditionNames(kind, names) }, reason)
}

// conditionNames lists conditions of one kind: "kb rule a", "kb rules a, b".
func conditionNames(kind string, names []string) string {
	if len(names) == 1 {
		return kind + " " + names[0]
	}
	return kind + "s " + strings.Join(names, ", ")
}

// decisionHolding is the decisions sequence and the decision that hold a
// match within decisions[].rules.
func decisionHolding(at FormMatch) (decisions, decision *yaml.Node) {
	chain := append(slices.Clip(at.Ancestors), at.Parent)
	for i := 1; i+1 < len(chain); i++ {
		if chain[i].Kind == yaml.SequenceNode && resolveAlias(mappingValue(chain[i-1], "decisions")) == chain[i] {
			return chain[i], chain[i+1]
		}
	}
	return nil, nil
}

// ruleTreeHolds reports whether node is part of the rule tree at rules.
func ruleTreeHolds(rules, node *yaml.Node) bool {
	rules = resolveAlias(rules)
	if rules == nil {
		return false
	}
	if rules == node {
		return true
	}
	return slices.ContainsFunc(sequenceItems(resolveAlias(mappingValue(rules, "conditions"))), func(child *yaml.Node) bool {
		return ruleTreeHolds(child, node)
	})
}

// dropScoreInput drops the projection score input at, which reads a removed
// signal, and the score when it reads nothing else: then also the mappings of
// the score, and the conditions on their outputs, which never match.
func dropScoreInput(m *Migration, at FormMatch, input string, reason neverMatches) {
	chain := append(slices.Clip(at.Ancestors), at.Parent)
	if len(chain) < 5 || !slices.Contains(at.Parent.Content, at.Node) {
		return
	}
	score, scores, projections, recipe := chain[len(chain)-2], chain[len(chain)-3], chain[len(chain)-4], chain[len(chain)-5]
	scorePath := at.Path
	if i := strings.LastIndex(at.Path, ".inputs["); i >= 0 {
		scorePath = at.Path[:i]
	}
	removeItem(at.Parent, at.Node)
	if len(at.Parent.Content) > 0 {
		m.Note(scorePath, fmt.Sprintf("the %s input is dropped: %s; re-fit the bands of the mappings that read the score", input, reason.why))
		return
	}
	removeItem(scores, score)
	scoreName := scalarAt(score, "name")
	prefix, _, _ := strings.Cut(scorePath, ".projections.")
	var outputs []string
	if mappings := resolveAlias(mappingValue(projections, "mappings")); mappings != nil {
		for _, mapping := range slices.Clone(mappings.Content) {
			if scalarAt(mapping, "source") != scoreName {
				continue
			}
			for _, output := range sequenceAt(mapping, "outputs") {
				outputs = append(outputs, scalarAt(output, "name"))
			}
			removeItem(mappings, mapping)
		}
		if len(mappings.Content) == 0 {
			m.Delete(projections, "mappings")
		}
	}
	if len(sequenceItems(scores)) == 0 {
		m.Delete(projections, "scores")
	}
	deleteEmpty(recipe, "projections")
	m.Note(scorePath, fmt.Sprintf("score %s is dropped: it read only the %s input, and %s; %s", scoreName, input, reason.why, reason.restore))
	if len(outputs) == 0 {
		return
	}
	output := neverMatches{why: "score " + scoreName + " is dropped", restore: reason.restore}
	decisions := resolveAlias(mappingValue(recipe, "decisions"))
	for _, decision := range slices.Clone(sequenceItems(decisions)) {
		var names []string
		removed := func(node *yaml.Node) bool {
			name := scalarAt(node, "name")
			if scalarAt(node, "type") != SignalTypeProjection || !slices.Contains(outputs, name) {
				return false
			}
			if !slices.Contains(names, name) {
				names = append(names, name)
			}
			return true
		}
		path := prefix + ".decisions[" + itemLabel(decision, slices.Index(decisions.Content, decision)) + "]"
		settleDecision(m, decisions, decision, path, removed, func() string { return conditionNames("projection output", names) }, output)
	}
}
