package config

import (
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// contextCapacityFinding reports a band whose reachable models do not match
// the context windows declared in model_config.
type contextCapacityFinding struct {
	band string
	// reachableWindow is the largest window among models the band can route to.
	reachableWindow int
	// unreachableModel and unreachableWindow name a routed model declaring a
	// larger window that no decision gated on the band can use. Empty when the
	// band already reaches the largest routed model.
	unreachableModel  string
	unreachableWindow int
	// minTokens is set when the band matches more tokens than any model it
	// reaches can hold.
	minTokens int
}

// gainsNoCapacity reports whether a larger routed model exists but is out of
// reach of this band.
func (f contextCapacityFinding) gainsNoCapacity() bool { return f.unreachableModel != "" }

// cannotServe reports whether the band matches requests none of its models fit.
func (f contextCapacityFinding) cannotServe() bool { return f.minTokens > 0 }

// validateContextCapacity cross-checks routing.signals.context bands against
// the context windows declared in model_config. The two are otherwise
// validated independently, so a band can route large requests to a model no
// larger than the one they came from, or to one that cannot hold what the
// band matches, and nothing says so.
//
// Findings are logged rather than returned. Escalating on size for
// long-context quality at equal window size is legitimate, so this never
// rejects a configuration.
func validateContextCapacity(cfg *RouterConfig) error {
	logContextCapacityFindings(contextCapacityIssues(cfg))
	return nil
}

// contextCapacityIssues returns one finding per band that has something to
// report, in configuration order.
func contextCapacityIssues(cfg *RouterConfig) []contextCapacityFinding {
	if cfg == nil || len(cfg.ContextRules) == 0 || len(cfg.Decisions) == 0 {
		return nil
	}
	// Only models some decision routes to are alternatives the operator could
	// switch a band to. A model declared in the catalogue but routed nowhere,
	// which is ordinary in a recipe-scoped configuration, is not.
	largestModel, largestWindow := largestDeclaredWindow(cfg.ModelConfig, routedModels(cfg.Decisions))

	var findings []contextCapacityFinding
	for _, rule := range cfg.ContextRules {
		band := strings.TrimSpace(rule.Name)
		if band == "" {
			continue
		}
		bounds, err := rule.Bounds()
		if err != nil || bounds.Min <= 0 {
			// Ranges are validated by validateContextContracts, and a band
			// with no lower bound gates nothing on size.
			continue
		}
		_, reachable := largestDeclaredWindow(cfg.ModelConfig, bandReachableModels(cfg.Decisions, band))
		if reachable <= 0 {
			// The band reaches no model, or none that declares a window.
			continue
		}
		finding := contextCapacityFinding{band: band, reachableWindow: reachable}
		if largestWindow > reachable {
			finding.unreachableModel, finding.unreachableWindow = largestModel, largestWindow
		}
		if bounds.Min > reachable {
			finding.minTokens = bounds.Min
		}
		if finding.gainsNoCapacity() || finding.cannotServe() {
			findings = append(findings, finding)
		}
	}
	return findings
}

func logContextCapacityFindings(findings []contextCapacityFinding) {
	for _, finding := range findings {
		if finding.gainsNoCapacity() {
			logging.Warnf(
				"routing.signals.context[%q]: decisions gated on this band declare a maximum context window of %d, but model %q declares %d and no decision the band can reach uses it; escalation on context size gains no capacity",
				finding.band, finding.reachableWindow, finding.unreachableModel, finding.unreachableWindow,
			)
		}
		if finding.cannotServe() {
			logging.Warnf(
				"routing.signals.context[%q]: min_tokens is %d but no model the band can reach declares more than %d; the decisions it gates can never serve what it matches",
				finding.band, finding.minTokens, finding.reachableWindow,
			)
		}
	}
}

// bandReachableModels returns every model named by a decision that a request
// inside the band can match, sorted so findings are deterministic. A decision
// must reference the band and its rule tree must still be satisfiable with the
// band true, so a decision gated on NOT band is excluded.
func bandReachableModels(decisions []Decision, band string) []string {
	return decisionModels(decisions, func(decision Decision) bool {
		names := make(map[string]bool)
		collectRuleNames(decision.Rules, SignalTypeContext, names)
		if !names[band] {
			return false
		}
		canMatch, _ := bandRuleOutcomes(&decision.Rules, band)
		return canMatch
	})
}

// bandRuleOutcomes reports whether node can evaluate true, and whether it can
// evaluate false, for a request inside band. Every other leaf is left free, so
// correlated leaves are not cross-checked and the result can only overstate
// what matches.
func bandRuleOutcomes(node *RuleNode, band string) (canTrue, canFalse bool) {
	if node.IsLeaf() {
		if node.Type == SignalTypeContext && node.Name == band {
			return true, false
		}
		return true, true
	}
	switch node.Operator {
	case RuleOperatorNot:
		if len(node.Conditions) != 1 {
			return true, true
		}
		childTrue, childFalse := bandRuleOutcomes(&node.Conditions[0], band)
		return childFalse, childTrue
	case RuleOperatorOr:
		canTrue, canFalse = false, true
		for i := range node.Conditions {
			childTrue, childFalse := bandRuleOutcomes(&node.Conditions[i], band)
			canTrue, canFalse = canTrue || childTrue, canFalse && childFalse
		}
		return canTrue, canFalse
	case RuleOperatorAnd, "":
		canTrue, canFalse = true, false
		for i := range node.Conditions {
			childTrue, childFalse := bandRuleOutcomes(&node.Conditions[i], band)
			canTrue, canFalse = canTrue && childTrue, canFalse || childFalse
		}
		return canTrue, canFalse
	default:
		return true, true
	}
}

// routedModels returns every model any decision names, sorted.
func routedModels(decisions []Decision) []string {
	return decisionModels(decisions, func(Decision) bool { return true })
}

func decisionModels(decisions []Decision, include func(Decision) bool) []string {
	seen := make(map[string]struct{})
	for i := range decisions {
		if !include(decisions[i]) {
			continue
		}
		for _, ref := range decisions[i].ModelRefs {
			if model := strings.TrimSpace(ref.Model); model != "" {
				seen[model] = struct{}{}
			}
		}
	}
	models := make([]string, 0, len(seen))
	for model := range seen {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

// largestDeclaredWindow returns the model declaring the biggest context window
// among models, ignoring any the configuration does not describe. Ties break on
// the first name in sorted order so messages stay stable.
func largestDeclaredWindow(modelConfig map[string]ModelParams, models []string) (string, int) {
	best, window := "", 0
	for _, model := range models {
		if size := modelConfig[model].ContextWindowSize; size > window {
			best, window = model, size
		}
	}
	return best, window
}
