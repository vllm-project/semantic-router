package handlers

import (
	"fmt"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	routerdecision "github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// applyEvalTrace drives the result's trace/highlight state from the router's
// recursive eval trace — the accurate source of truth for nested rules,
// classifier labels, and predicates, replacing the flat signal-name
// reconstruction. Returns false when the router returned no trace at all, so
// the caller can fall back to a clearly-labeled derived approximation.
func applyEvalTrace(result *TestQueryResult, routerResp *RouterEvalResponse) bool {
	if len(routerResp.EvalTrace) == 0 {
		return false
	}
	result.EvalTrace = routerResp.EvalTrace

	for _, dt := range routerResp.EvalTrace {
		result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("decision-%s", dt.DecisionName))
	}

	// Highlight leaves along the path of the decision the router actually
	// selected, not every decision that happened to also match while
	// evaluating all signals for the eval view.
	selected := selectedDecisionName(routerResp)
	for _, dt := range routerResp.EvalTrace {
		if dt.DecisionName == selected && dt.Matched {
			appendTraceLeafHighlights(result, dt.RootTrace)
		}
	}
	return true
}

func selectedDecisionName(routerResp *RouterEvalResponse) string {
	if routerResp.DecisionResult != nil {
		return routerResp.DecisionResult.DecisionName
	}
	return routerResp.RoutingDecision
}

func appendTraceLeafHighlights(result *TestQueryResult, node *routerdecision.TraceNode) {
	if node == nil {
		return
	}
	if node.NodeType == "leaf" && node.Matched {
		result.HighlightedPath = append(
			result.HighlightedPath,
			fmt.Sprintf("signal-%s-%s", node.SignalType, node.SignalName),
		)
	}
	for _, child := range node.Children {
		appendTraceLeafHighlights(result, child)
	}
}

// verifyRecipeScope echoes back what the router actually resolved and flags
// a mismatch against what the request's model should resolve to locally, so
// Topology never silently presents one recipe's outcome as another's.
func verifyRecipeScope(result *TestQueryResult, routerResp *RouterEvalResponse, configPath, requestModel string) {
	result.RequestedModel = routerResp.RequestedModel
	result.Recipe = routerResp.Recipe

	if requestModel == "" || routerResp.Recipe == "" {
		return
	}
	parsedConfig, err := routerconfig.Parse(configPath)
	if err != nil || parsedConfig == nil {
		return
	}
	expected, ok := parsedConfig.RecipeForRoutingModel(requestModel)
	if !ok || expected == nil {
		return
	}
	if string(expected.Name) != routerResp.Recipe {
		result.IsAccurate = false
		appendWarning(result, fmt.Sprintf(
			"Router evaluated recipe %q, expected %q for the selected scope",
			routerResp.Recipe, expected.Name,
		))
	}
}

func appendWarning(result *TestQueryResult, warning string) {
	if result.Warning == "" {
		result.Warning = warning
		return
	}
	result.Warning = result.Warning + "; " + warning
}

// appendEvaluatedRulesFromConfig is the compatibility fallback used only when
// the router returned no eval trace (older router, or a request that failed
// before eval_trace could be produced). It reads decisions' top-level
// conditions only, so it cannot represent nested rule groups, classifier
// labels, predicates, or on_error — callers must not report IsAccurate=true
// while this is the source.
func appendEvaluatedRulesFromConfig(result *TestQueryResult, configPath, requestModel string) {
	parsedConfig, err := routerconfig.Parse(configPath)
	if err != nil || parsedConfig == nil {
		return
	}

	parsedConfig = topologyConfigForRequestModel(parsedConfig, requestModel)
	matchedSignalNames := buildMatchedSignalNameSet(result.MatchedSignals)
	for _, decision := range parsedConfig.IntelligentRouting.Decisions {
		if result.MatchedDecision != "" && decision.Name == result.MatchedDecision {
			continue
		}
		result.EvaluatedRules = append(result.EvaluatedRules, buildEvaluatedRule(decision, matchedSignalNames))
	}
}

func topologyConfigForRequestModel(
	parsedConfig *routerconfig.RouterConfig,
	requestModel string,
) *routerconfig.RouterConfig {
	if parsedConfig == nil {
		return nil
	}
	recipe, ok := parsedConfig.RecipeForRoutingModel(requestModel)
	if !ok {
		return parsedConfig
	}
	scoped := parsedConfig.ConfigForRecipe(recipe)
	if scoped == nil {
		return parsedConfig
	}
	return scoped
}

func buildMatchedSignalNameSet(signals []MatchedSignal) map[string]bool {
	matchedSignalNames := make(map[string]bool, len(signals)*2)
	for _, signal := range signals {
		key := fmt.Sprintf("%s:%s", signal.Type, signal.Name)
		normalizedKey := fmt.Sprintf("%s:%s", signal.Type, normalizeSignalName(signal.Name))
		matchedSignalNames[key] = true
		matchedSignalNames[normalizedKey] = true
	}
	return matchedSignalNames
}

func buildEvaluatedRule(decision routerconfig.Decision, matchedSignalNames map[string]bool) EvaluatedRule {
	rule := EvaluatedRule{
		DecisionName: decision.Name,
		RuleOperator: strings.ToUpper(decision.Rules.Operator),
		Conditions:   []string{},
		IsMatch:      false,
		Priority:     decision.Priority,
	}
	if rule.RuleOperator == "" {
		rule.RuleOperator = "AND"
	}

	for _, condition := range decision.Rules.Conditions {
		conditionKey := fmt.Sprintf("%s:%s", condition.Type, condition.Name)
		normalizedConditionKey := fmt.Sprintf("%s:%s", condition.Type, normalizeSignalName(condition.Name))
		rule.Conditions = append(rule.Conditions, conditionKey)
		rule.TotalCount++
		if matchedSignalNames[conditionKey] || matchedSignalNames[normalizedConditionKey] {
			rule.MatchedCount++
		}
	}

	switch {
	case rule.TotalCount == 0:
		rule.IsMatch = true
	case rule.RuleOperator == "OR":
		rule.IsMatch = rule.MatchedCount > 0
	default:
		rule.IsMatch = rule.MatchedCount == rule.TotalCount
	}

	return rule
}
