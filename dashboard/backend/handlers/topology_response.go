package handlers

import (
	"fmt"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type topologySignalMapping struct {
	signalType        string
	names             []string
	defaultConfidence float64
	reason            string
	addPath           bool
}

// convertRouterResponse converts Router API response to TestQueryResult.
func convertRouterResponse(req TestQueryRequest, routerResp *RouterEvalResponse, configPath string) *TestQueryResult {
	result := newTestQueryResult(req)

	appendMatchedSignals(result, routerResp)
	appendSignalGroupHighlights(result)
	applyRouterDecision(result, routerResp)
	applyRecommendedModels(result, routerResp.RecommendedModels)
	appendEvaluatedRulesFromConfig(result, configPath)

	return result
}

func newTestQueryResult(req TestQueryRequest) *TestQueryResult {
	return &TestQueryResult{
		Query:           req.Query,
		Mode:            req.Mode,
		MatchedSignals:  []MatchedSignal{},
		MatchedModels:   []string{},
		HighlightedPath: []string{"client"},
		IsAccurate:      true,
		EvaluatedRules:  []EvaluatedRule{},
	}
}

func appendMatchedSignals(result *TestQueryResult, routerResp *RouterEvalResponse) {
	matchedSignals := matchedRouterSignals(routerResp)
	if matchedSignals == nil {
		return
	}

	for _, mapping := range topologySignalMappings(matchedSignals) {
		addMatchedSignals(result, mapping, routerResp.SignalConfidences, routerResp.SignalValues)
	}
}

func matchedRouterSignals(routerResp *RouterEvalResponse) *RouterMatchedSignals {
	if routerResp == nil || routerResp.DecisionResult == nil {
		return nil
	}
	return routerResp.DecisionResult.MatchedSignals
}

func topologySignalMappings(matchedSignals *RouterMatchedSignals) []topologySignalMapping {
	return []topologySignalMapping{
		{signalType: "keyword", names: matchedSignals.Keywords, defaultConfidence: 1.0, reason: "Keyword rule matched", addPath: true},
		{signalType: "embedding", names: matchedSignals.Embeddings, defaultConfidence: 0.85, reason: "Embedding similarity matched", addPath: true},
		{signalType: "domain", names: matchedSignals.Domains, defaultConfidence: 1.0, reason: "Domain classification matched", addPath: true},
		{signalType: "fact_check", names: matchedSignals.FactCheck, defaultConfidence: 0.9, reason: "Fact check signal matched"},
		{signalType: "preference", names: matchedSignals.Preferences, defaultConfidence: 1.0, reason: "User preference matched", addPath: true},
		{signalType: "user_feedback", names: matchedSignals.UserFeedback, defaultConfidence: 1.0, reason: "User feedback matched", addPath: true},
		{signalType: "language", names: matchedSignals.Language, defaultConfidence: 0.95, reason: "Language detected", addPath: true},
		{signalType: "context", names: matchedSignals.Context, defaultConfidence: 1.0, reason: "Context token count matched", addPath: true},
		{signalType: "structure", names: matchedSignals.Structure, defaultConfidence: 1.0, reason: "Structure rule matched", addPath: true},
		{signalType: "complexity", names: matchedSignals.Complexity, defaultConfidence: 0.9, reason: "Complexity level matched", addPath: true},
		{signalType: "modality", names: matchedSignals.Modality, defaultConfidence: 1.0, reason: "Modality signal matched", addPath: true},
		{signalType: "authz", names: matchedSignals.Authz, defaultConfidence: 1.0, reason: "Authorization signal matched", addPath: true},
		{signalType: "jailbreak", names: matchedSignals.Jailbreak, defaultConfidence: 1.0, reason: "Jailbreak signal matched", addPath: true},
		{signalType: "pii", names: matchedSignals.PII, defaultConfidence: 1.0, reason: "PII signal matched", addPath: true},
		{signalType: "kb", names: matchedSignals.KB, defaultConfidence: 1.0, reason: "Knowledge base signal matched", addPath: true},
		{signalType: "projection", names: matchedSignals.Projection, defaultConfidence: 1.0, reason: "Projection mapping matched", addPath: true},
	}
}

func addMatchedSignals(
	result *TestQueryResult,
	mapping topologySignalMapping,
	signalConfidences map[string]float64,
	signalValues map[string]float64,
) {
	for _, name := range mapping.names {
		confidence := matchedSignalConfidence(mapping.signalType, name, signalConfidences, mapping.defaultConfidence)
		result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
			Type:       mapping.signalType,
			Name:       name,
			Confidence: confidence,
			Value:      matchedSignalValue(mapping.signalType, name, signalValues),
			Reason:     mapping.reason,
		})
		if mapping.addPath {
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-%s-%s", mapping.signalType, name))
		}
	}
}

func matchedSignalConfidence(signalType string, name string, signalConfidences map[string]float64, fallback float64) float64 {
	if signalConfidences == nil {
		return fallback
	}
	if confidence, ok := signalConfidences[strings.ToLower(fmt.Sprintf("%s:%s", signalType, name))]; ok {
		return confidence
	}
	return fallback
}

func matchedSignalValue(signalType string, name string, signalValues map[string]float64) *float64 {
	if signalValues == nil {
		return nil
	}
	value, ok := signalValues[strings.ToLower(fmt.Sprintf("%s:%s", signalType, name))]
	if !ok {
		return nil
	}
	valueCopy := value
	return &valueCopy
}

func appendSignalGroupHighlights(result *TestQueryResult) {
	if len(result.MatchedSignals) == 0 {
		return
	}

	signalTypes := make(map[string]bool)
	for _, signal := range result.MatchedSignals {
		signalTypes[signal.Type] = true
	}
	for signalType := range signalTypes {
		result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-group-%s", signalType))
	}
}

func applyRouterDecision(result *TestQueryResult, routerResp *RouterEvalResponse) {
	if routerResp.DecisionResult != nil {
		result.MatchedDecision = routerResp.DecisionResult.DecisionName
		result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("decision-%s", routerResp.DecisionResult.DecisionName))
	}

	if routerResp.RoutingDecision == "" {
		return
	}

	result.MatchedDecision = routerResp.RoutingDecision
	result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("decision-%s", routerResp.RoutingDecision))
	if isSystemFallbackDecision(routerResp.RoutingDecision) {
		result.IsFallbackDecision = true
		result.FallbackReason = getFallbackReason(routerResp.RoutingDecision)
		result.HighlightedPath = append(result.HighlightedPath, "fallback-decision")
	}
}

func applyRecommendedModels(result *TestQueryResult, recommendedModels []string) {
	for _, recommendedModel := range recommendedModels {
		if recommendedModel == "" {
			continue
		}
		result.MatchedModels = append(result.MatchedModels, recommendedModel)
		result.HighlightedPath = append(
			result.HighlightedPath,
			fmt.Sprintf("model-%s", normalizeModelName(recommendedModel)),
		)
	}
}

func appendEvaluatedRulesFromConfig(result *TestQueryResult, configPath string) {
	parsedConfig, err := routerconfig.Parse(configPath)
	if err != nil || parsedConfig == nil {
		return
	}

	matchedSignalNames := buildMatchedSignalNameSet(result.MatchedSignals)
	for _, decision := range parsedConfig.IntelligentRouting.Decisions {
		if result.MatchedDecision != "" && decision.Name == result.MatchedDecision {
			continue
		}
		result.EvaluatedRules = append(result.EvaluatedRules, buildEvaluatedRule(decision, matchedSignalNames))
	}
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
