package handlers

import (
	"fmt"
	"strings"
)

type topologySignalMapping struct {
	signalType        string
	names             []string
	defaultConfidence float64
	reason            string
	addPath           bool
}

// convertRouterResponse converts Router API response to TestQueryResult. The
// router's eval_trace is the source of truth for decision/rule state; the
// flat config-derived reconstruction only runs when the router did not
// return a trace, and can never mark the result as accurate.
func convertRouterResponse(req TestQueryRequest, routerResp *RouterEvalResponse, configPath string) *TestQueryResult {
	result := newTestQueryResult(req)

	appendMatchedSignals(result, routerResp)
	appendSignalGroupHighlights(result)
	applyRouterDecision(result, routerResp)
	applyRecommendedModels(result, routerResp.RecommendedModels)
	verifyRecipeScope(result, routerResp, configPath, req.Model)

	if !applyEvalTrace(result, routerResp) {
		result.IsAccurate = false
		appendWarning(result, "Router did not return an eval trace; rule display is a derived approximation and may be incomplete.")
		appendEvaluatedRulesFromConfig(result, configPath, req.Model)
	}

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
		{signalType: "conversation", names: matchedSignals.Conversation, defaultConfidence: 1.0, reason: "Conversation structure signal matched", addPath: true},
		{signalType: "event", names: matchedSignals.Event, defaultConfidence: 1.0, reason: "Event signal matched", addPath: true},
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
		result.Algorithm = routerResp.DecisionResult.Algorithm
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
