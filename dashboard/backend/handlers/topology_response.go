package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type topologySignalMapping struct {
	signalType string
	names      []string
	reason     string
	addPath    bool
}

// convertRouterResponse converts Router API response to TestQueryResult.
func convertRouterResponse(req TestQueryRequest, routerResp *RouterEvalResponse, configPath string) *TestQueryResult {
	result := newTestQueryResult(req)
	result.SignalErrorMatches = routerResp.SignalErrorMatches
	result.EvalTrace = routerResp.EvalTrace
	result.SignalErrors = routerResp.SignalErrors
	result.AppliedUnknownPolicies = routerResp.AppliedUnknownPolicies
	result.DecisionError = routerResp.DecisionError
	result.SelectedModel = routerResp.SelectedModel
	result.RecommendedModels = routerResp.RecommendedModels
	result.SelectionStatus = routerResp.SelectionStatus
	result.SelectionMethod = routerResp.SelectionMethod
	result.SelectionReason = routerResp.SelectionReason
	if routerResp.DecisionError != "" {
		result.IsAccurate = false
		result.HTTPStatus = http.StatusServiceUnavailable
		result.Warning = routerResp.DecisionError
	} else if len(routerResp.SignalErrors) > 0 {
		result.Warning = "Some signals failed; review the routing diagnostics."
	}

	appendMatchedSignals(result, routerResp)
	appendSignalGroupHighlights(result)
	applyRouterDecision(result, routerResp)
	if routerResp.SelectedModel != "" {
		applyRecommendedModels(result, []string{routerResp.SelectedModel})
	} else if routerResp.SelectionStatus == "" {
		// Older routers do not report selection metadata.
		applyRecommendedModels(result, routerResp.RecommendedModels)
	}
	appendEvaluatedRulesFromTrace(result, configPath, req.Model)

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
		addMatchedSignals(result, mapping, routerResp.SignalConfidences, routerResp.SignalValues, routerResp.SignalErrorMatches)
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
		{signalType: "keyword", names: matchedSignals.Keywords, reason: "Keyword rule matched", addPath: true},
		{signalType: "embedding", names: matchedSignals.Embeddings, reason: "Embedding similarity matched", addPath: true},
		{signalType: "domain", names: matchedSignals.Domains, reason: "Domain classification matched", addPath: true},
		{signalType: "fact_check", names: matchedSignals.FactCheck, reason: "Fact check signal matched"},
		{signalType: "preference", names: matchedSignals.Preferences, reason: "User preference matched", addPath: true},
		{signalType: "user_feedback", names: matchedSignals.UserFeedback, reason: "User feedback matched", addPath: true},
		{signalType: "language", names: matchedSignals.Language, reason: "Language detected", addPath: true},
		{signalType: "context", names: matchedSignals.Context, reason: "Context token count matched", addPath: true},
		{signalType: "structure", names: matchedSignals.Structure, reason: "Structure rule matched", addPath: true},
		{signalType: "complexity", names: matchedSignals.Complexity, reason: "Complexity level matched", addPath: true},
		{signalType: "modality", names: matchedSignals.Modality, reason: "Modality signal matched", addPath: true},
		{signalType: "authz", names: matchedSignals.Authz, reason: "Authorization signal matched", addPath: true},
		{signalType: "jailbreak", names: matchedSignals.Jailbreak, reason: "Jailbreak signal matched", addPath: true},
		{signalType: "pii", names: matchedSignals.PII, reason: "PII signal matched", addPath: true},
		{signalType: "kb", names: matchedSignals.KB, reason: "Knowledge base signal matched", addPath: true},
		{signalType: "conversation", names: matchedSignals.Conversation, reason: "Conversation structure signal matched", addPath: true},
		{signalType: "event", names: matchedSignals.Event, reason: "Event signal matched", addPath: true},
		{signalType: "projection", names: matchedSignals.Projection, reason: "Projection mapping matched", addPath: true},
	}
}

func addMatchedSignals(
	result *TestQueryResult,
	mapping topologySignalMapping,
	signalConfidences map[string]float64,
	signalValues map[string]float64,
	signalErrorMatches map[string]bool,
) {
	for _, name := range mapping.names {
		key := strings.ToLower(fmt.Sprintf("%s:%s", mapping.signalType, name))
		confidence, reported := signalConfidences[key]
		reported = reported && !signalErrorMatches[key]
		if !reported {
			confidence = 0
		}

		result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
			Type:                mapping.signalType,
			Name:                name,
			Confidence:          confidence,
			ConfidenceAvailable: &reported,
			Value:               matchedSignalValue(mapping.signalType, name, signalValues),
			Reason:              mapping.reason,
		})
		if mapping.addPath {
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-%s-%s", mapping.signalType, name))
		}
	}
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
		result.DecisionConfidence = routerResp.DecisionResult.Confidence
		result.DecisionConfidenceAvailable = routerResp.DecisionResult.ConfidenceAvailable
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

// Trace types decode display fields only; EvalTrace retains the complete router payload.
type topologyDecisionTrace struct {
	DecisionName string             `json:"decision_name"`
	State        string             `json:"state"`
	Matched      bool               `json:"matched"`
	RootTrace    *topologyRuleTrace `json:"root_trace"`
}

type topologyRuleTrace struct {
	NodeType   string               `json:"node_type"`
	SignalType string               `json:"signal_type"`
	SignalName string               `json:"signal_name"`
	Label      string               `json:"label"`
	Matched    bool                 `json:"matched"`
	Children   []*topologyRuleTrace `json:"children"`
}

func appendEvaluatedRulesFromTrace(result *TestQueryResult, configPath, requestModel string) {
	var traces []topologyDecisionTrace
	if json.Unmarshal(result.EvalTrace, &traces) != nil {
		return
	}
	priorities := make(map[string]int)
	if parsedConfig, err := routerconfig.Parse(configPath); err == nil && parsedConfig != nil {
		for _, decision := range topologyConfigForRequestModel(parsedConfig, requestModel).IntelligentRouting.Decisions {
			priorities[decision.Name] = decision.Priority
		}
	}
	for _, trace := range traces {
		rule := EvaluatedRule{
			DecisionName: trace.DecisionName,
			State:        trace.State,
			IsMatch:      trace.Matched,
			Priority:     priorities[trace.DecisionName],
			Conditions:   []string{},
			Expression:   topologyTraceExpression(trace.RootTrace),
		}
		if trace.RootTrace != nil {
			rule.RuleOperator = trace.RootTrace.NodeType
			for _, child := range trace.RootTrace.Children {
				rule.Conditions = append(rule.Conditions, topologyTraceExpression(child))
				rule.TotalCount++
				if child != nil && child.Matched {
					rule.MatchedCount++
				}
			}
		}
		result.EvaluatedRules = append(result.EvaluatedRules, rule)
	}
}

func topologyTraceExpression(node *topologyRuleTrace) string {
	if node == nil {
		return "Unavailable"
	}
	if node.NodeType == "leaf" {
		key := fmt.Sprintf("%s:%s", node.SignalType, node.SignalName)
		if node.Label != "" {
			key += ":" + node.Label
		}
		return key
	}
	if node.NodeType == "fallback" {
		return "Always matches"
	}
	children := make([]string, 0, len(node.Children))
	for _, child := range node.Children {
		children = append(children, topologyTraceExpression(child))
	}
	return fmt.Sprintf("%s(%s)", node.NodeType, strings.Join(children, ", "))
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
