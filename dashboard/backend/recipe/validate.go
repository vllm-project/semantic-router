package recipe

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

type evalResponse struct {
	RequestedModel    string             `json:"requested_model"`
	SelectedModel     string             `json:"selected_model"`
	FinalModel        string             `json:"final_model"`
	SelectionStatus   string             `json:"selection_status"`
	SelectionMethod   string             `json:"selection_method"`
	SelectionReason   string             `json:"selection_reason"`
	Recipe            string             `json:"recipe"`
	DecisionResult    evalDecisionResult `json:"decision_result"`
	EvalTrace         []evalTrace        `json:"eval_trace"`
	RecommendedModels []string           `json:"recommended_models"`
	RoutingDecision   string             `json:"routing_decision"`
}

type evalDecisionResult struct {
	DecisionName   string              `json:"decision_name"`
	Algorithm      string              `json:"algorithm"`
	Plugins        []string            `json:"plugins"`
	MatchedSignals map[string][]string `json:"matched_signals"`
}

type evalTrace struct {
	DecisionName string `json:"decision_name"`
	Matched      bool   `json:"matched"`
}

func compareEvalResponse(raw json.RawMessage, probe ProbeDetail, allProbes []ProbeDetail) (ActualOutcome, ValidationChecks, []string, error) {
	var response evalResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return ActualOutcome{}, ValidationChecks{}, nil, fmt.Errorf("decode JSON: %w", err)
	}
	if strings.TrimSpace(response.RoutingDecision) == "" && strings.TrimSpace(response.DecisionResult.DecisionName) == "" {
		return ActualOutcome{}, ValidationChecks{}, nil, errors.New("response has no routing decision")
	}

	actualDecision := strings.TrimSpace(response.RoutingDecision)
	if actualDecision == "" {
		actualDecision = strings.TrimSpace(response.DecisionResult.DecisionName)
	}
	expectedRecipe := normalizeExpectedRecipe(probe.Expected.Recipe)
	actualPlugins := cleanStrings(response.DecisionResult.Plugins)
	actualModels := cleanStrings(response.RecommendedModels)
	actualSignals := nonNilSignalMap(response.DecisionResult.MatchedSignals)
	traceDecisions, tracePassed, traceFailures := compareTrace(response.EvalTrace, probe.Expected.Decision, allowedDecisions(allProbes, expectedRecipe))

	pluginsPassed, pluginFailures := compareStringAssertion(
		"plugin",
		probe.Expected.Plugins,
		probe.Expected.ForbiddenPlugins,
		actualPlugins,
		probe.Expected.PluginMatch,
	)
	signalsPassed, signalFailures := compareSignalAssertion(
		probe.Expected.Signals,
		probe.Expected.ForbiddenSignals,
		actualSignals,
		probe.Expected.SignalMatch,
	)
	aliasPassed := aliasMatches(probe.Expected.Alias, actualModels)
	selectionPassed, selectionFailures := compareEvalSelection(
		probe.Expected.Algorithm,
		response.SelectedModel,
		response.SelectionStatus,
		response.SelectionMethod,
		actualModels,
	)
	checks := ValidationChecks{
		Decision:  actualDecision == probe.Expected.Decision,
		Model:     probe.Model == "" || strings.TrimSpace(response.RequestedModel) == probe.Model,
		Recipe:    strings.TrimSpace(response.Recipe) == expectedRecipe,
		Algorithm: probe.Expected.Algorithm == "" || strings.TrimSpace(response.DecisionResult.Algorithm) == probe.Expected.Algorithm,
		Selection: selectionPassed,
		Plugins:   pluginsPassed,
		Signals:   signalsPassed,
		Alias:     aliasPassed,
		Trace:     tracePassed,
	}

	failures := []string{}
	if !checks.Decision {
		failures = append(failures, fmt.Sprintf("decision: got %q, want %q", actualDecision, probe.Expected.Decision))
	}
	if !checks.Model {
		failures = append(failures, fmt.Sprintf("requested model: got %q, want %q", response.RequestedModel, probe.Model))
	}
	if !checks.Recipe {
		failures = append(failures, fmt.Sprintf("recipe: got %q, want %q", response.Recipe, expectedRecipe))
	}
	if !checks.Algorithm {
		failures = append(failures, fmt.Sprintf("algorithm: got %q, want %q", response.DecisionResult.Algorithm, probe.Expected.Algorithm))
	}
	failures = append(failures, pluginFailures...)
	failures = append(failures, signalFailures...)
	failures = append(failures, selectionFailures...)
	if !checks.Alias {
		failures = append(failures, fmt.Sprintf("recommended models %v do not contain expected alias %q", actualModels, probe.Expected.Alias))
	}
	failures = append(failures, traceFailures...)

	return ActualOutcome{
		Decision:          actualDecision,
		Model:             firstNonEmpty(response.FinalModel, response.SelectedModel),
		RequestedModel:    strings.TrimSpace(response.RequestedModel),
		SelectionStatus:   strings.TrimSpace(response.SelectionStatus),
		SelectionMethod:   strings.TrimSpace(response.SelectionMethod),
		SelectionReason:   strings.TrimSpace(response.SelectionReason),
		Recipe:            strings.TrimSpace(response.Recipe),
		Algorithm:         strings.TrimSpace(response.DecisionResult.Algorithm),
		Plugins:           actualPlugins,
		RecommendedModels: actualModels,
		MatchedSignals:    actualSignals,
		TraceDecisions:    traceDecisions,
	}, checks, failures, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}

func compareEvalSelection(algorithm, selectedModel, status, method string, recommendedModels []string) (bool, []string) {
	algorithm = strings.TrimSpace(algorithm)
	if algorithm == "" {
		return true, nil
	}

	expectedStatuses := expectedSelectionStatuses(algorithm)
	selectedModel = strings.TrimSpace(selectedModel)
	status = strings.TrimSpace(status)
	method = strings.TrimSpace(method)

	failures := []string{}
	failures = append(failures, selectionStatusFailures(algorithm, expectedStatuses, status)...)
	failures = append(failures, selectionMethodFailures(algorithm, method)...)
	failures = append(failures, selectedModelFailures(status, selectedModel, recommendedModels)...)
	return len(failures) == 0, failures
}

func expectedSelectionStatuses(algorithm string) []string {
	return map[string][]string{
		"static":        {"selected", "execution_required"},
		"multi_factor":  {"selected", "execution_required"},
		"latency_aware": {"selected", "execution_required"},
		"workflows":     {"planned_final", "execution_required"},
		"fusion":        {"planned_final", "execution_required"},
		"remom":         {"planned_final", "execution_required"},
		"confidence":    {"execution_required"},
	}[algorithm]
}

func selectionStatusFailures(algorithm string, expectedStatuses []string, status string) []string {
	if status == "" {
		return []string{"selection_status is missing"}
	}
	if len(expectedStatuses) > 0 && !contains(expectedStatuses, status) {
		return []string{fmt.Sprintf(
			"selection_status=%q, want one of %v for algorithm %q",
			status,
			expectedStatuses,
			algorithm,
		)}
	}
	return nil
}

func selectionMethodFailures(algorithm, method string) []string {
	if method == "" {
		return []string{"selection_method is missing"}
	}
	if method != algorithm && (algorithm != "static" || method != "single") {
		return []string{fmt.Sprintf(
			"selection_method=%q, want %q",
			method,
			algorithm,
		)}
	}
	return nil
}

func selectedModelFailures(status, selectedModel string, recommendedModels []string) []string {
	failures := []string{}
	if (status == "selected" || status == "planned_final") && selectedModel == "" {
		failures = append(failures, fmt.Sprintf("selected_model is required for %s", status))
	}
	if status == "selected" && selectedModel != "" && !contains(recommendedModels, selectedModel) {
		failures = append(failures, "selected_model is not a recommended decision candidate")
	}
	if status == "execution_required" && selectedModel != "" {
		failures = append(failures, "execution_required must not fabricate selected_model")
	}
	return failures
}

func compareStringAssertion(label string, expected, forbidden, actual []string, matchMode string) (bool, []string) {
	expectedSet := stringSet(expected)
	forbiddenSet := stringSet(forbidden)
	actualSet := stringSet(actual)
	failures := []string{}
	for _, item := range sortedDifference(expectedSet, actualSet) {
		failures = append(failures, fmt.Sprintf("missing expected %s %q", label, item))
	}
	if matchMode == "exact" {
		for _, item := range sortedDifference(actualSet, expectedSet) {
			failures = append(failures, fmt.Sprintf("unexpected %s %q", label, item))
		}
	}
	for _, item := range sortedIntersection(forbiddenSet, actualSet) {
		failures = append(failures, fmt.Sprintf("forbidden %s matched %q", label, item))
	}
	return len(failures) == 0, failures
}

func compareSignalAssertion(expected, forbidden, actual map[string][]string, matchMode string) (bool, []string) {
	expectedSet := signalSet(expected)
	forbiddenSet := signalSet(forbidden)
	actualSet := signalSet(actual)
	failures := []string{}
	for _, item := range sortedDifference(expectedSet, actualSet) {
		failures = append(failures, fmt.Sprintf("missing expected signal %q", item))
	}
	if matchMode == "exact" {
		for _, item := range sortedDifference(actualSet, expectedSet) {
			failures = append(failures, fmt.Sprintf("unexpected signal %q", item))
		}
	}
	for _, item := range sortedIntersection(forbiddenSet, actualSet) {
		failures = append(failures, fmt.Sprintf("forbidden signal matched %q", item))
	}
	return len(failures) == 0, failures
}

func compareTrace(trace []evalTrace, expectedDecision string, allowed map[string]struct{}) ([]string, bool, []string) {
	if len(trace) == 0 {
		return []string{}, false, []string{"eval trace is missing or empty"}
	}
	decisions := make([]string, 0, len(trace))
	seen := map[string]struct{}{}
	matchedExpected := 0
	failures := []string{}
	for index, item := range trace {
		name := strings.TrimSpace(item.DecisionName)
		if name == "" {
			failures = append(failures, fmt.Sprintf("eval trace item %d has no decision name", index))
			continue
		}
		decisions = append(decisions, name)
		if _, duplicate := seen[name]; duplicate {
			failures = append(failures, fmt.Sprintf("eval trace contains duplicate decision %q", name))
		}
		seen[name] = struct{}{}
		if name == expectedDecision && item.Matched {
			matchedExpected++
		}
	}
	if matchedExpected != 1 {
		failures = append(failures, fmt.Sprintf("eval trace must contain exactly one matched %q decision; got %d", expectedDecision, matchedExpected))
	}
	if !equalSets(seen, allowed) {
		failures = append(failures, fmt.Sprintf("eval trace decisions %v do not match selected recipe decisions %v", sortedKeys(seen), sortedKeys(allowed)))
	}
	return decisions, len(failures) == 0, failures
}

func allowedDecisions(probes []ProbeDetail, expectedRecipe string) map[string]struct{} {
	allowed := map[string]struct{}{}
	expectedRecipe = normalizeExpectedRecipe(expectedRecipe)
	for _, probe := range probes {
		if normalizeExpectedRecipe(probe.Expected.Recipe) == expectedRecipe {
			allowed[probe.Expected.Decision] = struct{}{}
		}
	}
	return allowed
}

func normalizeExpectedRecipe(value string) string {
	if value = strings.TrimSpace(value); value != "" {
		return value
	}
	return "default"
}

func aliasMatches(expected string, actual []string) bool {
	if expected == "" {
		return true
	}
	if len(actual) == 1 {
		return actual[0] == expected
	}
	return contains(actual, expected)
}

func stringSet(values []string) map[string]struct{} {
	result := map[string]struct{}{}
	for _, value := range values {
		if normalized := strings.TrimSpace(value); normalized != "" {
			result[normalized] = struct{}{}
		}
	}
	return result
}

func signalSet(values map[string][]string) map[string]struct{} {
	result := map[string]struct{}{}
	for signalType, names := range values {
		for _, name := range names {
			result[signalType+":"+name] = struct{}{}
		}
	}
	return result
}

func sortedDifference(left, right map[string]struct{}) []string {
	result := []string{}
	for value := range left {
		if _, found := right[value]; !found {
			result = append(result, value)
		}
	}
	sort.Strings(result)
	return result
}

func sortedIntersection(left, right map[string]struct{}) []string {
	result := []string{}
	for value := range left {
		if _, found := right[value]; found {
			result = append(result, value)
		}
	}
	sort.Strings(result)
	return result
}

func equalSets(left, right map[string]struct{}) bool {
	if len(left) != len(right) {
		return false
	}
	for value := range left {
		if _, found := right[value]; !found {
			return false
		}
	}
	return true
}

func cleanStrings(values []string) []string {
	result := []string{}
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			result = append(result, value)
		}
	}
	return result
}
