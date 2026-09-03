package recipe

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCompareEvalResponseNormalizesDefaultRecipeForTrace(t *testing.T) {
	probes := []ProbeDetail{
		{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{Decision: "decision-a"}}},
		{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{Decision: "decision-b", Recipe: "default"}}},
	}
	response := json.RawMessage(`{
  "recipe":"default",
  "routing_decision":"decision-a",
  "decision_result":{"decision_name":"decision-a"},
  "eval_trace":[
    {"decision_name":"decision-a","matched":true},
    {"decision_name":"decision-b","matched":false}
  ]
}`)

	_, checks, failures, err := compareEvalResponse(response, probes[0], probes)
	if err != nil {
		t.Fatalf("compareEvalResponse(): %v", err)
	}
	if !checks.Recipe || !checks.Trace || len(failures) != 0 {
		t.Fatalf("checks = %#v, failures = %v", checks, failures)
	}
}

func TestAllowedDecisionsNormalizesRecipeNames(t *testing.T) {
	probes := []ProbeDetail{
		{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{Decision: "implicit", Recipe: ""}}},
		{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{Decision: "explicit", Recipe: " default "}}},
		{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{Decision: "named", Recipe: "balanced"}}},
	}

	allowed := allowedDecisions(probes, "default")
	if _, found := allowed["implicit"]; !found {
		t.Fatal("implicit default decision is missing")
	}
	if _, found := allowed["explicit"]; !found {
		t.Fatal("explicit default decision is missing")
	}
	if _, found := allowed["named"]; found {
		t.Fatal("named recipe decision must not be included")
	}
	if len(allowed) != 2 {
		t.Fatalf("allowed decisions = %v, want exactly the two default decisions", allowed)
	}

	_, passed, failures := compareTrace(
		[]evalTrace{{DecisionName: "implicit", Matched: true}},
		"implicit",
		allowed,
	)
	if passed || len(failures) == 0 {
		t.Fatal("trace without the sibling default decision must fail strict validation")
	}

	_, passed, failures = compareTrace(
		[]evalTrace{
			{DecisionName: "implicit", Matched: true},
			{DecisionName: "explicit", Matched: false},
			{DecisionName: "named", Matched: false},
		},
		"implicit",
		allowed,
	)
	if passed || len(failures) == 0 {
		t.Fatal("trace containing a named-recipe decision must fail strict validation")
	}
}

func TestCompareEvalResponseDoesNotMislabelCandidatesAsFinalModel(t *testing.T) {
	probe := ProbeDetail{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{Decision: "route-a"}}}
	response := json.RawMessage(`{
  "requested_model":"vllm-sr/auto",
  "recipe":"default",
  "routing_decision":"route-a",
  "decision_result":{"decision_name":"route-a"},
  "recommended_models":["candidate-a","candidate-b"],
  "eval_trace":[{"decision_name":"route-a","matched":true}]
}`)

	actual, checks, failures, err := compareEvalResponse(response, probe, []ProbeDetail{probe})
	if err != nil {
		t.Fatalf("compareEvalResponse(): %v", err)
	}
	if actual.Model != "" || actual.RequestedModel != "vllm-sr/auto" {
		t.Fatalf("actual = %#v, want no fabricated final model", actual)
	}
	if !checks.Selection || len(failures) != 0 {
		t.Fatalf("selection without an expected algorithm must be ignored: checks=%#v failures=%v", checks, failures)
	}
}

func TestCompareEvalResponsePreservesSelectionSemantics(t *testing.T) {
	probe := ProbeDetail{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{Decision: "route-a"}}}
	response := json.RawMessage(`{
  "requested_model":"vllm-sr/auto",
  "selected_model":"judge-model",
  "selection_status":"planned_final",
  "selection_method":"fusion",
  "selection_reason":"configured final-output model",
  "recipe":"default",
  "routing_decision":"route-a",
  "decision_result":{"decision_name":"route-a"},
  "eval_trace":[{"decision_name":"route-a","matched":true}]
}`)

	actual, _, _, err := compareEvalResponse(response, probe, []ProbeDetail{probe})
	if err != nil {
		t.Fatalf("compareEvalResponse(): %v", err)
	}
	if actual.Model != "judge-model" || actual.SelectionStatus != "planned_final" ||
		actual.SelectionMethod != "fusion" || actual.SelectionReason == "" {
		t.Fatalf("actual selection = %#v", actual)
	}
}

func TestCompareEvalResponseRejectsInvalidSelectionContracts(t *testing.T) {
	testCases := []struct {
		name              string
		algorithm         string
		selectedModel     string
		selectionStatus   string
		selectionMethod   string
		recommendedModels []string
		wantFailure       string
	}{
		{
			name: "missing status", algorithm: "static", selectedModel: "candidate-a",
			selectionMethod: "static", recommendedModels: []string{"candidate-a"},
			wantFailure: "selection_status is missing",
		},
		{
			name: "wrong status", algorithm: "static", selectedModel: "candidate-a",
			selectionStatus: "planned_final", selectionMethod: "static", recommendedModels: []string{"candidate-a"},
			wantFailure: `selection_status="planned_final", want one of [selected execution_required] for algorithm "static"`,
		},
		{
			name: "missing method", algorithm: "static", selectedModel: "candidate-a",
			selectionStatus: "selected", recommendedModels: []string{"candidate-a"},
			wantFailure: "selection_method is missing",
		},
		{
			name: "wrong method", algorithm: "static", selectedModel: "candidate-a",
			selectionStatus: "selected", selectionMethod: "fusion", recommendedModels: []string{"candidate-a"},
			wantFailure: `selection_method="fusion", want "static"`,
		},
		{
			name: "selected model missing", algorithm: "latency_aware",
			selectionStatus: "selected", selectionMethod: "latency_aware", recommendedModels: []string{"candidate-a"},
			wantFailure: "selected_model is required for selected",
		},
		{
			name: "selected model not a candidate", algorithm: "multi_factor", selectedModel: "not-a-candidate",
			selectionStatus: "selected", selectionMethod: "multi_factor", recommendedModels: []string{"candidate-a"},
			wantFailure: "selected_model is not a recommended decision candidate",
		},
		{
			name: "confidence fabricates final model", algorithm: "confidence", selectedModel: "candidate-a",
			selectionStatus: "execution_required", selectionMethod: "confidence", recommendedModels: []string{"candidate-a"},
			wantFailure: "execution_required must not fabricate selected_model",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			probe, response := selectionContractFixture(
				testCase.algorithm,
				testCase.selectedModel,
				testCase.selectionStatus,
				testCase.selectionMethod,
				testCase.recommendedModels,
			)
			_, checks, failures, err := compareEvalResponse(response, probe, []ProbeDetail{probe})
			if err != nil {
				t.Fatalf("compareEvalResponse(): %v", err)
			}
			if checks.Selection || len(failures) == 0 {
				t.Fatalf("selection check unexpectedly passed: checks=%#v failures=%v", checks, failures)
			}
			if got := strings.Join(failures, "\n"); !strings.Contains(got, testCase.wantFailure) {
				t.Fatalf("failures = %v, want %q", failures, testCase.wantFailure)
			}
		})
	}
}

func TestCompareEvalResponseAcceptsAlgorithmSelectionContracts(t *testing.T) {
	testCases := []struct {
		name              string
		algorithm         string
		selectedModel     string
		selectionStatus   string
		selectionMethod   string
		recommendedModels []string
	}{
		{name: "static", algorithm: "static", selectedModel: "candidate-a", selectionStatus: "selected", selectionMethod: "static", recommendedModels: []string{"candidate-a"}},
		{name: "static single method", algorithm: "static", selectedModel: "candidate-a", selectionStatus: "selected", selectionMethod: "single", recommendedModels: []string{"candidate-a"}},
		{name: "multi factor", algorithm: "multi_factor", selectedModel: "candidate-b", selectionStatus: "selected", selectionMethod: "multi_factor", recommendedModels: []string{"candidate-a", "candidate-b"}},
		{name: "latency aware", algorithm: "latency_aware", selectedModel: "candidate-a", selectionStatus: "selected", selectionMethod: "latency_aware", recommendedModels: []string{"candidate-a"}},
		{name: "workflows", algorithm: "workflows", selectedModel: "workflow-final", selectionStatus: "planned_final", selectionMethod: "workflows", recommendedModels: []string{"candidate-a"}},
		{name: "fusion", algorithm: "fusion", selectedModel: "fusion-final", selectionStatus: "planned_final", selectionMethod: "fusion", recommendedModels: []string{"candidate-a"}},
		{name: "remom", algorithm: "remom", selectedModel: "remom-final", selectionStatus: "planned_final", selectionMethod: "remom", recommendedModels: []string{"candidate-a"}},
		{name: "confidence", algorithm: "confidence", selectionStatus: "execution_required", selectionMethod: "confidence", recommendedModels: []string{"candidate-a", "candidate-b"}},
		{name: "static deferred by runtime policy", algorithm: "static", selectionStatus: "execution_required", selectionMethod: "static", recommendedModels: []string{"candidate-a", "candidate-b"}},
		{name: "multi factor deferred by runtime policy", algorithm: "multi_factor", selectionStatus: "execution_required", selectionMethod: "multi_factor", recommendedModels: []string{"candidate-a", "candidate-b"}},
		{name: "latency aware deferred by runtime policy", algorithm: "latency_aware", selectionStatus: "execution_required", selectionMethod: "latency_aware", recommendedModels: []string{"candidate-a", "candidate-b"}},
		{name: "workflow requires execution", algorithm: "workflows", selectionStatus: "execution_required", selectionMethod: "workflows", recommendedModels: []string{"candidate-a"}},
		{name: "fusion requires execution", algorithm: "fusion", selectionStatus: "execution_required", selectionMethod: "fusion", recommendedModels: []string{"candidate-a"}},
		{name: "remom requires execution", algorithm: "remom", selectionStatus: "execution_required", selectionMethod: "remom", recommendedModels: []string{"candidate-a"}},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			probe, response := selectionContractFixture(
				testCase.algorithm,
				testCase.selectedModel,
				testCase.selectionStatus,
				testCase.selectionMethod,
				testCase.recommendedModels,
			)
			_, checks, failures, err := compareEvalResponse(response, probe, []ProbeDetail{probe})
			if err != nil {
				t.Fatalf("compareEvalResponse(): %v", err)
			}
			if !checks.Selection || len(failures) != 0 {
				t.Fatalf("selection check failed: checks=%#v failures=%v", checks, failures)
			}
		})
	}
}

func selectionContractFixture(algorithm, selectedModel, status, method string, recommendedModels []string) (ProbeDetail, json.RawMessage) {
	probe := ProbeDetail{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{
		Decision:  "route-a",
		Algorithm: algorithm,
	}}}
	encoded, err := json.Marshal(evalResponse{
		SelectedModel:     selectedModel,
		SelectionStatus:   status,
		SelectionMethod:   method,
		Recipe:            "default",
		RoutingDecision:   "route-a",
		RecommendedModels: recommendedModels,
		DecisionResult: evalDecisionResult{
			DecisionName: "route-a",
			Algorithm:    algorithm,
		},
		EvalTrace: []evalTrace{{DecisionName: "route-a", Matched: true}},
	})
	if err != nil {
		panic(err)
	}
	return probe, encoded
}
