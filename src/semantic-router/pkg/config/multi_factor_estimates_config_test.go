package config

import (
	"strings"
	"testing"
)

func TestValidateMultiFactorEstimates(t *testing.T) {
	tests := []struct {
		name       string
		estimates  *MultiFactorEstimatesConfig
		wantErrSub string
	}{
		{name: "off"},
		{name: "observe on the decision model", estimates: &MultiFactorEstimatesConfig{Mode: MultiFactorEstimatesModeObserve}},
		{name: "observe on a deployment", estimates: &MultiFactorEstimatesConfig{Mode: MultiFactorEstimatesModeObserve, Deployment: "decider", TimeoutMs: 300}},
		{name: "no mode", estimates: &MultiFactorEstimatesConfig{}, wantErrSub: "estimates.mode is required"},
		{name: "apply is not available", estimates: &MultiFactorEstimatesConfig{Mode: DecisionAdaptationModeApply}, wantErrSub: `estimates.mode must be "observe"`},
		{name: "padded deployment", estimates: &MultiFactorEstimatesConfig{Mode: MultiFactorEstimatesModeObserve, Deployment: " decider"}, wantErrSub: "deployment must name"},
		{name: "negative timeout", estimates: &MultiFactorEstimatesConfig{Mode: MultiFactorEstimatesModeObserve, TimeoutMs: -1}, wantErrSub: "timeout_ms"},
		{name: "timeout past the limit", estimates: &MultiFactorEstimatesConfig{Mode: MultiFactorEstimatesModeObserve, TimeoutMs: MaxSystemOneTimeoutMs + 1}, wantErrSub: "timeout_ms"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateDecisionAlgorithmConfig("estimated", nil, &AlgorithmConfig{
				Type:        DecisionAlgorithmMultiFactor,
				MultiFactor: &MultiFactorSelectionConfig{Estimates: test.estimates},
			})
			if test.wantErrSub == "" && err != nil {
				t.Fatalf("valid estimates rejected: %v", err)
			}
			if test.wantErrSub != "" && (err == nil || !strings.Contains(err.Error(), test.wantErrSub)) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErrSub)
			}
		})
	}
}

// The estimates ask a yes/no and a choice question, so the deployment that
// answers them must answer both; it starts with the configuration that names
// it.
func TestMultiFactorEstimatesDeployment(t *testing.T) {
	cfg := systemOneConfig()
	cfg.ModelDeployments["choices"] = ModelDeployment{
		Provider: SystemOneProvider, Schema: "vllm-sr/5.0", Endpoint: "https://api.example.com/v1/systemone", Model: "vllm-sr/d3-ultra",
		Capabilities: &DeploymentCapabilities{QuestionTypes: []string{"choice"}},
	}
	cfg.ModelDeployments["estimator"] = ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://estimator:8100"}
	decision := func(deployment string) Decision {
		return Decision{Name: "route", Algorithm: &AlgorithmConfig{
			Type:        DecisionAlgorithmMultiFactor,
			MultiFactor: &MultiFactorSelectionConfig{Estimates: &MultiFactorEstimatesConfig{Mode: MultiFactorEstimatesModeObserve, Deployment: deployment}},
		}}
	}
	if err := validateDecisionSystemOneQuestions(cfg, decision("decider")); err != nil {
		t.Fatal(err)
	}
	err := validateDecisionSystemOneQuestions(cfg, decision("choices"))
	if err == nil || !strings.Contains(err.Error(), "algorithm.multi_factor.estimates") || !strings.Contains(err.Error(), "noul") {
		t.Fatalf("a deployment that answers no yes/no question: %v", err)
	}
	if err := validateDecisionSystemOneQuestions(cfg, decision("missing")); err == nil {
		t.Fatal("an undeclared deployment was accepted")
	}

	cfg.Decisions = []Decision{decision("estimator")}
	if _, used := ModelRuntimeDeploymentsInUse(cfg)["estimator"]; !used {
		t.Fatal("the estimating deployment must start")
	}
	if cfg.MultiFactorEstimatesDeployment(MultiFactorEstimatesConfig{Deployment: "estimator"}) != "estimator" {
		t.Fatal("estimates ask the deployment they name")
	}
	static := &AlgorithmConfig{Type: DecisionAlgorithmStatic, MultiFactor: &MultiFactorSelectionConfig{Estimates: &MultiFactorEstimatesConfig{Mode: MultiFactorEstimatesModeObserve}}}
	if static.MultiFactorEstimates() != nil || (*AlgorithmConfig)(nil).MultiFactorEstimates() != nil {
		t.Fatal("only a multi_factor algorithm asks for estimates")
	}
}
