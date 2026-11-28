package extproc

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

// estimateDecider answers both estimate questions about each candidate, by
// the candidate's first line, and holds the candidate named slow until the
// call's deadline.
type estimateDecider struct {
	mu          sync.Mutex
	deployments []string
	requests    []modelservice.Request
	success     map[string]float64
}

func (d *estimateDecider) Card(ctx context.Context, _ string) (modelservice.ModelCard, error) {
	if _, bounded := ctx.Deadline(); !bounded {
		panic("estimate discovery must have a preparation deadline")
	}
	return modelservice.ModelCard{Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul", "choice"}}, nil
}

func (d *estimateDecider) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	d.mu.Lock()
	d.deployments = append(d.deployments, deployment)
	d.requests = append(d.requests, request)
	d.mu.Unlock()
	model, _, _ := strings.Cut(request.Parts[estimateCandidatePart], ":")
	model, _, _ = strings.Cut(model, "\n")
	if model == "slow" {
		<-ctx.Done()
		return modelservice.Response{}, ctx.Err()
	}
	return modelservice.Response{Answers: map[string]modelservice.Answer{
		candidateSuccessQuestionID: {Type: "noul", Noul: d.success[model]},
		outputLengthQuestionID: {Type: "choice", Choice: "256_to_1k", Probabilities: map[string]float64{
			"under_256": .25, "256_to_1k": .75, "1k_to_4k": 0, "over_4k": 0,
		}},
	}}, nil
}

func estimatesConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults: {model: qwen}
  models:
    - name: qwen
      description: General model with optional reasoning
      backend_refs: [{name: qwen, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: slow
      backend_refs: [{name: slow, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
recipes:
  - name: default
    decisions:
      - name: choose
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs:
          - {model: qwen, use_reasoning: false}
          - {model: qwen, use_reasoning: true, reasoning_effort: high, reasoning_description: "Slower, for multi-step problems"}
          - {model: slow}
        algorithm:
          type: multi_factor
          multi_factor:
            estimates:
              mode: observe
              timeout_ms: 50
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
`))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// A decision with estimates asks the decision model both questions about
// each candidate, one state per candidate that tells two efforts of a model
// apart, records the answers, and selects as it would without them. A
// candidate whose answer misses the deadline is marked so.
func TestMultiFactorEstimatesAskTheDecisionModelAboutEachCandidate(t *testing.T) {
	cfg := estimatesConfig(t)
	decision := cfg.GetDecisionByName("choose")
	decider := &estimateDecider{success: map[string]float64{"qwen": .6}}
	cards, err := prepareDecisionSelectionCards(cfg, decider)
	if err != nil {
		t.Fatal(err)
	}
	if _, prepared := cards["primary"]; !prepared {
		t.Fatal("the estimating deployment's card was not prepared")
	}
	router := &OpenAIRouter{Config: cfg, decisionDecider: decider, decisionCards: cards}
	selCtx := func() *selection.SelectionContext {
		return &selection.SelectionContext{Query: "If 3x + 7 = 22, what is x?", DecisionName: decision.Name, CandidateModels: decision.ModelRefs}
	}
	selector := router.selectorForDecisionMethod(router.getSelectionMethod(decision.Algorithm), decision.Algorithm, nil)
	result, err := selector.Select(t.Context(), selCtx())
	if err != nil {
		t.Fatal(err)
	}
	base, err := selection.NewMultiFactorSelector(buildMultiFactorSelectionConfig(decision.Algorithm.MultiFactor)).Select(t.Context(), selCtx())
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedModel != base.SelectedModel {
		t.Fatalf("estimates changed the selection: %s, want %s", result.SelectedModel, base.SelectedModel)
	}

	if len(decider.requests) != 3 {
		t.Fatalf("asked %d states, want one per candidate", len(decider.requests))
	}
	described := map[string]bool{}
	for index, request := range decider.requests {
		if decider.deployments[index] != "primary" || request.Parts[estimateRequestPart] != "If 3x + 7 = 22, what is x?" || len(request.Questions) != 2 ||
			request.Questions[0].ID != candidateSuccessQuestionID || request.Questions[1].ID != outputLengthQuestionID {
			t.Fatalf("state %d asked %s: %+v", index, decider.deployments[index], request)
		}
		described[request.Parts[estimateCandidatePart]] = true
	}
	for _, want := range []string{
		"qwen: General model with optional reasoning\nReasoning: off",
		"qwen: General model with optional reasoning\nReasoning: on, effort high\nSlower, for multi-step problems",
		"slow\nReasoning: off",
	} {
		if !described[want] {
			t.Fatalf("no state describes %q: %v", want, described)
		}
	}

	estimates := result.MultiFactor.Estimates
	if estimates == nil || estimates.Mode != config.MultiFactorEstimatesModeObserve || len(estimates.Candidates) != 3 {
		t.Fatalf("estimates = %+v", estimates)
	}
	for _, estimate := range estimates.Candidates[:2] {
		if estimate.Status != selectiontrace.EstimateAvailable || *estimate.SuccessProbability != .6 ||
			*estimate.ExpectedOutputTokens != 512 || estimate.Deployment != "primary" {
			t.Fatalf("estimate = %+v", estimate)
		}
	}
	if late := estimates.Candidates[2]; late.Status != selectiontrace.EstimateDeadlineExceeded || late.SuccessProbability != nil {
		t.Fatalf("late estimate = %+v", late)
	}
}

// A deployment that is not ready leaves every candidate unestimated, and the
// selection goes on.
func TestMultiFactorEstimatesWithoutACard(t *testing.T) {
	cfg := estimatesConfig(t)
	decision := cfg.GetDecisionByName("choose")
	router := &OpenAIRouter{Config: cfg, decisionDecider: &estimateDecider{}}
	result, err := router.newDecisionMultiFactorSelector(decision.Algorithm.MultiFactor).Select(t.Context(), &selection.SelectionContext{
		Query: "test", CandidateModels: decision.ModelRefs,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, estimate := range result.MultiFactor.Estimates.Candidates {
		if estimate.Status != selectiontrace.EstimateUnavailable || estimate.Reason != "not_ready" {
			t.Fatalf("estimate = %+v", estimate)
		}
	}
}
