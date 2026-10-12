package extproc

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type recordingDecider struct {
	deployments []string
	requests    []modelservice.Request
	answer      modelservice.Answer
	cardCalls   int
	card        *modelservice.ModelCard
}

func (d *recordingDecider) Card(ctx context.Context, _ string) (modelservice.ModelCard, error) {
	if _, bounded := ctx.Deadline(); !bounded {
		panic("selector discovery must have a preparation deadline")
	}
	d.cardCalls++
	if d.card != nil {
		return *d.card, nil
	}
	return modelservice.ModelCard{Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice"}}, nil
}

func (d *recordingDecider) CurrentCard(string) (modelservice.ModelCard, bool) {
	if d.card != nil {
		return *d.card, true
	}
	return modelservice.ModelCard{Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice"}}, true
}

func (d *recordingDecider) Decide(_ context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	d.deployments = append(d.deployments, deployment)
	d.requests = append(d.requests, request)
	return modelservice.Response{Answers: map[string]modelservice.Answer{decisionSelectorQuestionID: d.answer}}, nil
}

// decisionSelectorConfig serves two models and lets the decision algorithm
// choose between them, on deployment or, without one, on the decision model.
func decisionSelectorConfig(t *testing.T, deployment string) *config.RouterConfig {
	t.Helper()
	selector := "        decision:\n          instructions: Which model should answer this request?\n" +
		"          candidates:\n            fast: Quick answers to routine requests\n            deep: Careful multi-step reasoning\n"
	deployments := "    deployments:\n      primary:\n        provider: model_runtime\n        artifact: vllm-sr/Vela-2.0-4B\n"
	if deployment != "" {
		selector += "          deployment: " + deployment + "\n"
		deployments += "      " + deployment + ":\n        provider: model_runtime\n" +
			"        artifact: vllm-sr/Decision-2.0-Kai-0.6B\n        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764\n"
	}
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
providers:
  defaults: {model: fast}
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: deep
      backend_refs: [{name: deep, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
routing:
  decisions:
    - name: choose
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: fast}, {model: deep}]
      algorithm:
        type: decision
` + selector + `global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
` + deployments))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func selectWithDecider(t *testing.T, cfg *config.RouterConfig, decider *recordingDecider) *selection.SelectionResult {
	t.Helper()
	decision := cfg.GetDecisionByName("choose")
	router := preparedSelectorRouter(t, cfg, decider)
	selector := router.selectorForDecisionMethod(router.getSelectionMethod(decision.Algorithm), decision.Algorithm, nil)
	result, err := selector.Select(t.Context(), &selection.SelectionContext{
		Query:           "Prove that the square root of two is irrational.",
		DecisionName:    decision.Name,
		CandidateModels: decision.ModelRefs,
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func preparedSelectorRouter(t *testing.T, cfg *config.RouterConfig, decider *recordingDecider) *OpenAIRouter {
	t.Helper()
	cards, err := prepareDecisionSelectionCards(cfg, decider)
	if err != nil {
		t.Fatal(err)
	}
	return &OpenAIRouter{Config: cfg, decisionDecider: decider, decisionCards: cards}
}

func TestDecisionSelectorDiscoveryIsPreparedOnceAndRejectsUnsupportedCard(t *testing.T) {
	cfg := decisionSelectorConfig(t, "")
	decider := &recordingDecider{answer: modelservice.Answer{Type: "choice", Choice: "fast", Probabilities: map[string]float64{"fast": .6, "deep": .4}}}
	router := preparedSelectorRouter(t, cfg, decider)
	decision := cfg.GetDecisionByName("choose")
	for range 2 {
		selector := router.newDecisionModelSelector(*decision.Algorithm.Decision)
		if _, err := selector.Select(t.Context(), &selection.SelectionContext{Query: "test", CandidateModels: decision.ModelRefs}); err != nil {
			t.Fatal(err)
		}
	}
	if decider.cardCalls != 1 {
		t.Fatalf("request path performed discovery: %d card calls", decider.cardCalls)
	}
	decider.card = &modelservice.ModelCard{Surfaces: []string{"decisions"}, QuestionTypes: []string{"score"}}
	if _, err := prepareDecisionSelectionCards(cfg, decider); err == nil {
		t.Fatal("unsupported selector admitted during generation preparation")
	}
}

func TestDecisionSelectorWithoutDeploymentAsksTheDecisionModel(t *testing.T) {
	cfg := decisionSelectorConfig(t, "")
	decider := &recordingDecider{answer: modelservice.Answer{
		Type: config.DecisionQuestionChoice, Choice: "deep",
		Probabilities: map[string]float64{"fast": 0.25, "deep": 0.75}, Confidence: 0.5,
	}}
	result := selectWithDecider(t, cfg, decider)
	const want = "primary"
	if len(decider.deployments) != 1 || decider.deployments[0] != want {
		t.Fatalf("the selector asked %v, want the decision model's deployment %s", decider.deployments, want)
	}
	if _, used := config.ModelRuntimeDeploymentsInUse(cfg)[want]; !used {
		t.Fatalf("the runtime must serve %s for the selector", want)
	}
	if result.SelectedModel != "deep" || result.Method != selection.MethodDecision || result.Score != 0.75 {
		t.Fatalf("result = %+v, want the decision model's choice deep", result)
	}
	if !strings.Contains(result.Reasoning, want) {
		t.Fatalf("reasoning %q must name the deployment that chose", result.Reasoning)
	}
	question := decider.requests[0].Questions[0]
	if question.ID != decisionSelectorQuestionID || question.Type != config.DecisionQuestionChoice ||
		len(question.Choices) != 2 || question.Choices[1] != (modelservice.Choice{Key: "deep", Description: "Careful multi-step reasoning"}) {
		t.Fatalf("question = %+v", question)
	}
}

func TestEvalPreviewsTheDecisionModelsChoice(t *testing.T) {
	cfg := decisionSelectorConfig(t, "")
	decider := &recordingDecider{answer: modelservice.Answer{
		Type: config.DecisionQuestionChoice, Choice: "deep",
		Probabilities: map[string]float64{"fast": 0.1, "deep": 0.9},
	}}
	router := preparedSelectorRouter(t, cfg, decider)
	result := router.SelectModelForEval(services.EvalModelSelectionInput{
		Context:  t.Context(),
		Decision: cfg.GetDecisionByName("choose"),
		Query:    "Prove that the square root of two is irrational.",
	})
	if result.Status != services.EvalSelectionSelected || result.SelectedModel != "deep" || result.Method != string(selection.MethodDecision) {
		t.Fatalf("Eval selection = %+v, want the decision model's choice deep", result)
	}
	if len(decider.deployments) != 1 || decider.deployments[0] != "primary" {
		t.Fatalf("Eval asked %v, want the decision model's deployment", decider.deployments)
	}
}

func TestDecisionSelectorKeepsItsOwnDeployment(t *testing.T) {
	cfg := decisionSelectorConfig(t, "kai")
	decider := &recordingDecider{answer: modelservice.Answer{
		Type: config.DecisionQuestionChoice, Choice: "fast",
		Probabilities: map[string]float64{"fast": 0.6, "deep": 0.4},
	}}
	if result := selectWithDecider(t, cfg, decider); result.SelectedModel != "fast" {
		t.Fatalf("selected %q, want the deployment's choice fast", result.SelectedModel)
	}
	if len(decider.deployments) != 1 || decider.deployments[0] != "kai" {
		t.Fatalf("the selector asked %v, want its own deployment kai", decider.deployments)
	}
}
