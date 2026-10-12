package classification

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type observedDecisionServices struct {
	cardServices
	ready    bool
	requests []modelservice.Request
}

func (*observedDecisionServices) Card(context.Context, string) (modelservice.ModelCard, error) {
	panic("attached decision preparation must not wait for discovery")
}

func (s *observedDecisionServices) CurrentCard(string) (modelservice.ModelCard, bool) {
	return s.card, s.ready
}

func (s *observedDecisionServices) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	s.requests = append(s.requests, request)
	answers := map[string]modelservice.Answer{}
	for _, q := range request.Questions {
		answers[q.ID] = modelservice.Answer{Type: q.Type, Noul: .9, InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: answers}, nil
}

func attachedDecisionModels(t *testing.T, explicit bool) (*classifierModelRuntime, *observedDecisionServices) {
	t.Helper()
	source := setQuestionOnTheDecisionModel
	if explicit {
		source = strings.Replace(source, "routing:\n", "routing:\n  model_bindings:\n    pii_classifier: {deployment: primary, contract: decision.v1}\n", 1)
	}
	cfg, err := config.ParseYAMLBytes([]byte(source))
	if err != nil {
		t.Fatal(err)
	}
	cfg.ModelDeployments["primary"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://attached.invalid"}
	services := &observedDecisionServices{}
	models, err := newClassifierModelRuntime(cfg, RecipeRuntimeOptions{Runtime: serving.New(services, nil)})
	if err != nil {
		t.Fatal(err)
	}
	return models, services
}

func decisionObservation(types ...string) modelservice.ModelCard {
	return modelservice.ModelCard{ID: "remote", Surfaces: []string{"decisions"}, QuestionTypes: types}
}

func TestAttachedDecisionQuestionsRecoverWithoutDiscovery(t *testing.T) {
	models, services := attachedDecisionModels(t, false)
	c := &Classifier{Config: models.cfg, models: models}
	if err := c.prepareDecisionSignals(); err != nil {
		t.Fatalf("offline attached provider blocked preparation: %v", err)
	}
	evaluate := func() *SignalResults {
		r := &SignalResults{SignalErrors: map[string]string{}, SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}}
		c.evaluateDecisionDeployment(t.Context(), r, &sync.Mutex{}, "needs tools", "primary", models.cfg.DecisionRules)
		return r
	}
	if result := evaluate(); result.SignalErrors["decision:needs"] == "" || len(services.requests) != 0 {
		t.Fatalf("offline model must remain unknown without a native call: %+v", result)
	}
	services.card, services.ready = decisionObservation("noul"), true
	if result := evaluate(); len(result.SignalErrors) != 0 || len(services.requests) != 1 || len(services.requests[0].Questions) != 2 {
		t.Fatalf("recovered model did not compose Set from Noul: %+v requests=%+v", result, services.requests)
	}
	for _, q := range services.requests[0].Questions {
		if q.Type != "noul" || !q.Truncate {
			t.Fatalf("recovery bypassed compiled task or changed input contract: %+v", q)
		}
	}
	services.card = decisionObservation("choice")
	if result := evaluate(); result.SignalErrors["decision:needs"] == "" || len(services.requests) != 1 {
		t.Fatal("changed attached capabilities reused an old card or bypassed compilation")
	}
	if err := c.prepareDecisionSignals(); !errors.Is(err, modelservice.ErrRejected) {
		t.Fatalf("known unsupported capability must reject preparation: %v", err)
	}
}

func TestAttachedDecisionSpanIsNeverComposedOrSilentlyAccepted(t *testing.T) {
	models, services := attachedDecisionModels(t, false)
	models.cfg.DecisionRules[0].Question.Type = "span"
	c := &Classifier{Config: models.cfg, models: models}
	if err := c.prepareDecisionSignals(); err != nil {
		t.Fatal(err)
	}
	services.card, services.ready = decisionObservation("noul"), true
	if err := c.prepareDecisionSignals(); !errors.Is(err, modelservice.ErrRejected) {
		t.Fatalf("Noul cannot provide locations: %v", err)
	}
	services.card = decisionObservation("span")
	if err := c.prepareDecisionSignals(); err != nil {
		t.Fatalf("native Span capability was lost: %v", err)
	}
}

func TestAttachedGenericTaskBindingRetainsPrivacyCompositionAfterRecovery(t *testing.T) {
	models, services := attachedDecisionModels(t, true)
	backend, err := prepareDecisionPII(models)
	if err != nil || backend == nil {
		t.Fatalf("offline generic binding must prepare its task: %v", err)
	}
	if result := backend.judge(t.Context(), "private input"); !errors.Is(result.err, modelservice.ErrUnavailable) || len(services.requests) != 0 {
		t.Fatalf("offline judgment was not unknown: %+v", result)
	}
	services.card, services.ready = decisionObservation("noul"), true
	result := backend.judge(t.Context(), "private input")
	if result.err != nil || result.presence != .9 || len(result.categories) == 0 || len(services.requests) != 1 {
		t.Fatalf("recovered judgment lost presence/categories: %+v requests=%+v", result, services.requests)
	}
	for _, q := range services.requests[0].Questions {
		if q.Type != "noul" || !q.RequireFullInput || q.Truncate {
			t.Fatalf("recovery changed strict privacy task input or bypassed Set composition: %+v", q)
		}
	}
	services.card = decisionObservation("choice")
	if result := backend.judge(t.Context(), "private input"); !errors.Is(result.err, modelservice.ErrRejected) || len(services.requests) != 1 {
		t.Fatalf("changed capabilities must fail before inference: %+v", result)
	}
	services.ready = false
	if result := backend.judge(t.Context(), "private input"); !errors.Is(result.err, modelservice.ErrUnavailable) {
		t.Fatal("lost readiness reused an old card")
	}
	definition, _ := modelservice.BuiltinTask("pii_presence")
	question := definition.Question
	question.Truncate = true
	if _, err := backend.judgment.preparePlan(definition, question); !errors.Is(err, modelservice.ErrRejected) {
		t.Fatalf("offline preparation weakened structural input validation: %v", err)
	}
}

func TestAttachedGenericBindingRejectsKnownNonDecisionService(t *testing.T) {
	models, services := attachedDecisionModels(t, true)
	services.card = modelservice.ModelCard{ID: "classifier", Surfaces: []string{"classify"}}
	services.ready = true
	if backend, err := prepareDecisionPII(models); !errors.Is(err, modelservice.ErrRejected) || backend != nil {
		t.Fatalf("explicit generic binding silently fell back to a different adapter: %+v %v", backend, err)
	}
}

func TestImplicitAttachedPIIStillUsesMetadataToSelectPreciseAdapter(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	models.cfg.ModelDeployments["primary"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://attached.invalid"}
	services.card = decisionObservation("span", "noul", "set")
	backend, err := prepareDecisionPII(models)
	if err != nil || backend != nil || len(services.asked) != 1 {
		t.Fatalf("implicit adapter selection changed to deferred generic tasks: %+v %v discovery=%v", backend, err, services.asked)
	}
}
