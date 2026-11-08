package classification

import (
	"context"
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

func attachedDecisionModels(t *testing.T) (*classifierModelRuntime, *observedDecisionServices) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(setQuestionOnTheDecisionModel))
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
	models, services := attachedDecisionModels(t)
	c := &Classifier{Config: models.cfg, models: models}
	if err := c.prepareSystemOneSignals(); err != nil {
		t.Fatalf("offline attached provider blocked preparation: %v", err)
	}
	evaluate := func() *SignalResults {
		r := &SignalResults{SignalErrors: map[string]string{}, SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}}
		var rules []askedRule
		for _, rule := range models.cfg.SystemOneRules {
			rules = append(rules, systemOneAsked(rule))
		}
		c.evaluateSystemOneDeployment(t.Context(), r, &sync.Mutex{}, &systemOneCall{deployment: "primary", request: modelservice.Request{State: "needs tools"}, rules: rules})
		return r
	}
	if result := evaluate(); result.SignalErrors["systemone:needs"] == "" || len(services.requests) != 0 {
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
	if result := evaluate(); result.SignalErrors["systemone:needs"] == "" || len(services.requests) != 1 {
		t.Fatal("changed attached capabilities reused an old card or bypassed compilation")
	}
	want := `recipes[default].signals.systemone[needs]: question_type_unsupported: resolves to deployment "primary", which answers choice questions and cannot answer set questions`
	if err := c.prepareSystemOneSignals(); err == nil || !strings.HasPrefix(err.Error(), want) {
		t.Fatalf("known unsupported capability must reject preparation: %v", err)
	}
}

func TestAttachedDecisionSpanIsNeverComposedOrSilentlyAccepted(t *testing.T) {
	models, services := attachedDecisionModels(t)
	models.cfg.SystemOneRules[0].Question.Type = "span"
	c := &Classifier{Config: models.cfg, models: models}
	if err := c.prepareSystemOneSignals(); err != nil {
		t.Fatal(err)
	}
	services.card, services.ready = decisionObservation("noul"), true
	want := `recipes[default].signals.systemone[needs]: span_required: resolves to deployment "primary", which cannot answer span questions`
	if err := c.prepareSystemOneSignals(); err == nil || !strings.HasPrefix(err.Error(), want) {
		t.Fatalf("Noul cannot provide locations: %v", err)
	}
	services.card = decisionObservation("span")
	if err := c.prepareSystemOneSignals(); err != nil {
		t.Fatalf("native Span capability was lost: %v", err)
	}
}
