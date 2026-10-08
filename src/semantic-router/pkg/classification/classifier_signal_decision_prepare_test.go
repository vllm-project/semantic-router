package classification

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// cardServices serves one model card for every deployment and records which
// deployments preparation asked for.
type cardServices struct {
	card  modelservice.ModelCard
	asked []string
}

func (s *cardServices) Card(_ context.Context, deployment string) (modelservice.ModelCard, error) {
	s.asked = append(s.asked, deployment)
	return s.card, nil
}

func (s *cardServices) Classify(context.Context, string, modelservice.ClassifyRequest) (modelservice.ClassifyResponse, error) {
	return modelservice.ClassifyResponse{}, nil
}

func (s *cardServices) Embed(context.Context, string, modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	return modelservice.EmbedResponse{}, nil
}

func (s *cardServices) Rerank(context.Context, string, modelservice.RerankRequest) (modelservice.RerankResponse, error) {
	return modelservice.RerankResponse{}, nil
}

const setQuestionOnTheDecisionModel = `version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
  signals:
    decision:
      - name: needs
        question:
          type: set
          instructions: What does a good answer need?
          labels:
            - {key: tools}
            - {key: deliberation}
  decisions:
    - name: tools
      priority: 100
      rules:
        operator: AND
        conditions:
          - {type: decision, name: needs, label: tools, on_error: no_match}
      modelRefs: [{model: general}]
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
    deployments:
      primary: {provider: model_runtime, artifact: vllm-sr/Vela-2.0-4B, device: auto, profile: exact}
`

func preparedSetQuestion(t *testing.T, card modelservice.ModelCard) (*cardServices, error) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(setQuestionOnTheDecisionModel))
	if err != nil {
		t.Fatal(err)
	}
	services := &cardServices{card: card}
	models, err := newClassifierModelRuntime(cfg, RecipeRuntimeOptions{Runtime: serving.New(services, nil)})
	if err != nil {
		t.Fatal(err)
	}
	classifier := &Classifier{Config: cfg, models: models}
	return services, classifier.prepareDecisionSignals()
}

func TestASetQuestionWithoutDeploymentIsCheckedOnTheDecisionModel(t *testing.T) {
	vela2 := modelservice.ModelCard{ID: "vllm-sr/Vela-2.0-4B", Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice", "noul", "score", "set", "span"}}
	services, err := preparedSetQuestion(t, vela2)
	if err != nil {
		t.Fatalf("a set question on the Vela 2.0 decision model must prepare: %v", err)
	}
	if len(services.asked) != 1 || services.asked[0] != "primary" {
		t.Fatalf("preparation asked %v, want the decision model's deployment", services.asked)
	}
	systemOne := modelservice.ModelCard{ID: "vllm-sr/Decision-2.0-Kai-0.6B", Surfaces: []string{"decisions"}}
	if _, err = preparedSetQuestion(t, systemOne); err != nil {
		t.Fatalf("a model with Noul must prepare a composed Set: %v", err)
	}
	unsupported := modelservice.ModelCard{ID: "choice-only", Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice"}}
	if _, err = preparedSetQuestion(t, unsupported); err == nil {
		t.Fatal("missing Set and Noul must reject preparation")
	}
}
