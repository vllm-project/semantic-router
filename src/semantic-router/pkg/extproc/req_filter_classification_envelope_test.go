package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

func TestMetadataDecisionEvaluatesWithoutTextContent(t *testing.T) {
	cfg, err := parseExtProcAuthoringConfig(t, `
version: v0.4
models:
  - name: model-a
    card: {}
    connections:
      - provider: vllm
        endpoint: http://127.0.0.1:8000
        model: model-a
recipes:
  - name: canary
    document:
      signals:
        metadata:
          - name: canary
            key: cohort
            predicate:
              equals: canary
      decisions:
        - name: canary-route
          priority: 10
          rules:
            type: metadata
            name: canary
entrypoints:
  - name: vllm-sr/auto
    recipe: canary
    assignments:
      canary-route:
        models: [{model: model-a}]
`)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}
	requestContext := &RequestContext{
		TraceContext: context.Background(),
		Headers:      map[string]string{},
	}
	router.resolveEntrypointForRequest("vllm-sr/auto", requestContext)

	decision, _, _, selectedModel, err := router.performDecisionEvaluation(
		"vllm-sr/auto",
		signalConversationHistory{metadata: map[string]string{"cohort": "canary"}},
		requestContext,
	)
	if err != nil {
		t.Fatalf("performDecisionEvaluation() error = %v", err)
	}
	if decision != "canary-route" || selectedModel != "model-a" {
		t.Fatalf("decision/model = %q/%q, want canary-route/model-a", decision, selectedModel)
	}
}
