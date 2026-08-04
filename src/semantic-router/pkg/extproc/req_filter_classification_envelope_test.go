package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMetadataDecisionEvaluatesWithoutTextContent(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: model-a
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
      modelRefs:
        - model: model-a
          use_reasoning: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	requestContext := &RequestContext{
		TraceContext: context.Background(),
		Headers:      map[string]string{},
	}

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
