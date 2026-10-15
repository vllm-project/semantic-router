package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

const genericPrivacyConfig = `version: v0.3
providers:
  defaults: {model: backend}
  models:
    - name: backend
      backend_refs: [{name: local, endpoint: localhost:8000, protocol: http}]
global:
  model_catalog:
    system:
      decision_model: {deployment: judgment}
    deployments:
      judgment: {provider: model_runtime, endpoint: http://judgment.invalid:8100}
routing:
  signals:
    pii: [{name: privacy, threshold: 0.5, include_history: true}]
  decisions:
    - name: private
      rules: {operator: AND, conditions: [{type: pii, name: privacy}]}
      modelRefs: [{model: backend}]
`

func TestGenericDecisionDefaultPreparesPIIWithoutSpanAndSharesBundle(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(genericPrivacyConfig))
	if err != nil {
		t.Fatal(err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{"judgment": {}})
	classifier, err := buildClassifierWithAdmission(cfg, nil, nil, nil, nil, RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	if err = classifier.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	if !classifier.IsPIIEnabled() || decisionPIIInference(classifier.piiInference) == nil {
		t.Fatal("default still required token spans")
	}
	before, beforeTasks := fake.Bundles()
	result := classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{Text: "contact person", CurrentUserText: "contact person", NonUserMessages: []string{"prior context"}, RequestFacts: RequestFacts{Context: t.Context()}}, classifier.Config.Decisions, true)
	after, afterTasks := fake.Bundles()
	if after-before != 1 || afterTasks-beforeTasks != 1 {
		t.Fatalf("PII states/tasks lost existing bundle fusion: %d bundles %d tasks", after-before, afterTasks-beforeTasks)
	}
	if len(result.SignalErrors) != 0 || len(result.PIIEvidence) != 2 || fake.Calls("classify") != 0 {
		t.Fatalf("generic task failed or asked token head: %+v", result.SignalErrors)
	}
}

func TestExplicitPIITokenContractRejectsMissingSpanAtPreparation(t *testing.T) {
	source := strings.Replace(genericPrivacyConfig, "    deployments:", "    bindings:\n      pii_classifier: {deployment: judgment, contract: token_spans.v1}\n    deployments:", 1)
	cfg, err := config.ParseYAMLBytes([]byte(source))
	if err != nil {
		t.Fatal(err)
	}
	runtime, _ := servingtest.Runtime(t, map[string]runtimetest.Model{"judgment": {}})
	if _, err = buildClassifierWithAdmission(cfg, nil, nil, nil, nil, RecipeRuntimeOptions{Runtime: runtime}); err == nil || !strings.Contains(err.Error(), "token_spans.v1 binding requires a span model") {
		t.Fatalf("missing precise capability silently ran: %v", err)
	}
}
