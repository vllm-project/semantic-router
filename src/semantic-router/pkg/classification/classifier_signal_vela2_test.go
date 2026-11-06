package classification

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// vela2RoutingConfig routes on a Set label, a Span label and the PII signal,
// all answered by one Vela 2.0 deployment.
const vela2RoutingConfig = `
version: v0.3
providers:
  defaults:
    model: private-model
  models:
    - name: private-model
      backend_refs:
        - name: private-backend
          endpoint: localhost:8000
          protocol: http
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        endpoint: http://vela2.invalid:8100
      kai:
        provider: model_runtime
        endpoint: http://kai.invalid:8100
    bindings:
      pii: vela2
routing:
  signals:
    systemone:
      - name: topics
        deployment: vela2
        question:
          type: set
          instructions: Which topics does the request mention?
          labels:
            - key: billing
              description: payments, invoices or refunds
            - key: shipping
              description: deliveries or returns
      - name: places
        deployment: vela2
        question:
          type: span
          instructions: Which spans name a place?
          labels:
            - key: city
              description: a city name
      - name: urgent
        deployment: vela2
        question:
          type: noul
          instructions: Is this request urgent?
    pii:
      - name: personal_data
        predicate: {gte: 0.5}
  decisions:
    - name: private-billing
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: systemone
            name: topics
            label: billing
          - type: systemone
            name: places
            label: city
          - type: pii
            name: personal_data
      modelRefs:
        - model: private-model
`

func vela2Classifier(t *testing.T, yaml string) (*Classifier, *runtimetest.Runtime) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(yaml))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"vela2": {Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON", "EMAIL_ADDRESS"}}},
		"kai":   {},
	})
	classifier, err := buildClassifier(cfg, RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	primeAttachedDeploymentCards(t, runtime, cfg)
	return classifier, fake
}

// primeAttachedDeploymentCards waits for the card of every attached
// deployment, which its watcher would otherwise discover after startup.
func primeAttachedDeploymentCards(t *testing.T, runtime interface {
	DeploymentCard(context.Context, string, config.ModelDeployment) (modelservice.ModelCard, error)
}, cfg *config.RouterConfig,
) {
	t.Helper()
	for name, deployment := range cfg.ModelDeployments {
		if deployment.IsModelRuntime() && !deployment.Managed() {
			if _, err := runtime.DeploymentCard(context.Background(), name, deployment); err != nil {
				t.Fatalf("%s card: %v", name, err)
			}
		}
	}
}

func TestVela2SetSpanAndPIIAnswersRouteFromOneCall(t *testing.T) {
	classifier, fake := vela2Classifier(t, vela2RoutingConfig)
	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	before, beforeTasks := fake.Bundles()
	text := "please refund the billing for person in city"
	results := classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{Text: text, RequestFacts: RequestFacts{Context: context.Background()}}, classifier.Config.Decisions, true)
	for _, want := range []string{"topics:billing", "places:city"} {
		if !slices.Contains(results.MatchedSystemOneRules, want) {
			t.Fatalf("matched decision rules %v lack %q", results.MatchedSystemOneRules, want)
		}
	}
	if slices.Contains(results.MatchedSystemOneRules, "topics:shipping") {
		t.Fatalf("an unselected set label must not match: %v", results.MatchedSystemOneRules)
	}
	if _, asked := results.SignalValues["systemone:urgent"]; asked {
		t.Fatal("a rule no decision reads is not asked")
	}
	values := results.SignalValues
	if values["systemone:topics:billing"] != 0.9 || values["systemone:topics:shipping"] != 0.1 || values["systemone:places:city"] != 0.95 || values["systemone:places"] != 0.95 {
		t.Fatalf("every label probability is a signal value: %v", values)
	}
	if !slices.Contains(results.MatchedPIIRules, "personal_data") || !slices.Contains(results.PIIEntities, "PERSON") {
		t.Fatalf("PII from the ready-made question: rules %v entities %v errors %v", results.MatchedPIIRules, results.PIIEntities, results.SignalErrors)
	}
	after, afterTasks := fake.Bundles()
	if after-before != 1 || afterTasks-beforeTasks != 1 {
		t.Fatalf("the decision signals and the PII question to one deployment travel in one task: %d bundles, %d tasks", after-before, afterTasks-beforeTasks)
	}
	if fake.Calls("classify") != 0 {
		t.Fatal("no classify head is asked")
	}
	decision, err := classifier.EvaluateDecisionWithEngine(results)
	if err != nil || decision == nil || decision.Decision.Name != "private-billing" {
		t.Fatalf("decision = %+v, %v", decision, err)
	}
}

func TestGenericDecisionSetCompositionAndNativeSpanCapability(t *testing.T) {
	yaml := strings.Replace(vela2RoutingConfig, "      - name: topics\n        deployment: vela2", "      - name: topics\n        deployment: kai", 1)
	classifier, _ := vela2Classifier(t, yaml)
	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("a set task can compose native noul questions: %v", err)
	}
	yaml = strings.Replace(vela2RoutingConfig, "      - name: places\n        deployment: vela2", "      - name: places\n        deployment: kai", 1)
	classifier, _ = vela2Classifier(t, yaml)
	err := classifier.InitializeRuntime()
	if err == nil || !strings.Contains(err.Error(), "routing.signals.systemone[places]: span_required: ") {
		t.Fatalf("precise spans require a native span capability, got %v", err)
	}
}
