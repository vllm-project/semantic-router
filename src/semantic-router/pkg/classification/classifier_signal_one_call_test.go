package classification

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// oneCallConfig asks one Vela 2.0 deployment from three signals: the prompt
// guard and the PII rule read the conversation's history too, and two
// decision questions route. TIMEOUT is the tone question's own timeout.
const oneCallConfig = `
version: v0.5
providers:
  defaults:
    model: general-model
  models:
    - name: general-model
      backend_refs:
        - name: backend
          endpoint: localhost:8000
          protocol: http
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        endpoint: http://vela2.invalid:8100
        input: {overflow: window, max_tokens: 4096}
    bindings:
      jailbreak: vela2
      pii: vela2
recipes:
  - name: default
    signals:
      jailbreak:
        - name: attack
          predicate: {gte: 0.9}
          input: {history: true}
      pii:
        - name: personal_data
          predicate: {gte: 0.5}
          input: {history: true}
      systemone:
        - name: tone
          deployment: vela2
          timeout_ms: TIMEOUT
          question:
            type: choice
            instructions: What tone does the request take?
            choices:
              - key: neutral
              - key: urgent
        - name: topics
          deployment: vela2
          question:
            type: set
            instructions: Which topics does the request mention?
            labels:
              - key: billing
              - key: shipping
    decisions:
      - name: everything
        priority: 100
        rules:
          operator: OR
          conditions:
            - {type: jailbreak, name: attack}
            - {type: pii, name: personal_data}
            - {type: systemone, name: tone, label: urgent}
            - {type: systemone, name: topics, label: billing}
        modelRefs:
          - model: general-model
`

// oneCallGuardRule and oneCallPIIRule are the history-aware rules of
// oneCallConfig.
const (
	oneCallGuardRule = "      jailbreak:\n        - name: attack\n          predicate: {gte: 0.9}\n          input: {history: true}\n"
	oneCallPIIRule   = "      pii:\n        - name: personal_data\n          predicate: {gte: 0.5}\n          input: {history: true}\n"
)

func oneCallClassifier(t *testing.T, timeout string) (*Classifier, *runtimetest.Runtime) {
	t.Helper()
	return oneCallClassifierFrom(t, strings.Replace(oneCallConfig, "TIMEOUT", timeout, 1))
}

func oneCallClassifierFrom(t *testing.T, document string) (*Classifier, *runtimetest.Runtime) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"vela2": {Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON", "EMAIL_ADDRESS"}}, Joint: true},
	})
	classifier, err := buildClassifier(cfg, RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	if err = classifier.InitializeRuntime(); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	primeAttachedDeploymentCards(t, runtime, cfg)
	return classifier, fake
}

// slowDecider answers late, as a loaded CPU can delay a stage's call past
// the bundle's window.
type slowDecider struct {
	modelservice.Decider
	delay time.Duration
}

func (s slowDecider) Decide(ctx context.Context, deployment string, request modelservice.Request) (modelservice.Response, error) {
	time.Sleep(s.delay)
	return s.Decider.Decide(ctx, deployment, request)
}

var oneCallHistory = []string{"Earlier I wrote to person about the invoice.", "The assistant said it would check."}

func evaluateOneCall(t *testing.T, classifier *Classifier, text string, history []string) *SignalResults {
	t.Helper()
	return classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: text, CurrentUserText: text, PriorUserMessages: history,
		RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
}

// answers is what a stage's signals published, the part a split would change.
func answers(results *SignalResults) map[string]interface{} {
	return map[string]interface{}{
		"values": results.SignalValues, "confidences": results.SignalConfidences, "errors": results.SignalErrors,
		"jailbreak": results.MatchedJailbreakRules, "pii": results.MatchedPIIRules, "entities": results.PIIEntities,
		"systemone": results.MatchedSystemOneRules,
	}
}

func TestAStageAsksVela2OneCallWhateverItsSignalsTiming(t *testing.T) {
	classifier, fake := oneCallClassifier(t, "0")
	text := "please refund the billing for my order"
	together := answers(evaluateOneCall(t, classifier, text, oneCallHistory))
	asked := len(fake.Decisions())
	decider := classifier.decider()
	for _, delay := range []time.Duration{50 * time.Millisecond, 200 * time.Millisecond, 500 * time.Millisecond} {
		classifier.SetDecisionDecider(slowDecider{decider, delay})
		// A new word every run keeps the Router's result cache out of it.
		results := answers(evaluateOneCall(t, classifier, text+" "+delay.String(), oneCallHistory))
		if calls := len(fake.Decisions()) - asked; calls != 1 {
			t.Fatalf("%v late: %d decisions calls, want one", delay, calls)
		}
		asked = len(fake.Decisions())
		if !reflect.DeepEqual(results, together) {
			t.Fatalf("%v late changed the answers:\n%+v\n%+v", delay, results, together)
		}
	}
	classifier.decisionDecider = nil
}

// A rule that reads the history asks the request with its earlier user
// messages, a state of its own in the stage's one call: the guard reads them
// as context, and the pii rule, a span question, scans them with the request.
func TestHistoryAwareRulesAskTheirStateInTheStagesOneCall(t *testing.T) {
	document := strings.Replace(oneCallConfig, "TIMEOUT", "0", 1)
	for name, tc := range map[string]struct {
		document string
		states   int
		reading  int
	}{
		"the guard and the pii rule": {document, 3, 2},
		"the guard alone":            {strings.Replace(document, oneCallPIIRule, strings.Replace(oneCallPIIRule, "          input: {history: true}\n", "", 1), 1), 2, 1},
	} {
		t.Run(name, func(t *testing.T) {
			classifier, fake := oneCallClassifierFrom(t, tc.document)
			before := len(fake.Decisions())
			evaluateOneCall(t, classifier, "please refund the billing for my order", oneCallHistory)
			asked := fake.Decisions()[before:]
			if len(asked) != 1 || asked[0].Questions != nil || asked[0].States == nil || len(*asked[0].States) != tc.states {
				t.Fatalf("one call, a batch with the history in an entry of its own: %+v", asked)
			}
			reading := 0
			for _, state := range *asked[0].States {
				encoded, err := json.Marshal(state.State)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(encoded), oneCallHistory[0]) {
					reading += len(state.Questions)
				}
			}
			if reading != tc.reading {
				t.Fatalf("%d questions read the history, want %d: %+v", reading, tc.reading, asked[0])
			}
		})
	}
}

func TestRoutingAndSafetyQuestionsKeepTheirReadBudgetsInTheSharedCall(t *testing.T) {
	classifier, fake := oneCallClassifier(t, "0")
	before := len(fake.Decisions())
	evaluateOneCall(t, classifier, "please refund the billing for my order", nil)
	asked := fake.Decisions()[before:]
	if len(asked) != 1 || asked[0].Options == nil || asked[0].Options.MaxTokens == nil || *asked[0].Options.MaxTokens != 4096 {
		t.Fatalf("one call with the deployment's scan budget: %+v", asked)
	}
	truncates := func(question api.Question) bool {
		return question.Overflow != nil && *question.Overflow == api.QuestionOverflowTruncate
	}
	for id, question := range asked[0].Questions {
		routing := id == "systemone:tone" || id == "systemone:topics"
		if truncates(question) != routing {
			t.Fatalf("question %s: routing questions read the first tokens, safety ones the whole text: %+v", id, question)
		}
	}
}

func TestADecisionTimeoutDoesNotDropAnAnswerOfTheSharedCall(t *testing.T) {
	classifier, fake := oneCallClassifier(t, "100")
	fake.SetDelay(300 * time.Millisecond)
	results := evaluateOneCall(t, classifier, "please refund the billing for my order", nil)
	if code, failed := results.SignalErrors["systemone:tone"]; failed {
		t.Fatalf("the tone question shares the call with the guard and the PII question, which the stage waits for anyway; it timed out: %s", code)
	}
	if _, answered := results.SignalValues["systemone:tone"]; !answered {
		t.Fatalf("tone has no answer: %v", results.SignalValues)
	}
	// Asked alone, the question still stops at its own timeout.
	alone := strings.NewReplacer(oneCallGuardRule, "", oneCallPIIRule, "",
		"            - {type: jailbreak, name: attack}\n", "", "            - {type: pii, name: personal_data}\n", "",
	).Replace(strings.Replace(oneCallConfig, "TIMEOUT", "100", 1))
	classifier, fake = oneCallClassifierFrom(t, alone)
	fake.SetDelay(300 * time.Millisecond)
	results = evaluateOneCall(t, classifier, "please refund the billing for my order once more", nil)
	if code := results.SignalErrors["systemone:tone"]; code != systemOneSignalErrorPrefix+"timeout" {
		t.Fatalf("alone, the tone question times out at 100 ms: %q", code)
	}
}
