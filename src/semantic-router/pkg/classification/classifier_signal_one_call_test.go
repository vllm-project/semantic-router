package classification

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// oneCallConfig asks one Vela 2.0 deployment from three signals: the prompt
// guard and the PII rule read the conversation's history too, and two
// decision questions route. TIMEOUT is the tone question's own timeout.
const oneCallConfig = `
version: v0.3
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
      prompt_guard: {deployment: vela2, contract: label_distribution.v1}
      pii_classifier: {deployment: vela2, contract: token_spans.v1}
routing:
  signals:
    jailbreak:
      - name: attack
        threshold: 0.9
        include_history: true
    pii:
      - name: personal_data
        threshold: 0.5
        include_history: true
    decision:
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
          - {type: decision, name: tone, label: urgent}
          - {type: decision, name: topics, label: billing}
      modelRefs:
        - model: general-model
`

func oneCallClassifier(t *testing.T, timeout string) (*Classifier, *runtimetest.Runtime) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(strings.Replace(oneCallConfig, "TIMEOUT", timeout, 1)))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"vela2": {Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON", "EMAIL_ADDRESS"}}, Joint: true},
	})
	classifier, err := buildClassifierWithAdmission(cfg, nil, nil, nil, nil, RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	if err = classifier.InitializeRuntime(); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	return classifier, fake
}

// slowTokens does its local work before it asks the PII question, as a
// loaded CPU can delay a signal past the bundle's window.
type slowTokens struct {
	*ownedTokenBackend
	delay time.Duration
}

func (s slowTokens) ClassifyTokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	time.Sleep(s.delay)
	return s.ownedTokenBackend.ClassifyTokens(ctx, text)
}

// slowDecider does the same for the decision signal.
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
		Text: text, CurrentUserText: text, NonUserMessages: history,
		RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
}

// answers is what a stage's signals published, the part a split would change.
func answers(results *SignalResults) map[string]interface{} {
	return map[string]interface{}{
		"values": results.SignalValues, "confidences": results.SignalConfidences, "errors": results.SignalErrors,
		"jailbreak": results.MatchedJailbreakRules, "pii": results.MatchedPIIRules, "entities": results.PIIEntities,
		"decision": results.MatchedDecisionRules,
	}
}

func TestAStageAsksVela2OneCallWhateverItsSignalsTiming(t *testing.T) {
	classifier, fake := oneCallClassifier(t, "0")
	text := "please refund the billing for my order"
	together := answers(evaluateOneCall(t, classifier, text, oneCallHistory))
	asked := len(fake.Decisions())
	pii, decider := classifier.piiInference.(*ownedTokenBackend), classifier.decider()
	for _, delay := range []time.Duration{50 * time.Millisecond, 200 * time.Millisecond, 500 * time.Millisecond} {
		for signal, slow := range map[string]func(){
			"pii":      func() { classifier.piiInference = slowTokens{pii, delay} },
			"decision": func() { classifier.SetDecisionDecider(slowDecider{decider, delay}) },
		} {
			classifier.piiInference, classifier.decisionDecider = pii, nil
			slow()
			// A new word every run keeps the Router's result cache out of it.
			results := answers(evaluateOneCall(t, classifier, text+" "+signal+delay.String(), oneCallHistory))
			if calls := len(fake.Decisions()) - asked; calls != 1 {
				t.Fatalf("%s %v late: %d decisions calls, want one", signal, delay, calls)
			}
			asked = len(fake.Decisions())
			if !reflect.DeepEqual(results, together) {
				t.Fatalf("%s %v late changed the answers:\n%+v\n%+v", signal, delay, results, together)
			}
		}
	}
	classifier.piiInference, classifier.decisionDecider = pii, nil
}

func TestHistoryAwareRulesAskEveryMessageInTheStagesOneCall(t *testing.T) {
	classifier, fake := oneCallClassifier(t, "0")
	before := len(fake.Decisions())
	results := evaluateOneCall(t, classifier, "please refund the billing for my order", oneCallHistory)
	asked := fake.Decisions()[before:]
	if len(asked) != 1 {
		t.Fatalf("the request and its history in %d decisions calls, want one", len(asked))
	}
	call := asked[0]
	if call.States == nil || len(*call.States) != len(oneCallHistory) {
		t.Fatalf("each earlier message is a state of the call: %+v", call.States)
	}
	for _, entry := range *call.States {
		if len(entry.Questions) != 2 {
			t.Fatalf("an earlier message is asked the guard's and the PII question only: %+v", entry.Questions)
		}
	}
	if len(call.Questions) != 4 {
		t.Fatalf("the request is asked every signal's question: %v", call.Questions)
	}
	if !slices.Contains(results.MatchedPIIRules, "personal_data") || !slices.Contains(results.PIIEntities, "PERSON") {
		t.Fatalf("the PII in an earlier message reaches its rule: rules %v entities %v errors %v", results.MatchedPIIRules, results.PIIEntities, results.SignalErrors)
	}
	if len(results.SignalErrors) != 0 {
		t.Fatalf("errors %v", results.SignalErrors)
	}
	// With history read by the guard only, the earlier messages are asked its
	// question alone, still in the stage's one call.
	classifier.Config.PIIRules[0].IncludeHistory = false
	before = len(fake.Decisions())
	results = evaluateOneCall(t, classifier, "please refund the billing for my order again", oneCallHistory)
	asked = fake.Decisions()[before:]
	if len(asked) != 1 || asked[0].States == nil || len(*asked[0].States) != len(oneCallHistory) || len(asked[0].Questions) != 4 {
		t.Fatalf("one call, the earlier messages as states: %+v", asked)
	}
	for _, entry := range *asked[0].States {
		if _, guard := entry.Questions["prompt_guard:attack"]; !guard || len(entry.Questions) != 1 {
			t.Fatalf("an earlier message is asked the guard's question only: %+v", entry.Questions)
		}
	}
	if slices.Contains(results.MatchedPIIRules, "personal_data") {
		t.Fatalf("the PII rule no longer reads the history: %v", results.PIIEntities)
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
		routing := id == "tone" || id == "topics"
		if truncates(question) != routing {
			t.Fatalf("question %s: routing questions read the first tokens, safety ones the whole text: %+v", id, question)
		}
	}
}

func TestADecisionTimeoutDoesNotDropAnAnswerOfTheSharedCall(t *testing.T) {
	classifier, fake := oneCallClassifier(t, "100")
	fake.SetDelay(300 * time.Millisecond)
	results := evaluateOneCall(t, classifier, "please refund the billing for my order", nil)
	if code, failed := results.SignalErrors["decision:tone"]; failed {
		t.Fatalf("the tone question shares the call with the guard and the PII question, which the stage waits for anyway; it timed out: %s", code)
	}
	if _, answered := results.SignalValues["decision:tone"]; !answered {
		t.Fatalf("tone has no answer: %v", results.SignalValues)
	}
	// Asked alone, the question still stops at its own timeout.
	classifier.Config.JailbreakRules, classifier.Config.PIIRules = nil, nil
	results = evaluateOneCall(t, classifier, "please refund the billing for my order once more", nil)
	if code := results.SignalErrors["decision:tone"]; code != decisionSignalErrorPrefix+"timeout" {
		t.Fatalf("alone, the tone question times out at 100 ms: %q", code)
	}
}
