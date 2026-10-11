package classification

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// vela2BuiltInConfig binds every built-in signal Vela 2.0 answers to one
// deployment, with decisions that read each of them.
const vela2BuiltInConfig = `
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
    bindings:
      domain_classifier: {deployment: vela2, contract: label_distribution.v1}
      prompt_guard: {deployment: vela2, contract: label_distribution.v1}
      pii_classifier: {deployment: vela2, contract: token_spans.v1}
      fact_check_classifier: {deployment: vela2, contract: label_distribution.v1}
      feedback_detector: {deployment: vela2, contract: label_distribution.v1}
      modality_detector: {deployment: vela2, contract: label_distribution.v1}
      safety.unsafe_request: {deployment: vela2, contract: label_distribution.v1}
    modules:
      modality_detector:
        enabled: true
        method: classifier
        confidence_threshold: 0.6
routing:
  modelCards:
    - name: general-model
      modality: omni
  signals:
    domains:
      - name: biology
        mmlu_categories: [biology]
      - name: physics
        mmlu_categories: [physics]
    jailbreak:
      - name: attack
        threshold: 0.5
    pii:
      - name: personal_data
        threshold: 0.5
    fact_check:
      - name: needs_fact_check
      - name: no_fact_check_needed
    user_feedbacks:
      - name: satisfied
      - name: wrong_answer
    modality:
      - name: AR
      - name: DIFFUSION
    safety:
      - name: unsafe_request
        labels: [safe, unsafe]
        unsafe_labels: [unsafe]
        threshold: 0.5
  decisions:
    - name: every-signal
      priority: 10
      rules:
        operator: OR
        conditions:
          - {type: domain, name: biology}
          - {type: domain, name: physics}
          - {type: jailbreak, name: attack}
          - {type: pii, name: personal_data}
          - {type: fact_check, name: needs_fact_check}
          - {type: fact_check, name: no_fact_check_needed}
          - {type: user_feedback, name: satisfied}
          - {type: user_feedback, name: wrong_answer}
          - {type: modality, name: AR}
          - {type: modality, name: DIFFUSION}
          - {type: safety, name: unsafe_request}
      modelRefs:
        - model: general-model
`

// The fake runtime answers a Choice question with 0.7 for its first option,
// so each consumer reads the first label of its Vela 1.0 head: biology,
// benign, NO_FACT_CHECK_NEEDED, SAT, AR and safe.
func TestBuiltInSignalsOnVela2RouteFromOneCall(t *testing.T) {
	classifier, fake := vela2Classifier(t, vela2BuiltInConfig)
	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	before, beforeTasks := fake.Bundles()
	text := "a question about person names"
	results := classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: text, CurrentUserText: text, HasPriorAssistantReply: true,
		RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
	if len(results.SignalErrors) != 0 {
		t.Fatalf("signal errors: %v", results.SignalErrors)
	}
	for kind, matched := range map[string][]string{
		"domain":        results.MatchedDomainRules,
		"fact_check":    results.MatchedFactCheckRules,
		"user_feedback": results.MatchedUserFeedbackRules,
		"modality":      results.MatchedModalityRules,
		"pii":           results.MatchedPIIRules,
	} {
		want := map[string]string{"domain": "biology", "fact_check": "no_fact_check_needed", "user_feedback": "satisfied", "modality": "AR", "pii": "personal_data"}[kind]
		if !slices.Equal(matched, []string{want}) {
			t.Fatalf("%s matched %v, want [%s]", kind, matched, want)
		}
	}
	if len(results.MatchedJailbreakRules) != 0 || len(results.MatchedSafetyRules) != 0 {
		t.Fatalf("benign and safe answers match no guard: jailbreak %v safety %v", results.MatchedJailbreakRules, results.MatchedSafetyRules)
	}
	if risk := results.SignalValues["safety:unsafe_request"]; math.Abs(risk-0.3) > 1e-6 {
		t.Fatalf("the safety risk is the unsafe option's probability, got %v", risk)
	}
	if !slices.Contains(results.PIIEntities, "PERSON") {
		t.Fatalf("PII entities %v", results.PIIEntities)
	}
	after, afterTasks := fake.Bundles()
	if after-before != 1 || afterTasks-beforeTasks != 1 {
		t.Fatalf("every question about the request travels in one task: %d bundles, %d tasks", after-before, afterTasks-beforeTasks)
	}
	if fake.Calls("classify") != 0 {
		t.Fatal("no classify head is asked")
	}

	// A long request is neither sampled for the semantic signals nor cut into
	// Guard chunks: the model reads it whole, still in one call.
	long := strings.Repeat("a long question about person names and their places ", 400)
	before, beforeTasks = fake.Bundles()
	results = classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: long, CurrentUserText: long, HasPriorAssistantReply: true,
		RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
	if len(results.SignalErrors) != 0 || !slices.Equal(results.MatchedDomainRules, []string{"biology"}) {
		t.Fatalf("long request: errors %v, domains %v", results.SignalErrors, results.MatchedDomainRules)
	}
	after, afterTasks = fake.Bundles()
	if after-before != 1 || afterTasks-beforeTasks != 1 {
		t.Fatalf("a long request also travels whole in one task: %d bundles, %d tasks", after-before, afterTasks-beforeTasks)
	}

	// Surrounding whitespace reaches every signal as it is, so PII shares the call.
	padded := "\n  a question about person names and their places \n"
	before, beforeTasks = fake.Bundles()
	results = classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: padded, CurrentUserText: padded, HasPriorAssistantReply: true,
		RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
	if len(results.SignalErrors) != 0 {
		t.Fatalf("padded request: errors %v", results.SignalErrors)
	}
	after, afterTasks = fake.Bundles()
	if after-before != 1 || afterTasks-beforeTasks != 1 {
		t.Fatalf("a request with surrounding whitespace also travels in one task: %d bundles, %d tasks", after-before, afterTasks-beforeTasks)
	}

	// Prompt compression shortens the text the bounded signals read; every
	// signal Vela 2.0 answers reads the request as it came, in one call.
	original := "a question about person names and their places, with details"
	before, beforeTasks = fake.Bundles()
	results = classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: "a question about person names", UncompressedText: original, CurrentUserText: original,
		SkipCompressionSignals: map[string]bool{config.SignalTypeJailbreak: true, config.SignalTypePII: true},
		HasPriorAssistantReply: true, RequestFacts: RequestFacts{Context: context.Background()},
	}, classifier.Config.Decisions, true)
	if len(results.SignalErrors) != 0 {
		t.Fatalf("compressed request: errors %v", results.SignalErrors)
	}
	after, afterTasks = fake.Bundles()
	if after-before != 1 || afterTasks-beforeTasks != 1 {
		t.Fatalf("a compressed request also travels in one task: %d bundles, %d tasks", after-before, afterTasks-beforeTasks)
	}
}

func TestBuiltInSignalOnVela2RefusesAConsumerWindow(t *testing.T) {
	yaml := strings.Replace(vela2BuiltInConfig, "    modules:\n", "    modules:\n      prompt_guard:\n        enabled: true\n        max_sequence_length: 8192\n        window: {size: 512, overlap: 255}\n", 1)
	classifier, _ := vela2Classifier(t, yaml)
	err := classifier.InitializeRuntime()
	if err == nil || !strings.Contains(err.Error(), "reads the whole text for prompt_guard") {
		t.Fatalf("a Guard window on a Vela 2.0 binding must fail preparation, got %v", err)
	}
}
