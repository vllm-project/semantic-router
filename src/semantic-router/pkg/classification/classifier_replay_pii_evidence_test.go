package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestReplayPromptEvidenceCoversOnlyACompleteCleanReadOfThePrompt(t *testing.T) {
	const prompt = "Summarize the meeting agenda."
	email := modelservice.Span{Label: "EMAIL_ADDRESS", Text: "alice@example.test", Probability: 0.99}
	for _, test := range []struct {
		name     string
		answer   *modelservice.Answer
		fail     error
		rules    bool
		clean    bool
		complete bool
	}{
		{name: "clean", rules: true, answer: &modelservice.Answer{Type: "span", InputCoverage: "complete"}, clean: true, complete: true},
		{name: "personal data", rules: true, answer: &modelservice.Answer{Type: "span", InputCoverage: "complete", Spans: []modelservice.Span{email}}, complete: true},
		{name: "read failed", rules: true, fail: modelservice.ErrUnavailable},
		{name: "read incomplete", rules: true, answer: &modelservice.Answer{Type: "span", Error: "max_length_exceeded"}},
		{name: "no rules"},
	} {
		t.Run(test.name, func(t *testing.T) {
			answers := map[string]modelservice.Answer{}
			if test.answer != nil {
				answers[presetQuestionID(presets.PII)] = *test.answer
			}
			decider := newFakeDecider(answers)
			if test.fail != nil {
				decider.failures["guard-model"] = test.fail
			}
			cfg := &config.RouterConfig{}
			if test.rules {
				cfg.PIIRules = []config.PIIRule{{Name: "privacy", Deployment: "guard-model"}}
			}
			classifier, err := newClassifierWithOptions(cfg)
			if err != nil {
				t.Fatal(err)
			}
			classifier.SetDecisionDecider(decider)
			evidence := classifier.ReplayPromptEvidence(t.Context(), prompt)
			if evidence.CoversClean("request", prompt) != test.clean {
				t.Fatalf("clean=%v want=%v (%+v)", evidence.CoversClean("request", prompt), test.clean, evidence)
			}
			if (evidence.Coverage == "complete") != test.complete {
				t.Fatalf("coverage=%q, want complete=%v", evidence.Coverage, test.complete)
			}
			if evidence.CoversClean("request", prompt+" ") {
				t.Fatal("evidence must cover exactly the prompt that was read")
			}
			if !test.rules && len(decider.asked) != 0 {
				t.Fatalf("a recipe without pii rules asked the model: %+v", decider.asked)
			}
		})
	}
}
