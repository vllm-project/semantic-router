package classification

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// The feedback question judges the answer a user turn replies to, so it reads
// the last assistant turn with the request.
func TestAFeedbackRuleReadsTheAnswerTheTurnRepliesTo(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.UserFeedbackRules = []config.UserFeedbackRule{{Name: "wrong", Labels: []string{"WRONG_ANSWER"}, Deployment: "kai"}}
	decider := newFakeDecider(map[string]modelservice.Answer{
		presetQuestionID(presets.Feedback): {Type: "choice", Choice: "WRONG_ANSWER", Probabilities: map[string]float64{"WRONG_ANSWER": 0.9, "SAT": 0.1}},
	})
	classifier := newApplicationTestClassifier(t, cfg, decider)
	ask := func(reading systemOneReading) modelservice.Request {
		classifier.evaluateApplicationSignals(context.Background(), newSignalResults(), &sync.Mutex{}, config.SignalTypeUserFeedback, reading, usedApplicationRules(cfg))
		return decider.asked[len(decider.asked)-1]
	}

	turns := []ConversationTurn{
		{Role: "user", Text: "How do I reset the router?"},
		{Role: "assistant", Text: "Hold the button for ten seconds."},
		{Role: "user", Text: "It has no button."},
	}
	asked := ask(systemOneReading{request: "That did not work.", turns: turns})
	want := map[string]string{"request": "That did not work.", "previous_answer": "Hold the button for ten seconds."}
	if asked.State != "" || len(asked.Parts) != len(want) {
		t.Fatalf("the request and the answer it replies to: %+v", asked)
	}
	for name, text := range want {
		if asked.Parts[name] != text {
			t.Fatalf("part %s = %q, want %q", name, asked.Parts[name], text)
		}
	}
	if asked := ask(systemOneReading{request: "That did not work.", turns: turns[:1]}); asked.State != "That did not work." || asked.Parts != nil {
		t.Fatalf("a turn with no answer before it reads the request alone: %+v", asked)
	}
}
