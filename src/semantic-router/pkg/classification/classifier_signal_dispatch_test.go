package classification

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestATurnRelationIsAskedOnlyAfterEarlierTurns(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.TurnRelationRules = []config.TurnRelationRule{{Name: "reasked", Labels: []string{"reask"}, Deployment: "kai"}}
	decider := newFakeDecider(map[string]modelservice.Answer{
		presetQuestionID(presets.TurnRelation): {Type: "choice", Choice: "reask", Probabilities: map[string]float64{"continue": 0.2, "reask": 0.6, "shift": 0.1, "return": 0.1}},
	})
	classifier := newApplicationTestClassifier(t, cfg, decider)
	evaluate := func(turns []ConversationTurn) *SignalResults {
		results := newSignalResults()
		input := SignalEvaluationInput{Text: "That fix is still wrong.", RequestFacts: RequestFacts{Turns: turns}}
		text := func(string) string { return input.Text }
		for _, dispatch := range classifier.buildApplicationSignalDispatchers(input, results, &sync.Mutex{}, text, usedApplicationRules(cfg)) {
			if dispatch.signalType == config.SignalTypeTurnRelation {
				dispatch.evaluate(context.Background())
			}
		}
		return results
	}

	if results := evaluate(nil); len(results.MatchedTurnRelationRules) != 0 || decider.calls["kai"] != 0 {
		t.Fatalf("a first turn relates to nothing and asks nothing: matched=%v calls=%v", results.MatchedTurnRelationRules, decider.calls)
	}
	earlier := []ConversationTurn{{Role: "user", Text: "Fix this function."}, {Role: "assistant", Text: "Index the list by key."}}
	if results := evaluate(earlier); !slices.Contains(results.MatchedTurnRelationRules, "reasked") || decider.calls["kai"] != 1 {
		t.Fatalf("a later turn asks its relation: matched=%v calls=%v", results.MatchedTurnRelationRules, decider.calls)
	}
}
