package classification

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type signalContextTokenCounter struct {
	calls int
}

func (c *signalContextTokenCounter) CountTokens(string) (int, error) {
	c.calls++
	return 1, nil
}

func TestEvaluateAllSignalsWithContextIncludesSignalsFromAllRecipes(t *testing.T) {
	contextRules := []config.ContextRule{{Name: "recipe-only-context", MinTokens: "0", MaxTokens: "10"}}
	defaultDecision := config.Decision{
		Name:  "default-route",
		Rules: config.RuleNode{Type: config.SignalTypeKeyword, Name: "default-keyword"},
	}
	recipeDecision := config.Decision{
		Name:  "recipe-route",
		Rules: config.RuleNode{Type: config.SignalTypeContext, Name: "recipe-only-context"},
	}
	counter := &signalContextTokenCounter{}
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals:   config.Signals{ContextRules: contextRules},
				Decisions: []config.Decision{defaultDecision},
			},
			Recipes: []config.RoutingRecipe{
				{Name: config.DefaultRecipeName, Decisions: []config.Decision{defaultDecision}},
				{Name: "recipe-only", Decisions: []config.Decision{recipeDecision}},
			},
		},
		contextClassifier: NewContextClassifier(counter, contextRules),
	}

	results := classifier.EvaluateAllSignalsWithContext(
		"hello", "hello", "hello", nil, nil, false, false, "", nil, ConversationFacts{}, "",
	)

	if counter.calls != 1 {
		t.Fatalf("context classifier calls = %d, want 1", counter.calls)
	}
	if !reflect.DeepEqual(results.MatchedContextRules, []string{"recipe-only-context"}) {
		t.Fatalf("matched context rules = %v, want [recipe-only-context]", results.MatchedContextRules)
	}
}
