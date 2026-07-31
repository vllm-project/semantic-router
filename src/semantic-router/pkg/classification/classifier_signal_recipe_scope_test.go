package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSignalEvaluationScopesMetadataToCandidateDecisions(t *testing.T) {
	denied := "denied"
	keywordDecision := config.Decision{
		Name: "keyword-route",
		Rules: config.RuleNode{
			Type: config.SignalTypeKeyword,
			Name: "urgent",
		},
	}
	classifier := &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{MetadataRules: []config.MetadataRule{{
				Name:      "consent-denied",
				Key:       "consent",
				Predicate: config.MetadataPredicate{Equals: &denied},
			}}},
			Decisions: []config.Decision{keywordDecision},
		},
	}}

	results, err := classifier.EvaluateAllSignalsWithHeaders(
		SignalEvaluationInput{
			Text:            "hello",
			ContextText:     "hello",
			CurrentUserText: "hello",
			RequestFacts: RequestFacts{
				Metadata: map[string]string{"consent": "denied"},
			},
		},
	)
	if err != nil {
		t.Fatalf("EvaluateAllSignalsWithHeaders() error = %v", err)
	}
	if len(results.MatchedMetadataRules) != 0 {
		t.Fatalf("metadata signals evaluated outside recipe scope: %v", results.MatchedMetadataRules)
	}
}
