package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Issue #2880: a rule with include_history must match a keyword that only
// appears in an earlier user message.
func TestKeywordIncludeHistoryMatchesPriorUserMessage(t *testing.T) {
	// 1. One keyword rule, with the new flag turned ON.
	rules := []config.KeywordRule{{
		Name:           "local_only",
		Operator:       "OR",
		Keywords:       []string{"local model only"},
		IncludeHistory: true,
	}}

	// 2. Build the keyword engine and a router classifier that uses it.
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = rules
	classifier := &Classifier{Config: cfg, keywordClassifier: kc}

	// 3. The chat from the issue: keyword only in the EARLIER message.
	prior := []string{"This is confidential, use local model only"}
	current := "Use the biggest model"

	// 4. Run all signals, the same way a real request does.
	results := classifier.EvaluateAllSignalsWithContext(
		current, // text the signals evaluate
		current, // context text (for length counting)
		current, // current user text
		prior,   // earlier user messages
		nil,     // assistant/system messages
		true,    // has a prior assistant reply
		true,    // force-evaluate every signal
		"",      // uncompressed text
		nil,     // signals that skip compression
		ConversationFacts{},
		"", // image URL
	)

	// 5. Expect the rule to have matched.
	if len(results.MatchedKeywordRules) != 1 || results.MatchedKeywordRules[0] != "local_only" {
		t.Fatalf("MatchedKeywordRules = %v, want [local_only]", results.MatchedKeywordRules)
	}
}

func TestKeywordWithoutIncludeHistoryIgnoresPriorUserMessage(t *testing.T) {
	// 1. One keyword rule, with the new flag turned OFF.
	rules := []config.KeywordRule{{
		Name:     "local_only",
		Operator: "OR",
		Keywords: []string{"local model only"},
	}}

	// 2. Build the keyword engine and a router classifier that uses it.
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = rules
	classifier := &Classifier{Config: cfg, keywordClassifier: kc}

	// 3. The chat from the issue: keyword only in the EARLIER message.
	prior := []string{"This is confidential, use local model only"}
	current := "Use the biggest model"

	// 4. Run all signals, the same way a real request does.
	results := classifier.EvaluateAllSignalsWithContext(
		current, // text the signals evaluate
		current, // context text (for length counting)
		current, // current user text
		prior,   // earlier user messages
		nil,     // assistant/system messages
		true,    // has a prior assistant reply
		true,    // force-evaluate every signal
		"",      // uncompressed text
		nil,     // signals that skip compression
		ConversationFacts{},
		"", // image URL
	)

	// 5. Expect the rule to have not matched.
	if len(results.MatchedKeywordRules) != 0 {
		t.Fatalf("MatchedKeywordRules = %v, want none", results.MatchedKeywordRules)
	}
}
