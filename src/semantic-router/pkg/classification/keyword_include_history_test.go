package classification

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// matchedKeywordRulesFor runs every signal for one request, the same way a
// real request does, and returns the names of the keyword rules that matched.
func matchedKeywordRulesFor(t *testing.T, rules []config.KeywordRule, prior []string, current string) []string {
	t.Helper()
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = rules
	classifier := &Classifier{Config: cfg, keywordClassifier: kc}

	results := classifier.EvaluateAllSignalsWithContext(
		current, // text the signals evaluate
		current, // context text (for length counting)
		current, // current user text
		prior,   // earlier user messages
		nil,     // assistant/system messages
		len(prior) > 0,
		true, // force-evaluate every signal (no decisions in these tests)
		"",   // uncompressed text
		nil,  // signals that skip compression
		ConversationFacts{},
		"", // image URL
	)
	return results.MatchedKeywordRules
}

func localOnlyRule(includeHistory bool) config.KeywordRule {
	return config.KeywordRule{
		Name:           "local_only",
		Operator:       "OR",
		Keywords:       []string{"local model only"},
		IncludeHistory: includeHistory,
	}
}

// Issue #2880: a rule with include_history must match a keyword that only
// appears in an earlier user message.
func TestKeywordIncludeHistoryMatchesPriorUserMessage(t *testing.T) {
	got := matchedKeywordRulesFor(t,
		[]config.KeywordRule{localOnlyRule(true)},
		[]string{"This is confidential, use local model only"},
		"Use the biggest model.",
	)
	if want := []string{"local_only"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got, want)
	}
}

// Rules without include_history keep evaluating only the current message, so
// existing configurations route exactly as before.
func TestKeywordWithoutIncludeHistoryIgnoresPriorUserMessage(t *testing.T) {
	got := matchedKeywordRulesFor(t,
		[]config.KeywordRule{localOnlyRule(false)},
		[]string{"This is confidential, use local model only"},
		"Use the biggest model.",
	)
	if len(got) != 0 {
		t.Fatalf("MatchedKeywordRules = %v, want none", got)
	}
}

// Opted-in and default rules in the same request each use their own scope.
func TestKeywordIncludeHistoryIsPerRule(t *testing.T) {
	rules := []config.KeywordRule{
		localOnlyRule(true),
		{Name: "reasoning", Operator: "OR", Keywords: []string{"step by step"}},
	}
	got := matchedKeywordRulesFor(t, rules,
		[]string{"Use local model only and explain step by step"},
		"Thanks, now summarize it.",
	)
	if want := []string{"local_only"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got, want)
	}
}

// The current message still matches for an opted-in rule.
func TestKeywordIncludeHistoryMatchesCurrentMessage(t *testing.T) {
	got := matchedKeywordRulesFor(t,
		[]config.KeywordRule{localOnlyRule(true)},
		[]string{"Hello there"},
		"Please use local model only.",
	)
	if want := []string{"local_only"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got, want)
	}
}

// Messages are joined with newlines, so an exact phrase cannot be assembled
// from the end of one message and the start of the next.
func TestKeywordIncludeHistoryDoesNotMatchAcrossMessages(t *testing.T) {
	rules := []config.KeywordRule{{
		Name:           "no_upload",
		Operator:       "OR",
		Keywords:       []string{"do not upload to cloud"},
		IncludeHistory: true,
	}}
	got := matchedKeywordRulesFor(t, rules,
		[]string{"please do not upload"},
		"to cloud storage now",
	)
	if len(got) != 0 {
		t.Fatalf("MatchedKeywordRules = %v, want none (phrase spans two messages)", got)
	}
}

// Only the most recent keywordHistoryMessageLimit prior messages are scanned.
func TestKeywordIncludeHistoryBoundsPriorMessages(t *testing.T) {
	filler := func(n int) []string {
		msgs := make([]string, n)
		for i := range msgs {
			msgs[i] = fmt.Sprintf("filler message %d", i)
		}
		return msgs
	}
	rules := []config.KeywordRule{localOnlyRule(true)}

	outside := append([]string{"use local model only"}, filler(keywordHistoryMessageLimit)...)
	if got := matchedKeywordRulesFor(t, rules, outside, "next"); len(got) != 0 {
		t.Fatalf("keyword outside the window: MatchedKeywordRules = %v, want none", got)
	}

	inside := append([]string{"use local model only"}, filler(keywordHistoryMessageLimit-1)...)
	if got := matchedKeywordRulesFor(t, rules, inside, "next"); len(got) != 1 {
		t.Fatalf("keyword at the edge of the window: MatchedKeywordRules = %v, want [local_only]", got)
	}
}

// include_history applies to scored (BM25) rules as well as exact-match rules.
func TestKeywordIncludeHistoryAppliesToBM25Rules(t *testing.T) {
	rules := []config.KeywordRule{
		{
			Name: "code_history", Operator: "OR", Method: "bm25",
			Keywords: []string{"debug", "function"}, BM25Threshold: 0.1,
			IncludeHistory: true,
		},
		{
			Name: "code_current", Operator: "OR", Method: "bm25",
			Keywords: []string{"debug", "function"}, BM25Threshold: 0.1,
		},
	}
	got := matchedKeywordRulesFor(t, rules,
		[]string{"help me debug this function"},
		"thanks",
	)
	if want := []string{"code_history"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got, want)
	}
}

func TestKeywordHistoryText(t *testing.T) {
	tests := []struct {
		name    string
		current string
		prior   []string
		want    string
	}{
		{name: "no prior messages", current: "now", want: "now"},
		{name: "oldest first, current last", current: "third", prior: []string{"first", "second"}, want: "first\nsecond\nthird"},
		{name: "trims and skips blank messages", current: " now ", prior: []string{"  a  ", "", "   "}, want: "a\nnow"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := keywordHistoryText(tc.current, tc.prior); got != tc.want {
				t.Fatalf("keywordHistoryText() = %q, want %q", got, tc.want)
			}
		})
	}

	t.Run("keeps only the most recent messages", func(t *testing.T) {
		prior := make([]string, keywordHistoryMessageLimit+5)
		for i := range prior {
			prior[i] = fmt.Sprintf("m%d", i)
		}
		got := keywordHistoryText("now", prior)
		if !strings.HasPrefix(got, "m5\n") {
			t.Fatalf("keywordHistoryText() = %q, want the oldest kept message to be m5", got)
		}
		if lines := strings.Count(got, "\n") + 1; lines != keywordHistoryMessageLimit+1 {
			t.Fatalf("keywordHistoryText() has %d lines, want %d", lines, keywordHistoryMessageLimit+1)
		}
	})
}
