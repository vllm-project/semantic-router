package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func countingKeywordClassifier(t testing.TB, rules ...config.KeywordRule) *KeywordClassifier {
	t.Helper()
	classifier, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatal(err)
	}
	return classifier
}

func bound(value float64) *float64 { return &value }

// A literal keyword counts where it is a word of its own: a letter or digit
// of a non-CJK script joins it to a longer word, a CJK neighbor does not.
func TestKeywordCountHonorsLiteralBoundaries(t *testing.T) {
	classifier := countingKeywordClassifier(t, config.KeywordRule{
		Name: "must", Operator: "OR", Keywords: []string{"must"}, Count: &config.NumericPredicate{GTE: bound(0)},
	})
	for _, test := range []struct {
		text string
		want float64
	}{
		{"must", 1},
		{"Must must", 2},
		{"mustard almostmust must2 must_", 0},
		{"émust musté ٣must", 0},
		{"必须must严格", 1},
		{"🙂must🙂", 1},
		{"\xffmust\xff", 1},
		{"must\x00must", 2},
	} {
		t.Run(fmt.Sprintf("%q", test.text), func(t *testing.T) {
			_, values, err := classifier.MatchAllWithHistory(test.text, "")
			if err != nil || values["must"] != test.want {
				t.Fatalf("count = %v (%v), want %v", values["must"], err, test.want)
			}
		})
	}
}

// The keywords rule migrate writes for a structure rule on text_bytes counts
// the UTF-8 bytes of the text, as text_bytes did, whatever its characters.
func TestMigratedTextBytesRuleCountsUTF8Bytes(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "config", "testdata", "migrate", "removed-structure-text-bytes", "after.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Recipes []struct {
			Signals struct {
				Keywords []config.KeywordRule `yaml:"keywords"`
			} `yaml:"signals"`
		} `yaml:"recipes"`
	}
	if err := yaml.Unmarshal(data, &document); err != nil || len(document.Recipes) == 0 || len(document.Recipes[0].Signals.Keywords) != 1 {
		t.Fatalf("after.yaml holds no keywords rule: %v", err)
	}
	rule := document.Recipes[0].Signals.Keywords[0]
	classifier := countingKeywordClassifier(t, rule)
	for _, text := range []string{
		"", "plain ASCII text", "line one\nline two", "café crème", "必须严格", "🙂 emoji 🚀", "mixed: é 中 🙂 \t\n",
		strings.Repeat("a", 3998) + "é", // 4000 bytes in 3999 characters
		strings.Repeat("中", 1333),       // 3999 bytes
		strings.Repeat("中", 1333) + "a", // 4000 bytes in 1334 characters
	} {
		matches, values, err := classifier.MatchAllWithHistory(text, "")
		if err != nil {
			t.Fatal(err)
		}
		if got := values[rule.Name]; got != float64(len(text)) {
			t.Errorf("a %d-byte text counts %v", len(text), got)
		}
		if matched := len(matches) > 0; matched != (len(text) >= 4000) {
			t.Errorf("a %d-byte text matched %v, want %v", len(text), matched, len(text) >= 4000)
		}
	}
}

// A counting rule matches when its count, or density per 1,000 characters,
// satisfies its predicate, a count of zero included, and publishes the value
// either way; a regex keyword counts each match and a CJK keyword its
// substrings.
func TestKeywordCountAndDensityPredicates(t *testing.T) {
	classifier := countingKeywordClassifier(t,
		config.KeywordRule{Name: "many_questions", Operator: "OR", Method: "regex", Keywords: []string{`[?？]`}, Count: &config.NumericPredicate{GTE: bound(3)}},
		config.KeywordRule{Name: "at_most_one_question", Operator: "OR", Method: "regex", Keywords: []string{`[?？]`}, Count: &config.NumericPredicate{LTE: bound(1)}},
		config.KeywordRule{Name: "constraint_dense", Operator: "OR", Keywords: []string{"at most", "至少"}, Density: &config.NumericPredicate{GT: bound(20)}},
	)
	for _, test := range []struct {
		text    string
		matched []string
		values  map[string]float64
	}{
		{"Why? How？ When?", []string{"many_questions"}, map[string]float64{"many_questions": 3, "at_most_one_question": 3, "constraint_dense": 0}},
		{"Summarize this.", []string{"at_most_one_question"}, map[string]float64{"many_questions": 0, "at_most_one_question": 0, "constraint_dense": 0}},
		{"至少三点，at most 5 lines?", []string{"at_most_one_question", "constraint_dense"}, map[string]float64{"many_questions": 1, "at_most_one_question": 1, "constraint_dense": 2000.0 / 21}},
	} {
		matches, values, err := classifier.MatchAllWithHistory(test.text, "")
		if err != nil {
			t.Fatal(err)
		}
		var matched []string
		for _, match := range matches {
			matched = append(matched, match.RuleName)
		}
		if strings.Join(matched, ",") != strings.Join(test.matched, ",") {
			t.Errorf("%q matched %v, want %v", test.text, matched, test.matched)
		}
		for name, want := range test.values {
			if got := values[name]; got != want {
				t.Errorf("%q %s = %v, want %v", test.text, name, got, want)
			}
		}
	}
}

// A counting rule with include_history counts over the earlier user turns
// too, and the signal publishes its value as keyword:<rule>.
func TestKeywordCountSignalPublishesValueWithHistory(t *testing.T) {
	classifier := &Classifier{keywordClassifier: countingKeywordClassifier(t, config.KeywordRule{
		Name: "repeated_urgency", Operator: "OR", Keywords: []string{"urgent"}, IncludeHistory: true, Count: &config.NumericPredicate{GTE: bound(2)},
	})}
	results := &SignalResults{SignalValues: map[string]float64{}, Metrics: &SignalMetricsCollection{}}
	classifier.evaluateKeywordSignalWithHistory(results, &sync.Mutex{}, "still urgent", []string{"this is urgent", "hello"})
	if got := results.SignalValues["keyword:repeated_urgency"]; got != 2 {
		t.Fatalf("keyword:repeated_urgency = %v, want 2", got)
	}
	if len(results.MatchedKeywordRules) != 1 || results.MatchedKeywordRules[0] != "repeated_urgency" {
		t.Fatalf("matched = %v", results.MatchedKeywordRules)
	}
}

// The classifier refuses a count on a rule that does not locate occurrences.
func TestKeywordClassifierRefusesCountWithoutOccurrences(t *testing.T) {
	for _, rule := range []config.KeywordRule{
		{Name: "all", Operator: "AND", Keywords: []string{"a", "b"}, Count: &config.NumericPredicate{GTE: bound(2)}},
		{Name: "scored", Operator: "OR", Method: "bm25", Keywords: []string{"a"}, Density: &config.NumericPredicate{GTE: bound(1)}},
	} {
		if _, err := NewKeywordClassifier([]config.KeywordRule{rule}); err == nil {
			t.Errorf("rule %q was accepted", rule.Name)
		}
	}
}

// The built-in balance long-context probes repeatedly contain "without".
// Decoding the preceding rune must not rescan the entire growing prefix.
func BenchmarkKeywordCountLongContext(b *testing.B) {
	classifier := countingKeywordClassifier(b, config.KeywordRule{
		Name: "markers", Operator: "OR", Keywords: []string{"without", "must", "exactly"}, Count: &config.NumericPredicate{GTE: bound(1)},
	})
	for _, repetitions := range []int{1000, 10000} {
		b.Run(fmt.Sprintf("repetitions=%d", repetitions), func(b *testing.B) {
			text := strings.Repeat("Neutral long-context payload without routing intent. ", repetitions)
			b.SetBytes(int64(len(text)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, values, _ := classifier.MatchAllWithHistory(text, ""); values["markers"] != float64(repetitions) {
					b.Fatalf("count = %v, want %d", values["markers"], repetitions)
				}
			}
		})
	}
}
