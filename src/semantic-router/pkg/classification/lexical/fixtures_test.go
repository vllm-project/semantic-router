package lexical

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
)

// The fixtures were recorded from nlp-binding (bm25 2.3.2, ngrammatic 0.7.0)
// before it was removed: 29 rule sets over 266 texts each, covering
// operators, thresholds, arities, case sensitivity, duplicate and stop-word
// keywords, typos, and non-ASCII text.
type keywordFixtures struct {
	Cases []struct {
		Name   string `json:"name"`
		Method string `json:"method"`
		Rules  []struct {
			Name          string   `json:"name"`
			Operator      string   `json:"operator"`
			Keywords      []string `json:"keywords"`
			Threshold     float32  `json:"threshold"`
			CaseSensitive bool     `json:"case_sensitive"`
			Arity         int      `json:"arity"`
		} `json:"rules"`
		Texts []struct {
			Text  string         `json:"text"`
			First *fixtureMatch  `json:"first"`
			All   []fixtureMatch `json:"all"`
		} `json:"texts"`
	} `json:"cases"`
}

type fixtureMatch struct {
	Rule          string    `json:"rule"`
	Keywords      []string  `json:"keywords"`
	Scores        []float32 `json:"scores"`
	MatchCount    int       `json:"match_count"`
	TotalKeywords int       `json:"total_keywords"`
}

type matcher interface {
	Match(*Text) (Match, bool)
}

func loadKeywordFixtures(t *testing.T) keywordFixtures {
	t.Helper()
	file, err := os.Open("testdata/keyword_binding_fixtures.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	reader, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	var fixtures keywordFixtures
	if err := json.NewDecoder(reader).Decode(&fixtures); err != nil {
		t.Fatal(err)
	}
	return fixtures
}

func TestMatchesBindingFixtures(t *testing.T) {
	fixtures := loadKeywordFixtures(t)
	compared := 0
	for _, fixture := range fixtures.Cases {
		names := make([]string, len(fixture.Rules))
		matchers := make([]matcher, len(fixture.Rules))
		for i, r := range fixture.Rules {
			rule := Rule{Name: r.Name, Operator: r.Operator, Keywords: r.Keywords, Threshold: r.Threshold, CaseSensitive: r.CaseSensitive}
			var err error
			if fixture.Method == "bm25" {
				matchers[i], err = NewBM25(rule)
			} else {
				matchers[i], err = NewNgram(rule, r.Arity)
			}
			if err != nil {
				t.Fatalf("%s: %v", fixture.Name, err)
			}
			names[i] = r.Name
		}
		for _, recorded := range fixture.Texts {
			text := NewText(recorded.Text)
			var all []fixtureMatch
			for i, m := range matchers {
				if match, ok := m.Match(text); ok {
					all = append(all, fixtureMatch{Rule: names[i], Keywords: match.Keywords, Scores: match.Scores, MatchCount: match.MatchCount, TotalKeywords: match.TotalKeywords})
				}
			}
			var first *fixtureMatch
			if len(all) > 0 {
				first = &all[0]
			}
			if got, want := canonicalMatch(first), canonicalMatch(recorded.First); got != want {
				t.Errorf("%s %q first match:\n got  %s\n want %s", fixture.Name, recorded.Text, got, want)
			}
			if got, want := canonicalMatches(all), canonicalMatches(recorded.All); got != want {
				t.Errorf("%s %q matches:\n got  %s\n want %s", fixture.Name, recorded.Text, got, want)
			}
			compared++
		}
	}
	if compared < 7000 {
		t.Fatalf("compared only %d fixture texts", compared)
	}
}

// canonicalMatch orders keyword/score pairs, since the binding reported equal
// scores in hash order. Scores compare bit for bit.
func canonicalMatch(m *fixtureMatch) string {
	if m == nil {
		return "none"
	}
	pairs := make([]string, len(m.Keywords))
	for i, keyword := range m.Keywords {
		pairs[i] = fmt.Sprintf("%s=%08x", keyword, math.Float32bits(m.Scores[i]))
	}
	sort.Strings(pairs)
	return fmt.Sprintf("%s %d/%d [%s]", m.Rule, m.MatchCount, m.TotalKeywords, strings.Join(pairs, " "))
}

func canonicalMatches(matches []fixtureMatch) string {
	parts := make([]string, len(matches))
	for i := range matches {
		parts[i] = canonicalMatch(&matches[i])
	}
	return strings.Join(parts, "; ")
}
