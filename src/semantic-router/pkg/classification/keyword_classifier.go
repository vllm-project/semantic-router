package classification

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lexical"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultBM25Threshold  = 0.1
	defaultNgramThreshold = 0.4
	defaultNgramArity     = 3
)

// KeywordClassifier implements keyword-based classification logic.
// Each rule uses regex (the default), bm25 or ngram matching, and rules are
// evaluated in configuration order.
type KeywordClassifier struct {
	rules []keywordRule
}

// keywordRule is one configured rule with the engine that evaluates it:
// exactly one of regex and scored is set.
type keywordRule struct {
	name   string
	method string
	regex  *preppedKeywordRule
	scored scoredKeywordMatcher
}

// scoredKeywordMatcher is a BM25 or n-gram rule.
type scoredKeywordMatcher interface {
	Match(*lexical.Text) (lexical.Match, bool)
}

// NewKeywordClassifier creates a new KeywordClassifier.
func NewKeywordClassifier(cfgRules []config.KeywordRule) (*KeywordClassifier, error) {
	kc := &KeywordClassifier{rules: make([]keywordRule, 0, len(cfgRules))}
	for _, rule := range cfgRules {
		switch rule.Operator {
		case "AND", "OR", "NOR":
		default:
			return nil, fmt.Errorf("unsupported keyword rule operator: %q for rule %q", rule.Operator, rule.Name)
		}
		compiled, err := newKeywordRule(rule)
		if err != nil {
			return nil, err
		}
		kc.rules = append(kc.rules, compiled)
	}
	return kc, nil
}

func newKeywordRule(rule config.KeywordRule) (keywordRule, error) {
	method := strings.ToLower(rule.Method)
	if method == "" {
		method = "regex"
	}
	compiled := keywordRule{name: rule.Name, method: method}
	scoredRule := lexical.Rule{Name: rule.Name, Operator: rule.Operator, Keywords: rule.Keywords, CaseSensitive: rule.CaseSensitive}
	switch method {
	case "bm25":
		scoredRule.Threshold = rule.BM25Threshold
		if scoredRule.Threshold == 0 {
			scoredRule.Threshold = defaultBM25Threshold
		}
		matcher, err := lexical.NewBM25(scoredRule)
		if err != nil {
			return keywordRule{}, fmt.Errorf("failed to add BM25 rule %q: %w", rule.Name, err)
		}
		compiled.scored = matcher
		logging.Debugf("Keyword rule %q using BM25 method (threshold=%.2f, keywords=%d)",
			rule.Name, scoredRule.Threshold, len(rule.Keywords))
	case "ngram":
		scoredRule.Threshold = rule.NgramThreshold
		if scoredRule.Threshold == 0 {
			scoredRule.Threshold = defaultNgramThreshold
		}
		arity := rule.NgramArity
		if arity == 0 {
			arity = defaultNgramArity
		}
		matcher, err := lexical.NewNgram(scoredRule, arity)
		if err != nil {
			return keywordRule{}, fmt.Errorf("failed to add N-gram rule %q: %w", rule.Name, err)
		}
		compiled.scored = matcher
		logging.Debugf("Keyword rule %q using N-gram method (threshold=%.2f, arity=%d, keywords=%d)",
			rule.Name, scoredRule.Threshold, arity, len(rule.Keywords))
	case "regex":
		prepped, err := prepRegexRule(rule)
		if err != nil {
			return keywordRule{}, err
		}
		compiled.regex = &prepped
		logging.Debugf("Keyword rule %q using regex method (keywords=%d, fuzzy=%v)",
			rule.Name, len(rule.Keywords), rule.FuzzyMatch)
	default:
		return keywordRule{}, fmt.Errorf("unsupported keyword rule method: %q for rule %q (valid: regex, bm25, ngram)", rule.Method, rule.Name)
	}
	return compiled, nil
}

// Classify performs keyword-based classification on the given text.
// Returns category, confidence, and error.
func (c *KeywordClassifier) Classify(text string) (string, float64, error) {
	if c == nil {
		return "", 0.0, nil
	}
	category, _, matchCount, totalKeywords, err := c.ClassifyWithKeywordsAndCount(text)
	if err != nil || category == "" {
		return category, 0.0, err
	}

	if totalKeywords == 0 {
		return category, 1.0, nil
	}

	ratio := float64(matchCount) / float64(totalKeywords)
	confidence := 0.5 + (ratio * 0.5)

	return category, confidence, nil
}

// ClassifyWithKeywords performs keyword-based classification and returns matched keywords.
func (c *KeywordClassifier) ClassifyWithKeywords(text string) (string, []string, error) {
	category, keywords, _, _, err := c.ClassifyWithKeywordsAndCount(text)
	return category, keywords, err
}
