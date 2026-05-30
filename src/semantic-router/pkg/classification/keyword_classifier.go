package classification

import (
	"fmt"
	"strings"

	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// KeywordClassifier implements keyword-based classification logic.
// It supports regex, bm25, and ngram matching methods per rule.
type KeywordClassifier struct {
	regexRules []preppedKeywordRule

	bm25Classifier  *nlp_binding.BM25Classifier
	ngramClassifier *nlp_binding.NgramClassifier

	ruleOrder []ruleRef
}

// ruleRef tracks the method and name for ordered rule evaluation.
type ruleRef struct {
	method string
	name   string
}

// NewKeywordClassifier creates a new KeywordClassifier.
// Rules with method "bm25" or "ngram" are dispatched to Rust-backed
// classifiers; all others use the regex engine.
func NewKeywordClassifier(cfgRules []config.KeywordRule) (*KeywordClassifier, error) {
	kc := &KeywordClassifier{}

	var hasBM25, hasNgram bool

	for _, rule := range cfgRules {
		switch rule.Operator {
		case "AND", "OR", "NOR":
		default:
			return nil, fmt.Errorf("unsupported keyword rule operator: %q for rule %q", rule.Operator, rule.Name)
		}

		method := strings.ToLower(rule.Method)
		if method == "" {
			method = "regex"
		}

		switch method {
		case "bm25":
			hasBM25 = true
		case "ngram":
			hasNgram = true
		case "regex":
		default:
			return nil, fmt.Errorf("unsupported keyword rule method: %q for rule %q (valid: regex, bm25, ngram)", rule.Method, rule.Name)
		}
	}

	if hasBM25 {
		kc.bm25Classifier = nlp_binding.NewBM25Classifier()
	}
	if hasNgram {
		kc.ngramClassifier = nlp_binding.NewNgramClassifier()
	}

	for _, rule := range cfgRules {
		method := strings.ToLower(rule.Method)
		if method == "" {
			method = "regex"
		}

		switch method {
		case "bm25":
			threshold := rule.BM25Threshold
			if threshold == 0 {
				threshold = 0.1
			}
			err := kc.bm25Classifier.AddRule(
				rule.Name, rule.Operator, rule.Keywords, threshold, rule.CaseSensitive,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to add BM25 rule %q: %w", rule.Name, err)
			}
			kc.ruleOrder = append(kc.ruleOrder, ruleRef{method: "bm25", name: rule.Name})
			logging.Debugf("Keyword rule %q using BM25 method (threshold=%.2f, keywords=%d)",
				rule.Name, threshold, len(rule.Keywords))

		case "ngram":
			threshold := rule.NgramThreshold
			if threshold == 0 {
				threshold = 0.4
			}
			arity := rule.NgramArity
			if arity == 0 {
				arity = 3
			}
			err := kc.ngramClassifier.AddRule(
				rule.Name, rule.Operator, rule.Keywords, threshold, rule.CaseSensitive, arity,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to add N-gram rule %q: %w", rule.Name, err)
			}
			kc.ruleOrder = append(kc.ruleOrder, ruleRef{method: "ngram", name: rule.Name})
			logging.Debugf("Keyword rule %q using N-gram method (threshold=%.2f, arity=%d, keywords=%d)",
				rule.Name, threshold, arity, len(rule.Keywords))

		case "regex":
			preppedRule, err := prepRegexRule(rule)
			if err != nil {
				return nil, err
			}
			kc.regexRules = append(kc.regexRules, preppedRule)
			kc.ruleOrder = append(kc.ruleOrder, ruleRef{method: "regex", name: rule.Name})
			logging.Debugf("Keyword rule %q using regex method (keywords=%d, fuzzy=%v)",
				rule.Name, len(rule.Keywords), rule.FuzzyMatch)
		}
	}

	return kc, nil
}

// Free releases Rust-side resources. Call when the classifier is no longer needed.
func (c *KeywordClassifier) Free() {
	if c.bm25Classifier != nil {
		c.bm25Classifier.Free()
	}
	if c.ngramClassifier != nil {
		c.ngramClassifier.Free()
	}
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
