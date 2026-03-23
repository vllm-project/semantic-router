package classification

import (
	"fmt"
	"math"
	"regexp"
	"strings"
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type structureRuntimeRule struct {
	config.StructureRule
	regex *regexp.Regexp
}

type StructureClassifier struct {
	tokenCounter TokenCounter
	rules        []structureRuntimeRule
}

type StructureMatch struct {
	RuleName    string
	Value       float64
	Confidence  float64
	Description string
}

func NewStructureClassifier(
	tokenCounter TokenCounter,
	rules []config.StructureRule,
) (*StructureClassifier, error) {
	compiled := make([]structureRuntimeRule, 0, len(rules))
	for _, rule := range rules {
		runtimeRule := structureRuntimeRule{StructureRule: rule}
		if strings.EqualFold(strings.TrimSpace(rule.Feature.Source.Type), "regex") {
			pattern := rule.Feature.Source.Pattern
			if pattern == "" {
				return nil, fmt.Errorf("structure rule %q missing regex pattern", rule.Name)
			}
			if !rule.Feature.Source.CaseSensitive && !strings.HasPrefix(pattern, "(?i)") {
				pattern = "(?i)" + pattern
			}
			compiledRegex, err := regexp.Compile(pattern)
			if err != nil {
				return nil, fmt.Errorf("structure rule %q invalid regex %q: %w", rule.Name, rule.Feature.Source.Pattern, err)
			}
			runtimeRule.regex = compiledRegex
		}
		compiled = append(compiled, runtimeRule)
	}
	return &StructureClassifier{
		tokenCounter: tokenCounter,
		rules:        compiled,
	}, nil
}

func (c *StructureClassifier) Classify(text string) ([]StructureMatch, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}

	matches := make([]StructureMatch, 0, len(c.rules))
	for _, rule := range c.rules {
		value, matched := c.evaluateRule(rule, text)
		if !matched {
			continue
		}
		matches = append(matches, StructureMatch{
			RuleName:    rule.Name,
			Value:       value,
			Confidence:  structureConfidence(value, rule.Predicate),
			Description: rule.Description,
		})
	}
	return matches, nil
}

func (c *StructureClassifier) evaluateRule(rule structureRuntimeRule, text string) (float64, bool) {
	value := c.extractFeatureValue(rule, text)
	if rule.Predicate == nil {
		return value, value > 0
	}
	return value, predicateMatches(value, rule.Predicate)
}

func (c *StructureClassifier) extractFeatureValue(rule structureRuntimeRule, text string) float64 {
	featureType := strings.ToLower(strings.TrimSpace(rule.Feature.Type))
	switch featureType {
	case "exists":
		if structureSourceCount(rule, text) > 0 {
			return 1
		}
		return 0
	case "count":
		return float64(structureSourceCount(rule, text))
	case "density":
		denominator := c.normalizeBy(rule.Feature.NormalizeBy, text)
		if denominator <= 0 {
			return 0
		}
		return float64(structureSourceCount(rule, text)) / denominator
	case "sequence":
		if structureSequenceMatched(rule, text) {
			return 1
		}
		return 0
	case "span_distance":
		distance, ok := structureSpanDistance(rule, text)
		if !ok {
			return 0
		}
		denominator := c.normalizeBy(rule.Feature.NormalizeBy, text)
		if denominator > 0 {
			return distance / denominator
		}
		return distance
	default:
		return 0
	}
}

func (c *StructureClassifier) normalizeBy(normalizeBy string, text string) float64 {
	switch strings.ToLower(strings.TrimSpace(normalizeBy)) {
	case "char_count":
		return float64(len([]rune(text)))
	case "word_count":
		return float64(len(strings.Fields(text)))
	case "token_count":
		if c.tokenCounter == nil {
			return 0
		}
		count, err := c.tokenCounter.CountTokens(text)
		if err != nil {
			return 0
		}
		return float64(count)
	default:
		return 0
	}
}

func structureSourceCount(rule structureRuntimeRule, text string) int {
	sourceType := strings.ToLower(strings.TrimSpace(rule.Feature.Source.Type))
	switch sourceType {
	case "regex":
		if rule.regex == nil {
			return 0
		}
		return len(rule.regex.FindAllStringIndex(text, -1))
	case "keyword_set":
		return keywordOccurrenceCount(text, rule.Feature.Source.Keywords, rule.Feature.Source.CaseSensitive)
	case "sequence":
		if structureSequenceMatched(rule, text) {
			return 1
		}
		return 0
	case "marker_pair":
		if _, ok := structureSpanDistance(rule, text); ok {
			return 1
		}
		return 0
	default:
		return 0
	}
}

func structureSequenceMatched(rule structureRuntimeRule, text string) bool {
	if len(rule.Feature.Source.Sequences) == 0 {
		return false
	}
	caseSensitive := rule.Feature.Source.CaseSensitive
	candidate := text
	if !caseSensitive {
		candidate = strings.ToLower(text)
	}
	for _, sequence := range rule.Feature.Source.Sequences {
		if len(sequence) == 0 {
			continue
		}
		searchFrom := 0
		matched := true
		for _, marker := range sequence {
			currentMarker := marker
			if !caseSensitive {
				currentMarker = strings.ToLower(marker)
			}
			idx := strings.Index(candidate[searchFrom:], currentMarker)
			if idx < 0 {
				matched = false
				break
			}
			searchFrom += idx + len(currentMarker)
		}
		if matched {
			return true
		}
	}
	return false
}

func structureSpanDistance(rule structureRuntimeRule, text string) (float64, bool) {
	startPos, startLen, ok := earliestMarker(text, rule.Feature.Source.Start, rule.Feature.Source.CaseSensitive)
	if !ok {
		return 0, false
	}
	endPos, _, ok := earliestMarkerAfter(
		text,
		rule.Feature.Source.End,
		startPos+startLen,
		rule.Feature.Source.CaseSensitive,
	)
	if !ok {
		return 0, false
	}
	return float64(max(0, endPos-(startPos+startLen))), true
}

func earliestMarker(text string, markers []string, caseSensitive bool) (int, int, bool) {
	bestIdx := -1
	bestLen := 0
	candidate := text
	if !caseSensitive {
		candidate = strings.ToLower(text)
	}
	for _, marker := range markers {
		currentMarker := marker
		if !caseSensitive {
			currentMarker = strings.ToLower(marker)
		}
		idx := strings.Index(candidate, currentMarker)
		if idx < 0 {
			continue
		}
		if bestIdx == -1 || idx < bestIdx {
			bestIdx = idx
			bestLen = len(currentMarker)
		}
	}
	return bestIdx, bestLen, bestIdx >= 0
}

func earliestMarkerAfter(text string, markers []string, offset int, caseSensitive bool) (int, int, bool) {
	if offset >= len(text) {
		return 0, 0, false
	}
	idx, markerLen, ok := earliestMarker(text[offset:], markers, caseSensitive)
	if !ok {
		return 0, 0, false
	}
	return idx + offset, markerLen, true
}

func keywordOccurrenceCount(text string, keywords []string, caseSensitive bool) int {
	if len(keywords) == 0 {
		return 0
	}
	candidate := text
	if !caseSensitive {
		candidate = strings.ToLower(text)
	}
	total := 0
	for _, keyword := range keywords {
		currentKeyword := keyword
		if !caseSensitive {
			currentKeyword = strings.ToLower(keyword)
		}
		total += countKeywordOccurrences(candidate, currentKeyword)
	}
	return total
}

func countKeywordOccurrences(text string, keyword string) int {
	if keyword == "" {
		return 0
	}
	count := 0
	searchFrom := 0
	for {
		idx := strings.Index(text[searchFrom:], keyword)
		if idx < 0 {
			return count
		}
		absoluteIdx := searchFrom + idx
		if keywordBoundaryMatch(text, absoluteIdx, absoluteIdx+len(keyword)) {
			count++
		}
		searchFrom = absoluteIdx + len(keyword)
	}
}

func keywordBoundaryMatch(text string, start int, end int) bool {
	if start > 0 {
		if prev, _ := utf8DecodeLastRuneInString(text[:start]); unicode.IsLetter(prev) || unicode.IsDigit(prev) {
			return false
		}
	}
	if end < len(text) {
		if next, _ := utf8DecodeRuneInString(text[end:]); unicode.IsLetter(next) || unicode.IsDigit(next) {
			return false
		}
	}
	return true
}

func predicateMatches(value float64, predicate *config.NumericPredicate) bool {
	if predicate == nil {
		return value > 0
	}
	if predicate.GT != nil && !(value > *predicate.GT) {
		return false
	}
	if predicate.GTE != nil && !(value >= *predicate.GTE) {
		return false
	}
	if predicate.LT != nil && !(value < *predicate.LT) {
		return false
	}
	if predicate.LTE != nil && !(value <= *predicate.LTE) {
		return false
	}
	return true
}

func structureConfidence(value float64, predicate *config.NumericPredicate) float64 {
	if predicate == nil {
		if value > 0 {
			return 1.0
		}
		return 0
	}
	if !predicateMatches(value, predicate) {
		return 0
	}
	distance := 1.0
	switch {
	case predicate.GT != nil:
		distance = math.Abs(value - *predicate.GT)
	case predicate.GTE != nil:
		distance = math.Abs(value - *predicate.GTE)
	case predicate.LT != nil:
		distance = math.Abs(*predicate.LT - value)
	case predicate.LTE != nil:
		distance = math.Abs(*predicate.LTE - value)
	}
	confidence := 1.0 / (1.0 + math.Exp(-4.0*distance))
	if confidence < 0.5 {
		return 0.5
	}
	return confidence
}

func utf8DecodeRuneInString(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return rune(0), 0
}

func utf8DecodeLastRuneInString(s string) (rune, int) {
	last := rune(0)
	size := 0
	for _, r := range s {
		last = r
		size = len(string(r))
	}
	return last, size
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
