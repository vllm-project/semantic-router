package classification

import (
	"fmt"
	"sort"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// RuleSpans returns where the named rule's keywords occur in text, in code
// points and labeled with the rule's name: each keyword's places as matching
// finds them, a literal keyword only where it is a word of its own. Only a
// rule that finds text (config.KeywordRuleFindsText) has places. A match of
// no text, such as a pattern of only `^`, covers nothing and is left out.
func (c *KeywordClassifier) RuleSpans(name, text string) ([]modelservice.Span, error) {
	var rule *keywordRule
	if c != nil {
		for i := range c.rules {
			if c.rules[i].name == name {
				rule = &c.rules[i]
				break
			}
		}
	}
	if rule == nil {
		return nil, fmt.Errorf("keyword rule %q is not declared in this recipe", name)
	}
	if rule.regex == nil || rule.measure != nil || rule.regex.Operator != "OR" || rule.regex.FuzzyMatch {
		return nil, fmt.Errorf("keyword rule %q does not locate its matches", name)
	}
	var places [][]int
	for index, pattern := range regexpsForRule(*rule.regex) {
		for _, place := range keywordPlaces(text, *rule.regex, index, pattern) {
			if place[0] < place[1] {
				places = append(places, place)
			}
		}
	}
	sort.Slice(places, func(i, j int) bool {
		if places[i][0] != places[j][0] {
			return places[i][0] < places[j][0]
		}
		return places[i][1] > places[j][1]
	})
	spans := make([]modelservice.Span, 0, len(places))
	at, point := 0, 0
	for _, place := range places {
		// places are sorted by start, so the code point count only moves on.
		point += utf8.RuneCountInString(text[at:place[0]])
		at = place[0]
		spans = append(spans, modelservice.Span{
			Label: name, Start: point, End: point + utf8.RuneCountInString(text[place[0]:place[1]]),
			Text: text[place[0]:place[1]], Probability: 1,
		})
	}
	return spans, nil
}
