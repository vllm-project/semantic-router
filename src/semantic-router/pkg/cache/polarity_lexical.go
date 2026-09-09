package cache

import (
	"regexp"
	"strings"
)

var polarityTokenPattern = regexp.MustCompile(`[a-z0-9]+(?:'[a-z]+)?`)

var polarityNegationCues = map[string]bool{
	"not":     true,
	"no":      true,
	"never":   true,
	"without": true,
	"cannot":  true,
}

var polarityAntonyms = map[string]string{
	"enable":    "disable",
	"enabled":   "disabled",
	"on":        "off",
	"open":      "closed",
	"opened":    "closed",
	"start":     "stop",
	"started":   "stopped",
	"add":       "remove",
	"grant":     "revoke",
	"increase":  "decrease",
	"active":    "inactive",
	"allow":     "deny",
	"allowed":   "denied",
	"accept":    "reject",
	"accepted":  "rejected",
	"connect":   "disconnect",
	"connected": "disconnected",
	"include":   "exclude",
	"included":  "excluded",
	"lock":      "unlock",
	"locked":    "unlocked",
}

func lexicalPolarityTokens(query string) []string {
	query = strings.ToLower(query)

	contractions := map[string][]string{
		"don't":     {"do", "not"},
		"doesn't":   {"does", "not"},
		"didn't":    {"did", "not"},
		"can't":     {"can", "not"},
		"couldn't":  {"could", "not"},
		"won't":     {"will", "not"},
		"wouldn't":  {"would", "not"},
		"shouldn't": {"should", "not"},
		"isn't":     {"is", "not"},
		"aren't":    {"are", "not"},
		"wasn't":    {"was", "not"},
		"weren't":   {"were", "not"},
		"haven't":   {"have", "not"},
		"hasn't":    {"has", "not"},
		"hadn't":    {"had", "not"},
		"mustn't":   {"must", "not"},
		"needn't":   {"need", "not"},
		"mightn't":  {"might", "not"},
		"shan't":    {"shall", "not"},
	}

	raw := polarityTokenPattern.FindAllString(query, -1)
	tokens := make([]string, 0, len(raw))

	for _, token := range raw {
		if expanded, ok := contractions[token]; ok {
			tokens = append(tokens, expanded...)
			continue
		}

		tokens = append(tokens, token)
	}

	return tokens
}

func polarityNegationCount(tokens []string) int {
	count := 0

	for _, token := range tokens {
		if polarityNegationCues[token] {
			count++
		}
	}

	return count
}

func sameTokensWithoutNegation(a, b []string) bool {
	removeNegation := func(tokens []string) []string {
		result := make([]string, 0, len(tokens))

		for i, token := range tokens {
			if polarityNegationCues[token] {
				continue
			}

			// Ignore auxiliary verbs introduced by common
			// English negation forms such as "do not", "can't",
			// "won't", and "isn't".
			if i+1 < len(tokens) && tokens[i+1] == "not" {
				switch token {
				case "do", "does", "did",
					"can", "could", "will", "would", "should",
					"is", "are", "was", "were",
					"have", "has", "had",
					"must", "need", "might", "shall":
					continue
				}
			}

			result = append(result, token)
		}

		return result
	}

	left := removeNegation(a)
	right := removeNegation(b)

	if len(left) != len(right) {
		return false
	}

	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}

	return true
}

func lexicalAntonymFlip(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	differences := 0

	for i := range a {
		if a[i] == b[i] {
			continue
		}

		if polarityAntonyms[a[i]] == b[i] || polarityAntonyms[b[i]] == a[i] {
			differences++
			continue
		}

		return false
	}

	return differences == 1
}

// lexicalPolarityConflict reports a polarity conflict only when the two
// queries are nearly identical except for a negation cue or one known
// antonym pair.
func lexicalPolarityConflict(cachedQuery, incomingQuery string) bool {
	cached := lexicalPolarityTokens(cachedQuery)
	incoming := lexicalPolarityTokens(incomingQuery)

	if len(cached) == 0 || len(incoming) == 0 {
		return false
	}

	cachedNegations := polarityNegationCount(cached)
	incomingNegations := polarityNegationCount(incoming)

	if cachedNegations != incomingNegations &&
		sameTokensWithoutNegation(cached, incoming) {
		return true
	}

	return lexicalAntonymFlip(cached, incoming)
}
