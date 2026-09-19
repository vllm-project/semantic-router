package cache

import (
	"strings"
	"unicode"
)

// The polarity guard compares token sets, catching common negation cues and
// antonym swaps without affecting more distant paraphrases. It cannot detect
// cue-less, word-order-only, or non-English polarity changes.

// tokenDiffLimit bounds polarity checks to near-identical token sets. A cue
// insertion differs by one token and an antonym swap by two.
const tokenDiffLimit = 2

// negationCues are tokens whose presence on exactly one side flips polarity.
// "n't" contractions are normalized to "not" before tokenization, and "cannot"
// is matched as a whole token.
var negationCues = map[string]struct{}{
	"not":     {},
	"no":      {},
	"never":   {},
	"without": {},
	"cannot":  {},
}

// antonymFlip is bidirectional. A flip requires opposite tokens in the two
// token differences, so an unpaired antonym does not trigger the guard.
var antonymFlip = buildAntonymFlip([][2]string{
	{"enable", "disable"},
	{"enabled", "disabled"},
	{"on", "off"},
	{"open", "closed"},
	{"open", "close"},
	{"start", "stop"},
	{"add", "remove"},
	{"grant", "revoke"},
	{"increase", "decrease"},
	{"active", "inactive"},
	{"forward", "back"},
	{"forward", "backward"},
})

func buildAntonymFlip(pairs [][2]string) map[string]map[string]struct{} {
	m := make(map[string]map[string]struct{}, len(pairs)*2)
	add := func(a, b string) {
		if m[a] == nil {
			m[a] = make(map[string]struct{})
		}
		m[a][b] = struct{}{}
	}
	for _, p := range pairs {
		add(p[0], p[1])
		add(p[1], p[0])
	}
	return m
}

// Preserve stems that a generic "n't" replacement would corrupt. "ain't" is
// ambiguous, so only its unambiguous negation cue is retained.
var irregularContractions = [][2]string{
	{"can't", "can not"},
	{"won't", "will not"},
	{"shan't", "shall not"},
	{"ain't", "not"},
}

// tokenizeForPolarity returns normalized unique tokens; order and repetition
// do not affect the guard.
func tokenizeForPolarity(s string) map[string]struct{} {
	s = strings.ToLower(s)
	// Normalize typographic apostrophes before expanding contractions so ASCII
	// and curly-apostrophe contractions take the same negation path.
	s = strings.ReplaceAll(s, "’", "'")
	for _, ic := range irregularContractions {
		s = strings.ReplaceAll(s, ic[0], ic[1])
	}
	s = strings.ReplaceAll(s, "n't", " not")
	set := make(map[string]struct{})
	for _, tok := range strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		set[tok] = struct{}{}
	}
	return set
}

func diffTokens(a, b map[string]struct{}) []string {
	var only []string
	for tok := range a {
		if _, ok := b[tok]; !ok {
			only = append(only, tok)
		}
	}
	return only
}

func containsAny(tokens []string, cues map[string]struct{}) bool {
	for _, tok := range tokens {
		if _, ok := cues[tok]; ok {
			return true
		}
	}
	return false
}

// polarityMismatch reports whether near-identical token sets differ in polarity.
func polarityMismatch(incoming, cached string) bool {
	a := tokenizeForPolarity(incoming)
	b := tokenizeForPolarity(cached)

	onlyA := diffTokens(a, b)
	onlyB := diffTokens(b, a)

	// Only near-identical, non-identical token sets can be polarity variants.
	if len(onlyA)+len(onlyB) == 0 || len(onlyA)+len(onlyB) > tokenDiffLimit {
		return false
	}

	if containsAny(onlyA, negationCues) != containsAny(onlyB, negationCues) {
		return true
	}

	onlyBSet := make(map[string]struct{}, len(onlyB))
	for _, tok := range onlyB {
		onlyBSet[tok] = struct{}{}
	}
	for _, tok := range onlyA {
		opposites := antonymFlip[tok]
		if opposites == nil {
			continue
		}
		for opp := range opposites {
			if _, ok := onlyBSet[opp]; ok {
				return true
			}
		}
	}

	return false
}
