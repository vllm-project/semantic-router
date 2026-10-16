package cache

import (
	"slices"
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

// tokenizeForPolarity returns sorted unique tokens backed by the caller's buffer
// when it fits. Cache entries prepare this immutable summary once on insertion;
// each lookup reuses one query summary across all candidates and fallback scans.
func tokenizeForPolarity(s string, tokens []string) []string {
	s = strings.ToLower(s)
	// Normalize typographic apostrophes before expanding contractions so ASCII
	// and curly-apostrophe contractions take the same negation path.
	s = strings.ReplaceAll(s, "’", "'")
	for _, ic := range irregularContractions {
		s = strings.ReplaceAll(s, ic[0], ic[1])
	}
	s = strings.ReplaceAll(s, "n't", " not")
	for tok := range strings.FieldsFuncSeq(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		tokens = append(tokens, tok)
	}
	slices.Sort(tokens)
	return slices.Compact(tokens)
}

// polarityMismatch reports whether near-identical token sets differ in polarity.
func polarityMismatch(incoming, cached string) bool {
	var incomingBuffer, cachedBuffer [32]string
	return polarityTokensMismatch(
		tokenizeForPolarity(incoming, incomingBuffer[:0]),
		tokenizeForPolarity(cached, cachedBuffer[:0]),
	)
}

// Sorted summaries let the bounded symmetric difference stay on the stack.
// Once more than tokenDiffLimit distinct tokens differ, the original surface
// gate cannot reject this pair, so no remaining tokens need to be visited.
func polarityTokensMismatch(incoming, cached []string) bool {
	var onlyIncoming, onlyCached [tokenDiffLimit]string
	incomingCount, cachedCount := 0, 0
	for i, j := 0, 0; i < len(incoming) || j < len(cached); {
		switch {
		case i < len(incoming) && j < len(cached) && incoming[i] == cached[j]:
			i++
			j++
		case j == len(cached) || (i < len(incoming) && incoming[i] < cached[j]):
			if incomingCount+cachedCount == tokenDiffLimit {
				return false
			}
			onlyIncoming[incomingCount] = incoming[i]
			incomingCount++
			i++
		default:
			if incomingCount+cachedCount == tokenDiffLimit {
				return false
			}
			onlyCached[cachedCount] = cached[j]
			cachedCount++
			j++
		}
	}
	if containsNegationCue(onlyIncoming[:incomingCount]) != containsNegationCue(onlyCached[:cachedCount]) {
		return true
	}
	for _, tok := range onlyIncoming[:incomingCount] {
		for _, other := range onlyCached[:cachedCount] {
			if _, ok := antonymFlip[tok][other]; ok {
				return true
			}
		}
	}
	return false
}

func containsNegationCue(tokens []string) bool {
	for _, tok := range tokens {
		if _, ok := negationCues[tok]; ok {
			return true
		}
	}
	return false
}

// negationGuardOutcome classifies a pair the guard accepted. Only a change made
// entirely of negation cues is one the English lexicon can judge; any other
// changed word may carry a negation it cannot see, such as "unsafe" or "nicht".
func negationGuardOutcome(incoming, cached []string) NegationGuardOutcome {
	differences := 0
	for i, j := 0, 0; i < len(incoming) || j < len(cached); {
		var token string
		switch {
		case i < len(incoming) && j < len(cached) && incoming[i] == cached[j]:
			i++
			j++
			continue
		case j == len(cached) || (i < len(incoming) && incoming[i] < cached[j]):
			token = incoming[i]
			i++
		default:
			token = cached[j]
			j++
		}
		differences++
		if _, cue := negationCues[token]; !cue || differences > tokenDiffLimit {
			return NegationGuardNotApplicable
		}
	}
	return NegationGuardChecked
}
