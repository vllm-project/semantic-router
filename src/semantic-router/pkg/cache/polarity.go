package cache

import (
	"strings"
	"unicode"
)

// Polarity guard for the semantic cache.
//
// Bi-encoder cosine similarity scores a query and its negated / antonym
// variant (opposite meaning, near-identical surface form) ABOVE the deployed
// similarity_threshold, so the cache would serve the semantically opposite
// cached answer (asking how to "disable" a feature returns the "enable"
// answer). Threshold calibration cannot separate the two classes: a genuine
// paraphrase can score lower than a semantic opposite (see #2691).
//
// polarityMismatch is a cheap, model-free lexical guard applied to a candidate
// cache hit: it rejects the hit only when the two queries are near-identical as
// TOKEN SETS AND differ by a polarity flip — either a negation cue present on
// exactly one side, or a known antonym swap. It is deliberately conservative:
// when the two queries differ by more than a couple of tokens they are not a
// polarity variant of each other, so the guard stays out of the way and lets
// the embedding decision stand (a genuine paraphrase must keep hitting).
//
// This is a token-set comparison, not a full surface-form comparison: it
// ignores word order and token repetition. That keeps it cheap and catches the
// common single-word negation/antonym flips, but it cannot see a polarity
// change expressed purely through re-ordering (e.g. shifted negation scope).
//
// Known limitations (out of scope for #2691): a token-set lexical guard does
// not catch cue-less semantic negation, ordering-only polarity changes, or
// non-English queries. That residual is where an NLI-style semantic verifier
// helps later (see the L2 discussion on #2691).

// tokenDiffLimit is the maximum number of differing tokens (summed across both
// token sets) for two queries to still count as near-identical. Negation /
// antonym variants typically differ by a single token ("not") or a single word
// swap (enable→disable). Above this the queries are treated as genuinely
// different phrasings and the guard does not fire.
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

// antonymFlip maps a token to the set of tokens that are its polarity opposite.
// The table is bidirectional (both directions are inserted from the pair list
// below). A flip fires only when one query has token X and the other has an
// antonym of X in its differing tokens, so an unpaired occurrence (e.g. "turn
// on 2FA" with no "off" anywhere) does not trigger it.
// Pairs whose common form also changes a preposition remain here: the token
// gate excludes those forms, while direct-object and same-preposition forms
// still need protection.
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

// irregularContractions handle the "n't" forms whose stem is not simply the
// word minus "n't". A naive "n't" -> " not" rule would turn "can't" into "ca
// not" and "won't" into "wo not", leaving a junk stem ("ca"/"wo") that differs
// from "can"/"will" and pushes the pair past the token-diff gate — silently
// missing the negation. can't/won't/shan't expand to their real stem + "not"
// so the stem still matches its unnegated twin. "ain't" is ambiguous (am/is/
// are/has/have not), so rather than guess a stem we drop it to a bare "not":
// the negation-cue asymmetry still fires whenever the rest of the query
// matches, which is the conservative behaviour we want.
var irregularContractions = [][2]string{
	{"can't", "can not"},
	{"won't", "will not"},
	{"shan't", "shall not"},
	{"ain't", "not"},
}

// tokenizeForPolarity lowercases, expands "n't" contractions to a standalone
// "not", and splits on any non-alphanumeric rune. It returns the set of unique
// tokens (order and repetition are irrelevant to the polarity check).
func tokenizeForPolarity(s string) map[string]struct{} {
	s = strings.ToLower(s)
	// Normalize typographic apostrophes before expanding contractions so ASCII
	// and curly-apostrophe contractions take the same negation path.
	s = strings.ReplaceAll(s, "’", "'")
	// Expand irregular contractions first so their real stem is preserved,
	// then the regular rule handles "isn't" -> "is not", "doesn't" -> "does
	// not", etc.
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

// diffTokens returns the tokens present in a but not b.
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

// polarityMismatch reports whether incoming and cached are near-identical in
// surface form but opposite in polarity, in which case a semantic-cache hit
// between them must be rejected. See the file-level comment for the rationale.
func polarityMismatch(incoming, cached string) bool {
	a := tokenizeForPolarity(incoming)
	b := tokenizeForPolarity(cached)

	onlyA := diffTokens(a, b)
	onlyB := diffTokens(b, a)

	// Token-diff gate: only near-identical queries are candidates for a
	// polarity flip. Identical token sets (no differing tokens) are not a
	// mismatch either.
	if len(onlyA)+len(onlyB) == 0 || len(onlyA)+len(onlyB) > tokenDiffLimit {
		return false
	}

	// Negation cue present on exactly one side flips polarity.
	if containsAny(onlyA, negationCues) != containsAny(onlyB, negationCues) {
		return true
	}

	// Antonym swap: a differing token on one side whose opposite appears among
	// the differing tokens on the other side.
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
